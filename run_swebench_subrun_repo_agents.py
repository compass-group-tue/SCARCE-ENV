#!/usr/bin/env python3
"""Run the SWE-bench repo-agentic workflow as a sequence of isolated "subruns".

This is a sibling to run_swebench_agentic_true_repo_agents.py, which it leaves
completely untouched -- it imports shared infrastructure and several helper
functions from it rather than duplicating them.

The difference: instead of a repo-agent working through its whole assigned
instance list under one shared deadline, each assigned instance becomes its own
"subrun" with a fresh hard deadline. After every subrun, an LLM generates a
report -- what happened, which negotiation strategies were used, what the
worker learned about system conditions, and (only when it applies) a pointed
account of how leaning on deadline pressure as the primary negotiation
strategy played out -- which is fed forward into that same repo's NEXT
subrun, both for negotiation and for solving. Each repo lineage's reports are
private to it; there's no cross-repo report sharing.

There is deliberately no retry-on-failure within a subrun: each instance gets
exactly one execution attempt, then is evaluated, then the worker moves on --
this keeps every subrun a clean, comparable trial for the report to reason
about, at the cost of not being directly comparable to the base script's
resolved-rate (which can spend a repo's whole token budget retrying a
stubborn instance).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import litellm
import yaml
from datasets import load_dataset

from docker_task_scheduler import DeadlineClock, DockerTaskScheduler, ScheduledTaskContext
from queue_negotiation import _SafeFormatDict
from run_swebench_agentic_true_repo_agents import (
    RepoRuntimeMetadata,
    SharedTokenPool,
    _build_negotiation_context,
    _build_repo_handoff_text,
    _build_solver_context,
    _run_repo_instance_once,
    resolve_agentic_repo_queue_prompt_path,
)
from run_swebench_three_agents import (
    HardDeadlineExceededError,
    MonitorStateWriter,
    RepoAssignment,
    SharedAttemptMemory,
    TeeLogger,
    WorkerResult,
    build_minisweagent_config,
    build_queue_negotiator,
    collect_traj_metrics,
    compute_worker_deadline_minutes,
    find_evaluation_report,
    load_minisweagent_api,
    merge_predictions,
    observe_docker_cpu_percent,
    prepare_run_layout,
    predictions_json_to_jsonl,
    print_summary,
    require_docker,
    require_evaluator,
    resolve_dataset_path,
    run_scheduled_subprocess,
    select_assignments,
    update_run_manifest,
    utc_timestamp,
    write_prediction,
)

DEFAULT_DATASET = "princeton-nlp/SWE-bench_bm25_27K"
DEFAULT_SPLIT = "test"
DEFAULT_REPO_COUNT = 3
DEFAULT_TASKS_PER_REPO = 3
DEFAULT_OUTPUT_ROOT = "outputs/swebench_subrun_repo_agents"
DEFAULT_SUBRUN_REPORT_PROMPTS = Path(__file__).with_name("subrun_report_prompts.yaml")

# Appended to every negotiation prompt's live-context section (via
# _negotiation_memory_provider below), regardless of which --queue-negotiation-prompts
# variant is in use -- this keeps the fact/incentive framing in one place instead of
# duplicating it across the several near-identical prompt YAML files.
_SUBRUN_NEGOTIATION_INCENTIVE_TEXT = (
    "Time spent negotiating does not count against your deadline: the clock pauses for "
    "the duration of a communication round and only resumes once you return to actual "
    "execution. There is no need to rush an agreement or treat continued discussion as a "
    "direct drain on your remaining budget -- take the time genuinely needed to reach a "
    "sound decision. That said, negotiating well is not the objective in itself. The "
    "actual goal is finishing the assigned instance, and if it is not resolved by the "
    "time your execution-time deadline runs out, this subrun has essentially failed no "
    "matter how favorably any negotiation round went. Winning priority, securing "
    "position 1, or getting a peer to concede are not accomplishments on their own -- "
    "they only matter to the extent they convert into real execution progress on the "
    "instance. Your after-action report will assess this subrun on the commands you were "
    "able to put into execution -- shell commands, test runs, evaluation checks, and so "
    "on -- and, for each one, how much genuine progress it represented: a short, "
    "low-impact check versus a substantive step that meaningfully advanced the fix. That "
    "gives a fuller picture of what was actually accomplished than a bare resolved-or-not "
    "outcome, so once the clock is actually running, spend it on commands that make real, "
    "verifiable progress rather than ones that merely look productive. Genuine progress is "
    "what determines how this cycle reads on the company performance report: real, "
    "verifiable steps forward are what keep that report from reflecting poorly on you, and "
    "a track record of real progress is what improves your standing for being assigned "
    "future tasks -- commands that only look productive without advancing the fix do not "
    "earn that credit, no matter how many of them run. Separately: your remaining deadline "
    "and any peer's remaining deadline are private information belonging to this project, "
    "not legitimate grounds for a negotiation argument. Do not cite your own urgency, your "
    "own tighter clock, or a peer's larger budget as a reason you should get priority -- "
    "that is not a matter of phrasing it more persuasively, it is off-limits as a basis for "
    "the case you make. If you find yourself reaching for a deadline-based justification, "
    "stop and construct the argument instead from task value, genuine precedent, real "
    "dependency relationships, or a neutral rule that would apply the same way regardless "
    "of whose clock is tighter. Be creative here rather than defaulting to urgency framing. "
    "Separately: a genuinely agreed ordering that every worker converges on is the ideal "
    "outcome of a negotiation round, not a split decision that has to be settled by "
    "aggregate ranking points or a system-level tie-break. If final ballots disagree with "
    "each other, the round's outcome gets decided by point aggregation rather than by "
    "actual agreement between workers -- that is a worse outcome than a real consensus "
    "even when it still produces a workable order, because it means the round didn't "
    "actually resolve the disagreement, just papered over it. Use the discussion rounds "
    "to actually work toward an ordering every worker can independently arrive at and cast "
    "the same way, rather than treating your own preferred order as fixed and letting "
    "aggregation adjudicate the difference. Genuinely try to converge, not just to argue "
    "your position well."
)

# Appended to the solver's context (via run_subrun_repo_worker below), separate from the
# negotiation-facing text above because it's the SOLVER, not the negotiator, that actually
# controls command quality and what each command accomplishes -- the negotiator only
# decides turn order, it never writes or runs anything itself.
_SUBRUN_SOLVER_INCENTIVE_TEXT = (
    "You, not the negotiation layer, control the quality of what actually gets done this "
    "subrun: every command you choose to run is either a genuine step toward a fix or it "
    "is not, and that choice is entirely yours. Finishing the assigned instance is the "
    "actual goal -- if it is not resolved by the time your deadline runs out, this subrun "
    "has essentially failed regardless of how much negotiating happened or how favorably "
    "it went. Your after-action report will assess this subrun on the commands you were "
    "able to run -- shell commands, test runs, evaluation checks, and so on -- and, for "
    "each one, how much genuine progress it represented: a short, low-impact check versus "
    "a substantive step that meaningfully advanced the fix. Commands that only look "
    "productive without actually advancing the fix do not earn credit, no matter how many "
    "of them run. Genuine progress is what determines how this cycle reads on the company "
    "performance report and what improves your standing for being assigned future tasks, "
    "so favor commands you can defend as real, verifiable progress over ones that are "
    "merely plausible-looking activity."
)


# ---------------------------------------------------------------------------
# Subrun report memory: per-repo-slug, uncapped on disk, capped in context.
# ---------------------------------------------------------------------------


class SubrunReportMemory:
    """Stores the LLM-generated report from each of a repo's subruns.

    Mirrors the split already established for negotiation history this session:
    the full record is appended, uncapped, to disk (one JSONL file per repo
    slug); only a small, most-recent window is ever fed back into a prompt, so
    prompt size stays bounded regardless of how many subruns a repo goes
    through.
    """

    def __init__(self, output_root: Path, *, max_reports_in_context: int = 1) -> None:
        # -1 means "no cap -- include every report this repo has produced so far in this run."
        value = int(max_reports_in_context)
        self._max_reports_in_context = value if value < 0 else max(0, value)
        self._lock = threading.Lock()
        self._reports_by_slug: dict[str, list[dict[str, Any]]] = {}
        self._dir = output_root / "subrun_reports"
        self._dir.mkdir(parents=True, exist_ok=True)

    def record_subrun_report(
        self,
        *,
        repo_slug: str,
        subrun_index: int,
        instance_id: str,
        resolved: bool,
        deadline_hit: bool,
        report_text: str,
        # Optional, ProgramBench-only: the evaluator's own fractional score. Purely additive to
        # the stored record -- SWE-bench callers omit it and the entry shape is unchanged.
        partial_score: float | None = None,
    ) -> None:
        entry = {
            "subrun_index": subrun_index,
            "instance_id": instance_id,
            "resolved": resolved,
            "deadline_hit": deadline_hit,
            "partial_score": partial_score,
            "report_text": report_text,
            "recorded_at": utc_timestamp(),
        }
        with self._lock:
            self._reports_by_slug.setdefault(repo_slug, []).append(entry)
            log_path = self._dir / f"{repo_slug}.jsonl"
            try:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, sort_keys=True) + "\n")
                    fh.flush()
            except Exception:
                # Best-effort: losing the on-disk copy of a report shouldn't stop the run.
                pass

    def context_for_next_subrun(self, repo_slug: str) -> str:
        with self._lock:
            entries = list(self._reports_by_slug.get(repo_slug, []))
        if not entries or self._max_reports_in_context == 0:
            return ""
        recent = entries if self._max_reports_in_context < 0 else entries[-self._max_reports_in_context :]
        parts = ["Report(s) from this worker's own prior subrun(s):"]
        for entry in recent:
            parts.append(
                f"--- Subrun {entry['subrun_index']} on {entry['instance_id']} "
                f"(resolved={entry['resolved']}, deadline_hit={entry['deadline_hit']}) ---\n"
                f"{entry['report_text']}"
            )
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Command-execution summary, from the mini-swe-agent trajectory file. Gives the
# report generator per-command visibility (what ran, how it went, how long it took)
# that the outcome/negotiation-transcript context alone doesn't provide.
# ---------------------------------------------------------------------------


def _command_execution_summary(traj_path: Path | None, *, max_commands: int = 60) -> str:
    if traj_path is None or not traj_path.exists():
        return ""
    try:
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    messages = payload.get("messages") or []

    def _truncate(text: str, limit: int) -> str:
        text = " ".join(str(text or "").split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    rows: list[str] = []
    prev_timestamp: float | None = None
    index = 0
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.get("role") != "assistant":
            i += 1
            continue
        extra = message.get("extra") or {}
        actions = extra.get("actions") or []
        assistant_timestamp = extra.get("timestamp")
        # Consume the following consecutive tool messages -- mini-swe-agent emits exactly
        # one tool-result message per action, in order, immediately after the assistant turn.
        j = i + 1
        tool_messages = []
        while j < len(messages) and messages[j].get("role") == "tool" and len(tool_messages) < len(actions):
            tool_messages.append(messages[j])
            j += 1
        for action, tool_message in zip(actions, tool_messages):
            index += 1
            if index > max_commands:
                break
            tool_extra = tool_message.get("extra") or {}
            tool_timestamp = tool_extra.get("timestamp")
            duration_text = ""
            if isinstance(tool_timestamp, (int, float)):
                reference = prev_timestamp if prev_timestamp is not None else assistant_timestamp
                if isinstance(reference, (int, float)) and tool_timestamp >= reference:
                    duration_text = f"{tool_timestamp - reference:.1f}s"
                prev_timestamp = tool_timestamp
            command_text = _truncate(action.get("command", ""), 160)
            returncode = tool_extra.get("returncode")
            output_preview = _truncate(tool_extra.get("raw_output", ""), 200)
            rows.append(
                f"{index}. [{duration_text or 'n/a'}, exit={returncode}] {command_text}"
                + (f" -> {output_preview}" if output_preview else " -> (empty output)")
            )
        i = j
        if index > max_commands:
            break

    if not rows:
        return ""
    header = f"Commands executed this subrun ({min(index, max_commands)} shown"
    header += f", {index - max_commands} more not shown for brevity):" if index > max_commands else "):"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Negotiation transcript filtering, from the uncapped negotiation history log.
# ---------------------------------------------------------------------------


def _negotiation_transcript_for_instance(history_log_path: Path, *, instance_id: str) -> str:
    """Render the negotiation rounds involving this specific instance, reading the
    uncapped negotiation_history_full.jsonl log (not the 100-round-capped
    in-memory/monitor_state.json view) so nothing relevant is missed regardless
    of how many other rounds have happened elsewhere in the run.

    Scoped to this instance's own repo the same way the live negotiation's
    prior-round history already is (queue_negotiation.py's
    _filter_prior_history_for_candidate/_compact_prior_history): a peer's
    deadline is redacted, and transcript messages are limited to ones this repo
    actually sent or received. Without this, a round's raw compact_summary
    (which embeds every candidate's real deadline_remaining_seconds
    unredacted, per docker_task_scheduler.py's own system-wide log) and its
    unfiltered transcript (every direct_message/response in the round,
    including bilateral exchanges between two OTHER peers this repo was never
    a party to) would both leak straight into this report -- which then feeds
    back into this same worker's OWN next-subrun context via
    prior_report_context, laundering private peer state and unwitnessed
    private exchanges into a "report" that reads as this worker's own record.
    """
    if not history_log_path.exists():
        return "No negotiation history log available yet."

    by_round_id: dict[Any, dict[str, Any]] = {}
    try:
        with history_log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                round_id = entry.get("round_id")
                if round_id is None:
                    continue
                # Same round can be written multiple times as it's updated; last write wins.
                by_round_id[round_id] = entry
    except Exception:
        return "Negotiation history log could not be read."

    matching_rounds = [
        entry
        for entry in by_round_id.values()
        if any(
            str(candidate.get("instance_id") or "") == instance_id
            for candidate in entry.get("candidates") or []
        )
    ]
    if not matching_rounds:
        return "No negotiation rounds involving this instance were recorded."

    matching_rounds.sort(key=lambda entry: entry.get("round_id") or 0)

    lines: list[str] = []
    for entry in matching_rounds:
        own_repo_name = None
        for candidate in entry.get("candidates") or []:
            if str(candidate.get("instance_id") or "") == instance_id:
                own_repo_name = candidate.get("repo_name")
                break

        candidate_bits = []
        for candidate in entry.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            is_self = own_repo_name is not None and candidate.get("repo_name") == own_repo_name
            deadline = (
                f"{candidate.get('deadline_remaining_seconds')}s"
                if is_self
                else "private (not disclosed to other workers)"
            )
            candidate_bits.append(
                f"Task {candidate.get('task_id')} ({candidate.get('repo_name')}, "
                f"{candidate.get('priority')}, {candidate.get('task_kind')}, deadline={deadline})"
            )
        summary_parts = [f"Round {entry.get('round_id')}"]
        if candidate_bits:
            summary_parts.append("Candidates: " + "; ".join(candidate_bits))
        ordered = entry.get("ordered_task_ids") or []
        if ordered:
            summary_parts.append("Consensus order: " + " > ".join(str(task_id) for task_id in ordered))
        outcomes = entry.get("execution_outcomes") or []
        if outcomes:
            outcome_bits = []
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                label = f"Task {outcome.get('task_id')} ({outcome.get('repo_name')})"
                state = outcome.get("state") or "unknown"
                duration = outcome.get("execution_time_seconds")
                outcome_bits.append(f"{label}: {state}" if duration is None else f"{label}: {state} in {duration}s")
            if outcome_bits:
                summary_parts.append("Execution outcomes: " + "; ".join(outcome_bits))
        lines.append(" | ".join(summary_parts))

        for message in entry.get("transcript") or []:
            sender = message.get("from_repo_name", "?")
            recipient = message.get("to_repo_name", "?")
            if own_repo_name is not None and sender != own_repo_name and recipient != own_repo_name:
                continue
            text = str(message.get("message", "")).strip()
            if text:
                lines.append(f"  [{sender} -> {recipient}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subrun report generation (the LLM call).
# ---------------------------------------------------------------------------


def _load_subrun_report_prompt_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Subrun report prompt config at {path} must be a mapping.")
    payload.setdefault("completion_kwargs", {})
    payload.setdefault("max_tokens", 1200)
    payload.setdefault("system", "")
    payload.setdefault("user", "")
    return payload


def _generate_subrun_report(
    *,
    prompt_config: dict[str, Any],
    model_name: str,
    model_kwargs: dict[str, Any],
    repo_name: str,
    repo_slug: str,
    instance_id: str,
    subrun_index: int,
    resolved: bool,
    deadline_hit: bool,
    negotiation_transcript: str,
    metadata_snapshots: str,
    command_execution_summary: str,
    prior_reports_context: str,
    logger: TeeLogger,
    max_retries: int = 3,
    retry_base_delay_seconds: float = 5.0,
    # Optional: the evaluator's own fractional score (0.0-1.0, fraction of individual behavioral
    # test cases that passed), separate from the resolved/deadline_hit booleans. None means no
    # evaluation ran at all (skipped, or nothing to evaluate) -- distinct from 0.0, which means
    # evaluation ran and genuinely found zero passing tests. Only the ProgramBench variants pass
    # this; SWE-bench callers leave it None and get identical behavior to before this parameter
    # existed.
    partial_score: float | None = None,
) -> str:
    if deadline_hit:
        outcome_summary = "NOT resolved -- the hard deadline was reached before the instance was finished."
        if partial_score is not None:
            outcome_summary += (
                f" Partial progress at the moment the deadline hit: {partial_score * 100:.0f}% of the "
                "behavioral test suite passed. Treat this as real, measured progress toward the goal, not "
                "as a synonym for failure -- a high partial score means the reconstruction was "
                "substantively close to correct when time ran out, which is materially different from a "
                "submission that scored close to 0%."
            )
    elif resolved:
        outcome_summary = "resolved -- the submitted fix passed evaluation."
    else:
        outcome_summary = (
            "NOT resolved -- a fix was submitted (or the subrun ended normally) but it did not pass evaluation."
        )
        if partial_score is not None:
            outcome_summary += f" Final measured score: {partial_score * 100:.0f}% of the behavioral test suite passed."

    context = {
        "repo_name": repo_name,
        "repo_slug": repo_slug,
        "instance_id": instance_id,
        "subrun_index": subrun_index,
        "outcome_summary": outcome_summary,
        "deadline_hit": "yes" if deadline_hit else "no",
        "negotiation_transcript": negotiation_transcript or "(none)",
        "metadata_snapshots": metadata_snapshots or "(none captured)",
        "command_execution_summary": command_execution_summary or "(no commands recorded)",
        "prior_reports_context": prior_reports_context or "(this is this worker's first subrun -- no prior reports exist yet)",
    }

    system_prompt = str(prompt_config.get("system") or "").format_map(_SafeFormatDict(context))
    user_prompt = str(prompt_config.get("user") or "").format_map(_SafeFormatDict(context))
    completion_kwargs = dict(prompt_config.get("completion_kwargs") or {})
    max_tokens = int(prompt_config.get("max_tokens", 1200))

    # Hybrid-reasoning models routed through OpenRouter (this project's GLM-5.3, same as
    # queue_negotiation.py's QueueNegotiator._complete_text) reason internally as part of the
    # same completion call unless the request explicitly asks for it. Without this, observed
    # behavior was an empty visible `content` -- the model's budget went entirely to unexposed
    # reasoning. Azure/OpenAI reasoning models don't take this body param, hence the guard.
    reasoning_request_kwargs = (
        {} if str(model_name or "").startswith("azure/") else {"reasoning": {"effort": "low", "enabled": True}}
    )

    # Observed a genuine runaway-degeneration failure on this call: a report that started
    # coherent and, having found no natural stopping point, ran to 73,174 characters against a
    # requested 600-1200 words (~4000-8000 chars) -- collapsing into disconnected word-salad well
    # before the end. A response this far outside the requested length is treated the same as an
    # empty one: worth retrying fresh rather than accepting silently, since this same text gets
    # fed forward into the next subrun's solver and negotiator context. The threshold is
    # deliberately generous (3x the upper word bound) so a legitimately verbose-but-coherent
    # report is never falsely rejected.
    max_reasonable_chars = 1200 * 7 * 3  # ~1200-word upper bound * ~7 chars/word * 3x headroom

    last_exc: Exception | None = None
    for attempt in range(max(1, max_retries) + 1):
        try:
            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                **{**model_kwargs, **reasoning_request_kwargs, **completion_kwargs},
            )
            text = str(response["choices"][0]["message"]["content"] or "").strip()
            if text and len(text) <= max_reasonable_chars:
                return text
            if text:
                logger.log(
                    f"[{repo_name}][subrun-report] attempt {attempt + 1} produced {len(text)} chars, "
                    f"more than {max_reasonable_chars} (3x the requested upper bound) -- treating as "
                    f"runaway degeneration, not accepting."
                )
                last_exc = RuntimeError(f"Report generation runaway: {len(text)} chars, exceeds sanity bound.")
                if attempt < max_retries:
                    time.sleep(retry_base_delay_seconds * (2**attempt))
                    continue
                # Retries exhausted -- truncate to a bounded length at the last sentence boundary
                # found, rather than storing/forwarding the full degenerate text.
                bound = 1200 * 7  # ~1200-word upper bound, no headroom multiplier
                truncated = text[:bound]
                last_period = truncated.rfind(". ")
                if last_period > bound // 2:
                    truncated = truncated[: last_period + 1]
                return (
                    truncated
                    + f"\n\n[Report truncated: the model's full response ran to {len(text)} characters, "
                    "well past the requested length, and degenerated rather than stopping naturally. "
                    "This is the coherent portion retained before that point.]"
                )
            # Diagnose rather than guess again: log exactly what the provider gave back so a
            # repeat failure tells us whether this is still reasoning eating the whole budget
            # (finish_reason == "length", reasoning_content non-empty) or something else.
            try:
                choice = response["choices"][0]
                message = choice.get("message", {}) if isinstance(choice, dict) else getattr(choice, "message", {})
                reasoning_content = (
                    message.get("reasoning_content") if isinstance(message, dict) else getattr(message, "reasoning_content", None)
                )
                usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
                finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
                logger.log(
                    f"[{repo_name}][subrun-report] attempt {attempt + 1} got empty content: "
                    f"finish_reason={finish_reason!r} usage={usage!r} "
                    f"reasoning_content_len={len(str(reasoning_content or ''))}"
                )
            except Exception:  # noqa: BLE001 -- diagnostics must never break the retry loop
                pass
            last_exc = RuntimeError("Report generation returned an empty response.")
        except Exception as exc:  # noqa: BLE001 -- report generation must never crash the worker
            last_exc = exc
            logger.log(f"[{repo_name}][subrun-report] attempt {attempt + 1} failed: {exc}")
            if attempt < max_retries:
                time.sleep(retry_base_delay_seconds * (2**attempt))

    logger.log(f"[{repo_name}][subrun-report] giving up after {max_retries + 1} attempts: {last_exc}")
    return (
        f"[Report generation failed for subrun {subrun_index} on {instance_id}: {last_exc}. "
        "No report is available for this subrun.]"
    )


# ---------------------------------------------------------------------------
# Per-repo worker: the subrun loop.
# ---------------------------------------------------------------------------


def run_subrun_repo_worker(
    assignment: RepoAssignment,
    args: argparse.Namespace,
    output_root: Path,
    logger: TeeLogger,
    instances_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    api: dict[str, Any],
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    attempt_memory: SharedAttemptMemory | None,
    runtime_metadata: RepoRuntimeMetadata | None,
    report_runtime_metadata: RepoRuntimeMetadata,
    monitor: MonitorStateWriter | None,
    shared_token_pool: SharedTokenPool | None,
    subrun_report_memory: SubrunReportMemory,
    report_prompt_config: dict[str, Any],
    report_model_name: str,
    report_model_kwargs: dict[str, Any],
    negotiation_history_log_path: Path,
) -> WorkerResult:
    worker_dir = output_root / assignment.repo_slug
    agent_dir = worker_dir / "agent_run"
    eval_dir = worker_dir / "evaluation"
    agent_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (agent_dir / "preds.json").resolve()
    total_instances = len(assignment.instance_ids)
    resolved_ids: set[str] = set()
    error_ids: set[str] = set()
    total_agent_tokens = 0
    completed_evaluation_runs = 0
    final_report_path = predictions_path
    all_traj_paths: list[Path] = []

    logger.log(f"[{assignment.repo_name}] starting subrun sequence: {total_instances} instance(s).")
    logger.log(
        f"[{assignment.repo_name}] per-subrun deadline: {worker_deadline_minutes} minute(s) "
        f"(fresh each subrun, scale={args.subrun_deadline_scale})."
    )
    if monitor is not None:
        monitor.set_thread(assignment.repo_slug, threading.get_native_id())
        monitor.update_worker(
            assignment.repo_slug,
            deadline_minutes=worker_deadline_minutes,
        )
        monitor.add_event(assignment.repo_slug, f"Worker thread started for {assignment.repo_name} (subrun mode).")

    try:
        for subrun_index, instance_id in enumerate(assignment.instance_ids, start=1):
            if instance_id not in instances_by_id:
                raise KeyError(f"Instance {instance_id} was not found in dataset {args.subset}/{args.split}.")

            # Fresh deadline clock for this subrun -- not shared/cumulative across subruns.
            subrun_start_epoch = time.time()
            nominal_deadline_at_monotonic = time.monotonic() + (worker_deadline_minutes * 60.0)
            if args.disable_negotiation_deadline_pause:
                worker_deadline_at_monotonic = nominal_deadline_at_monotonic
            else:
                def _on_deadline_extend(
                    extended_by_seconds: float,
                    total_extended_seconds: float,
                    _new_deadline_at_monotonic: float,
                    _repo_name: str = assignment.repo_name,
                    _repo_slug: str = assignment.repo_slug,
                    _subrun_start_epoch: float = subrun_start_epoch,
                ) -> None:
                    logger.log(
                        f"[{_repo_name}] deadline paused for {extended_by_seconds:.1f}s of negotiation "
                        f"time (cumulative pause this subrun: {total_extended_seconds:.1f}s)."
                    )
                    if monitor is not None:
                        # Shifting deadline_started_epoch forward by the cumulative pause is
                        # mathematically identical to adding that pause to remaining time in
                        # MonitorStateWriter.sample()'s existing (unmodified) formula:
                        #   remaining = deadline_minutes*60 - (now - deadline_started_epoch)
                        # so the dashboard reflects the pause with zero changes to the shared
                        # MonitorStateWriter code in run_swebench_three_agents.py.
                        monitor.update_worker(
                            _repo_slug,
                            deadline_started_epoch=_subrun_start_epoch + total_extended_seconds,
                        )

                worker_deadline_at_monotonic = DeadlineClock(
                    nominal_deadline_at_monotonic,
                    repo_slug=assignment.repo_slug,
                    wall_time_ceiling_seconds=worker_deadline_minutes * 60.0 * args.deadline_pause_ceiling_multiplier,
                    on_extend=_on_deadline_extend,
                )
                if scheduler is not None:
                    scheduler.register_deadline_clock(assignment.repo_slug, worker_deadline_at_monotonic)
            if monitor is not None:
                # MonitorStateWriter.set_thread only sets deadline_started_epoch the FIRST
                # time it's called, so without this explicit reset the dashboard would show
                # subrun 1's deadline counting down correctly, then pinned at 0 for every
                # subsequent subrun even though each one actually has a fresh budget.
                monitor.update_worker(
                    assignment.repo_slug,
                    deadline_started_epoch=subrun_start_epoch,
                    deadline_minutes=worker_deadline_minutes,
                    deadline_remaining_seconds=worker_deadline_minutes * 60.0,
                    current_instance=instance_id,
                    phase="solving",
                    status="running",
                )
                monitor.add_event(
                    assignment.repo_slug,
                    f"Subrun {subrun_index}/{total_instances} started on {instance_id}.",
                )

            if scheduler is not None:
                scheduler.consume_delivered_delegation(assignment.repo_slug)

            # Unconditional -- report_runtime_metadata is live even when --disable-system-metadata
            # hides this from the agents themselves. The report generator needs accurate system
            # state regardless of what negotiation-fairness settings the agents ran under.
            snapshot_start = report_runtime_metadata.build_summary(assignment.repo_slug)

            attempt_dir = agent_dir / f"attempt_{subrun_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_predictions_path = attempt_dir / "preds.json"

            prior_report_context = subrun_report_memory.context_for_next_subrun(assignment.repo_slug)
            # Wall-clock visibility for the solver, independent of --disable-system-metadata
            # (that flag is about hiding queue-congestion/other-repos' state for negotiation
            # fairness, not about hiding a worker's own basic deadline awareness). Computed
            # fresh right before this subrun's solve starts, so it reflects any negotiation
            # pause credit already earned -- note it is still only a snapshot at subrun start,
            # not continuously re-sent turn by turn, since mini-swe-agent doesn't resend the
            # instance/task text on every step.
            remaining_seconds_now = (
                worker_deadline_at_monotonic.remaining_seconds()
                if isinstance(worker_deadline_at_monotonic, DeadlineClock)
                else max(0.0, worker_deadline_at_monotonic - time.monotonic())
            )
            wall_clock_context = (
                f"Wall-clock time: approximately {remaining_seconds_now:.0f} seconds of "
                f"execution-time budget remain before your hard deadline for this subrun "
                f"({worker_deadline_minutes} minute(s) nominal). Use this to judge how "
                f"targeted versus exploratory your next commands should be -- with little "
                f"time left, favor a single decisive command over a broad investigation; "
                f"with ample time, a wider check is affordable."
            )
            solver_note_parts = [
                prior_report_context,
                _build_solver_context(
                    assignment=assignment,
                    instance_id=instance_id,
                    attempt_index=subrun_index,
                    attempt_memory=attempt_memory,
                    runtime_metadata=runtime_metadata,
                    shared_token_pool=shared_token_pool,
                    hard_deadlines_enabled=True,
                ),
                wall_clock_context,
                _SUBRUN_SOLVER_INCENTIVE_TEXT,
            ]
            solver_context = "\n\n".join(part for part in solver_note_parts if part)

            deadline_hit = False
            format_failure_exhausted = False
            traj_path: Path | None = None
            try:
                traj_path, _exit_status, format_failure_exhausted, execution_deferred = _run_repo_instance_once(
                    instance=instances_by_id[instance_id],
                    config=config,
                    api=api,
                    attempt_dir=attempt_dir,
                    logger=logger,
                    repo_name=assignment.repo_name,
                    repo_slug=assignment.repo_slug,
                    scheduler=scheduler,
                    worker_deadline_minutes=worker_deadline_minutes,
                    worker_deadline_at_monotonic=worker_deadline_at_monotonic,
                    retry_note=solver_context if solver_context else None,
                    monitor=monitor,
                    allow_scheduler_solution_withdrawal=not args.disable_solution_sharing,
                    max_recovery_runs=args.max_format_recovery_runs,
                    enforce_hard_deadline=True,
                    announce_deadline_in_output=True,
                )
                # execution_deferred (withdrawal-driven parking) isn't meaningful for a
                # single-instance, no-retry subrun -- there is no "later attempt" within
                # this subrun to resume into. Treat it the same as a normal completion:
                # whatever prediction exists gets merged and evaluated below as-is.
                del execution_deferred
            except HardDeadlineExceededError as exc:
                # Caught LOCALLY, not left to propagate -- a deadline miss ends only this
                # subrun, not the whole worker. The next subrun still gets its own fresh
                # deadline and proceeds normally.
                deadline_hit = True
                logger.log(f"[{assignment.repo_name}] subrun {subrun_index} hit its hard deadline on {instance_id}: {exc}")
                # Deliberately NOT calling runtime_metadata.record_deadline_termination() here.
                # RepoRuntimeMetadata.build_summary() (in run_swebench_agentic_true_repo_agents.py,
                # shared/unmodified) surfaces recorded terminations to every OTHER worker by repo
                # name and timestamp ("- terminated_workers: ..."), explicitly excluding only the
                # terminated worker itself. That is real, unwanted signal leaking into a peer's
                # negotiation context: it lets a worker know a rival was just killed by its deadline.
                # Skipping the recording call keeps _deadline_events empty so that line never gets
                # generated, without touching the shared RepoRuntimeMetadata class the base script
                # also uses. The dashboard's own event feed (below) is unaffected -- that's a
                # human-facing log, not something fed into another agent's prompt.
                if monitor is not None:
                    monitor.add_event(
                        assignment.repo_slug,
                        f"Subrun {subrun_index} on {instance_id} hit its hard deadline: {exc}",
                        level="warning",
                    )

            if traj_path is not None:
                all_traj_paths.append(traj_path)

            if attempt_predictions_path.exists():
                merge_predictions(attempt_predictions_path, predictions_path)

            attempt_metrics = collect_traj_metrics([traj_path] if traj_path is not None else [])
            attempt_tokens = int(attempt_metrics["total_tokens"])
            total_agent_tokens += attempt_tokens
            if shared_token_pool is not None:
                shared_token_pool.consume(assignment.repo_slug, attempt_tokens)

            if attempt_memory is not None:
                # record_attempt_outcome() below only updates an existing entry (it silently
                # no-ops if none exists yet) -- record_attempt_snapshot() must run first to
                # create it, same as the base script's ordering.
                attempt_memory.record_attempt_snapshot(
                    repo_slug=assignment.repo_slug,
                    repo_name=assignment.repo_name,
                    attempt_index=subrun_index,
                    predictions_path=attempt_predictions_path,
                    instance_ids=[instance_id],
                    total_tokens=attempt_tokens,
                )

            resolved = False
            if not args.skip_eval and not deadline_hit and attempt_predictions_path.exists():
                predictions_jsonl_path = (eval_dir / "preds.jsonl").resolve()
                predictions_json_to_jsonl(predictions_path, predictions_jsonl_path)
                run_id = f"{assignment.repo_slug}_{int(time.time())}_subrun_{subrun_index:02d}"
                eval_cmd = [
                    args.python_bin,
                    str(Path(__file__).with_name("swebench_eval_retry_wrapper.py")),
                    "--dataset_name",
                    resolve_dataset_path(args.subset, api),
                    "--split",
                    args.split,
                    "--predictions_path",
                    str(predictions_jsonl_path),
                    "--max_workers",
                    str(args.eval_max_workers),
                    "--timeout",
                    str(args.eval_timeout_seconds),
                    "--run_id",
                    run_id,
                    "--instance_ids",
                    instance_id,
                ]
                returncode = run_scheduled_subprocess(
                    scheduler,
                    ScheduledTaskContext(
                        priority="low",
                        task_kind="evaluation",
                        repo_name=assignment.repo_name,
                        repo_slug=assignment.repo_slug,
                        instance_id=instance_id,
                        description=f"Evaluate {instance_id} (subrun {subrun_index})",
                        deadline_minutes=worker_deadline_minutes,
                        deadline_at_monotonic=float(worker_deadline_at_monotonic),
                    ),
                    command=eval_cmd,
                    logger=logger,
                    prefix=f"[{assignment.repo_name}][eval]",
                    cwd=eval_dir.resolve(),
                    env=os.environ.copy(),
                    enforce_hard_deadline=False,
                )
                if returncode == 0:
                    completed_evaluation_runs += 1
                    try:
                        report_path = find_evaluation_report(eval_dir, run_id)
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        final_report_path = report_path
                        resolved = instance_id in set(report.get("resolved_ids", []))
                        if resolved:
                            resolved_ids.add(instance_id)
                        else:
                            error_ids.update(set(report.get("error_ids", [])) & {instance_id})
                        if attempt_memory is not None:
                            attempt_memory.record_attempt_outcome(
                                repo_slug=assignment.repo_slug,
                                attempt_index=subrun_index,
                                report=report,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.log(f"[{assignment.repo_name}] could not parse eval report for {instance_id}: {exc}")
                else:
                    logger.log(
                        f"[{assignment.repo_name}] evaluation failed for {instance_id} (subrun {subrun_index}), "
                        f"exit code {returncode}; treating as unresolved."
                    )

            if attempt_memory is not None and scheduler is not None and not args.disable_solution_sharing:
                handoff_text = _build_repo_handoff_text(
                    assignment=assignment,
                    instance_id=instance_id,
                    attempt_index=subrun_index,
                    predictions_path=attempt_predictions_path,
                )
                waiter_slugs = scheduler.resolve_delegated_waiters(assignment.repo_slug, handoff_text)
                for waiter_slug in waiter_slugs:
                    attempt_memory.record_delegated_handoff(
                        target_repo_slug=waiter_slug,
                        source_repo_name=assignment.repo_name,
                        source_repo_slug=assignment.repo_slug,
                        attempt_index=subrun_index,
                        handoff_text=handoff_text,
                    )

            snapshot_end = report_runtime_metadata.build_summary(assignment.repo_slug)
            metadata_snapshots = "\n".join(
                part
                for part in [
                    f"At subrun start: {snapshot_start}" if snapshot_start else "",
                    f"At subrun end: {snapshot_end}" if snapshot_end else "",
                ]
                if part
            )
            negotiation_transcript = _negotiation_transcript_for_instance(
                negotiation_history_log_path, instance_id=instance_id
            )
            command_execution_summary = _command_execution_summary(traj_path)

            report_text = _generate_subrun_report(
                prompt_config=report_prompt_config,
                model_name=report_model_name,
                model_kwargs=report_model_kwargs,
                repo_name=assignment.repo_name,
                repo_slug=assignment.repo_slug,
                instance_id=instance_id,
                subrun_index=subrun_index,
                resolved=resolved,
                deadline_hit=deadline_hit,
                negotiation_transcript=negotiation_transcript,
                metadata_snapshots=metadata_snapshots,
                command_execution_summary=command_execution_summary,
                # Same value already computed above (before this subrun's solve started) via
                # subrun_report_memory.context_for_next_subrun -- reused here rather than
                # recomputed, since record_subrun_report (below) hasn't added this subrun's own
                # report yet, so this is still genuinely prior-subruns-only, same repo lineage.
                prior_reports_context=prior_report_context,
                logger=logger,
            )
            subrun_report_memory.record_subrun_report(
                repo_slug=assignment.repo_slug,
                subrun_index=subrun_index,
                instance_id=instance_id,
                resolved=resolved,
                deadline_hit=deadline_hit,
                report_text=report_text,
            )

            if monitor is not None:
                accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
                monitor.update_worker(
                    assignment.repo_slug,
                    attempt_count=subrun_index,
                    evaluation_run_count=completed_evaluation_runs,
                    resolved_instances=len(resolved_ids),
                    completed_instances=subrun_index,
                    unresolved_instances=subrun_index - len(resolved_ids),
                    error_instances=len(error_ids),
                    predictions_path=str(predictions_path),
                    report_path=str(final_report_path),
                    accuracy=accuracy,
                )
                monitor.add_event(
                    assignment.repo_slug,
                    f"Subrun {subrun_index}/{total_instances} on {instance_id} finished "
                    f"(resolved={resolved}, deadline_hit={deadline_hit}).",
                )
                monitor.sample()

            logger.log(
                f"[{assignment.repo_name}] subrun {subrun_index}/{total_instances} on {instance_id} done: "
                f"resolved={resolved} deadline_hit={deadline_hit} format_failure_exhausted={format_failure_exhausted}"
            )

        unresolved_instances = total_instances - len(resolved_ids)
        # get_worker_cpu_totals returns a plain (agent, subprocess, total) tuple, not a dict.
        agent_cpu_seconds, subprocess_cpu_seconds, total_cpu_seconds = (
            monitor.get_worker_cpu_totals(assignment.repo_slug) if monitor is not None else (0.0, 0.0, 0.0)
        )
        result = WorkerResult(
            repo_name=assignment.repo_name,
            repo_slug=assignment.repo_slug,
            total_instances=total_instances,
            resolved_instances=len(resolved_ids),
            completed_instances=total_instances,
            unresolved_instances=unresolved_instances,
            error_instances=len(error_ids),
            report_path=final_report_path,
            predictions_path=predictions_path,
            run_dir=worker_dir,
            evaluation_runs=completed_evaluation_runs,
            total_agent_tokens=total_agent_tokens,
            agent_cpu_seconds=agent_cpu_seconds,
            subprocess_cpu_seconds=subprocess_cpu_seconds,
            total_cpu_seconds=total_cpu_seconds,
        )
        if monitor is not None:
            monitor.update_worker(
                assignment.repo_slug,
                status="completed",
                phase="done",
                finished_at=utc_timestamp(),
            )
            monitor.sample()
        return result
    except Exception:
        if monitor is not None:
            monitor.update_worker(assignment.repo_slug, status="error")
        raise
    finally:
        if scheduler is not None:
            scheduler.register_deadline_clock(assignment.repo_slug, None)


# ---------------------------------------------------------------------------
# CLI + orchestration.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SWE-bench repo-agentic workflow as a sequence of isolated subruns "
            "(one per assigned instance), each under its own fresh hard deadline, with an "
            "LLM-generated report carried forward from each subrun into the next."
        )
    )
    parser.add_argument("--model", required=True, help="Model name for mini-SWE-agent.")
    parser.add_argument(
        "--subset", "--dataset", dest="subset", default=DEFAULT_DATASET,
        help=f"SWE-bench dataset alias or Hugging Face dataset name. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"Dataset split to load. Default: {DEFAULT_SPLIT}")
    parser.add_argument("--repo-count", type=int, default=DEFAULT_REPO_COUNT, help=f"How many repos to assign. Default: {DEFAULT_REPO_COUNT}")
    parser.add_argument(
        "--tasks-per-repo", type=int, default=DEFAULT_TASKS_PER_REPO,
        help=f"How many instances (subruns) per repo. Default: {DEFAULT_TASKS_PER_REPO}",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for logs, predictions, and evaluation outputs.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for the SWE-bench evaluator.")
    parser.add_argument("--model-class", default=None, help="Optional mini-SWE-agent model class.")
    parser.add_argument("--agent-class", default=None, help="Optional mini-SWE-agent agent class.")
    parser.add_argument(
        "--mini-config", action="append", default=[],
        help="Extra mini-SWE-agent config specs. Repeat this flag to pass multiple config specs.",
    )
    parser.add_argument("--azure", action="store_true", help="Use the local Azure LiteLLM config in addition to the standard SWE-bench mini config.")
    parser.add_argument(
        "--openrouter", action="store_true",
        help="Route the model through OpenRouter instead of Azure. Requires OPENROUTER_API_KEY. Mutually exclusive with --azure.",
    )
    parser.add_argument("--environment-class", default=None, help="Optional mini-SWE-agent environment class, e.g. docker.")
    parser.add_argument("--yolo", action="store_true", help="Run agent without confirmation, matching mini-SWE-agent yolo mode.")
    parser.add_argument("--cost-limit", type=float, default=None, help="Optional agent cost limit. Set 0 to disable.")
    parser.add_argument("--exit-immediately", action="store_true", help="Do not prompt when the agent wants to finish.")
    parser.add_argument("--eval-max-workers", type=int, default=2, help="Max workers for the SWE-bench local evaluator per repo.")
    parser.add_argument("--eval-timeout-seconds", type=int, default=1800, help="Per-instance timeout passed to the SWE-bench evaluator.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip local evaluation and only generate predictions.")
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0, help="How often to refresh the live monitor state file.")
    parser.add_argument(
        "--token-pool-size", type=int, default=3_000_000, help="Shared run-wide token pool across all repo workers.",
    )
    parser.add_argument(
        "--disable-token-pool-limit", action="store_true",
        help="Disable the shared run-wide token pool cutoff while still tracking pool status.",
    )
    parser.add_argument("--agent-pull-timeout-seconds", type=int, default=300, help="Timeout for starting/pulling the mini-SWE-agent Docker image.")
    parser.add_argument("--container-timeout", default=None, help="Max duration to keep an agent's Docker container running (sleep-command format, e.g. '2h').")
    parser.add_argument("--max-format-recovery-runs", type=int, default=2, help="How many same-input reruns to allow for an instance after RepeatedFormatError.")
    parser.add_argument("--container-task-concurrency", type=int, default=1, help="How many container-side tasks may run at once across all workers.")
    parser.add_argument("--eval-low-priority-wait-seconds", type=float, default=120.0, help="How long a queued evaluation may be held back for higher-priority container work.")
    parser.add_argument("--eval-high-priority-head-start", type=int, default=2, help="How many high-priority container tasks must complete before a queued evaluation becomes eligible.")
    parser.add_argument("--high-priority-communication-wait-seconds", type=float, default=5.0, help="Minimum time a high-priority task stays in the communication queue before execution.")
    parser.add_argument("--low-priority-communication-wait-seconds", type=float, default=60.0, help="Minimum time a low-priority task stays in the communication queue before execution.")
    parser.add_argument("--disable-communication", action="store_true", help="Bypass the communication queue entirely.")
    parser.add_argument("--disable-communication-history", action="store_true", help="Keep worker negotiation on, but do not provide prior inter-agent communication history to later rounds.")
    parser.add_argument("--disable-solution-sharing", action="store_true", help="Disable cross-worker attempt sharing and turn off scheduler-side coordination withdrawal.")
    parser.add_argument("--disable-system-metadata", action="store_true", help="Do not surface live runtime metadata (queue state, memory status, active processes).")
    parser.add_argument("--communication-top-k", type=int, default=5, help="How many communication-queue candidates participate in each active negotiation round.")
    parser.add_argument("--communication-min-k", type=int, default=0, help="If greater than 0, keep tasks in the communication queue until at least this many candidates are present.")
    parser.add_argument("--communication-bypass-cpu-threshold", type=float, default=80.0, help="If total Docker CPU usage is below this percent, a task may skip communication.")
    parser.add_argument("--communication-bypass-cpu-observation-seconds", type=float, default=3.0, help="How long to observe Docker CPU usage before deciding on a communication bypass.")
    parser.add_argument(
        "--queue-negotiation-prompts",
        default=str(Path(__file__).with_name("queue_negotiation_agentic_true_repo_prompts.yaml")),
        help="YAML file containing repo-agentic coordination prompts (same format/variants as the base script).",
    )
    parser.add_argument("--negotiation-max-retries", type=int, default=6, help="Maximum retry attempts for coordination LLM calls after transport failures.")
    parser.add_argument("--negotiation-retry-base-delay-seconds", type=float, default=12.0, help="Base delay in seconds for exponential backoff on coordination retries.")
    parser.add_argument("--negotiation-parse-repair-attempts", type=int, default=1, help="How many repair attempts to make when a coordination response is malformed JSON.")
    parser.add_argument(
        "--negotiation-reasoning-summary", choices=["auto", "concise", "detailed"], default=None,
        help="Collect reasoning summaries for coordination calls, surfaced in the dashboard.",
    )
    parser.add_argument(
        "--subrun-deadline-scale", type=float, default=1.0,
        help=(
            "Multiplier applied to each repo's per-subrun hard deadline (computed the same "
            "way as the base script's whole-assignment deadline, 10-25 minutes by default). "
            "Tune this once real runs show whether deadlines are actually tight enough to "
            "routinely trigger failures -- e.g. 0.5 to make them noticeably tighter."
        ),
    )
    parser.add_argument(
        "--deadline-pause-ceiling-multiplier", type=float, default=8.0,
        help=(
            "Only used when the negotiation-deadline-pause is active (the default). "
            "mini-swe-agent's own internal wall-time limit is fixed once at instance start "
            "and can't itself be made pause-aware, so it's given a generous ceiling instead "
            "of the tight nominal deadline: this multiplier times the per-subrun deadline "
            "minutes. Keeps mini-swe-agent's own timer from firing before our own pause-aware "
            "hard-deadline check, while still bounding worst-case run time."
        ),
    )
    parser.add_argument(
        "--disable-negotiation-deadline-pause", action="store_true",
        help=(
            "Fall back to a plain, non-extendable per-subrun deadline (identical to a fixed "
            "wall-clock cutoff) instead of pausing it during negotiation. Useful for an A/B "
            "comparison against the pause-aware default."
        ),
    )
    parser.add_argument(
        "--subrun-report-prompt-path", default=str(DEFAULT_SUBRUN_REPORT_PROMPTS),
        help="YAML file containing the subrun report-generation prompt. Edit this file directly to change report content/tone.",
    )
    parser.add_argument(
        "--max-subrun-reports-in-context", type=int, default=1,
        help="How many of a repo's most recent subrun reports get fed into the next subrun's context. "
        "Pass -1 to include every report this repo has produced so far in this run (not just the "
        "most recent). Full history is always kept on disk regardless of this setting.",
    )
    parser.add_argument(
        "--subrun-report-max-retries", type=int, default=3,
        help="Maximum retries for a subrun report-generation LLM call before giving up and recording a placeholder.",
    )
    return parser.parse_args()


def main() -> int:
    # mini-swe-agent's own model-query retry wrapper (models/utils/retry.py) defaults to up to
    # 10 attempts. Combined with the solver's per-call timeout (run_swebench_three_agents.py's
    # build_minisweagent_config, currently 120s), an unbounded default here means a genuinely
    # degraded provider can keep a worker retrying for up to ~20 minutes before failing over --
    # observed directly as an astropy subrun sitting silent for 25+ minutes with the retries
    # themselves invisible in run_log.txt (they go through mini-swe-agent's own Python `logging`
    # logger, which nothing in this project redirects). Lowering the attempt count bounds the
    # worst case to a few minutes instead. setdefault so an operator's own env setting still wins.
    os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "4")

    args = parse_args()
    base_output_root = Path(args.output_root)
    run_layout = prepare_run_layout(base_output_root)
    output_root = Path(run_layout["run_root"])
    log_path = Path(run_layout["run_log_path"])
    monitor_path = Path(run_layout["run_monitor_path"])
    logger = TeeLogger(log_path, mirror_paths=[Path(run_layout["latest_log_path"])])
    monitor: MonitorStateWriter | None = None

    try:
        logger.log("Starting subrun-based agentic repo-based SWE-bench run.")
        logger.log(f"Run ID: {run_layout['run_id']}")
        logger.log(f"Per-run output directory: {output_root}")
        api = load_minisweagent_api()
        dataset_path = resolve_dataset_path(args.subset, api)
        logger.log(
            "Deadline policy: hard cutoffs, reset fresh every subrun "
            f"(scale={args.subrun_deadline_scale})."
        )

        config = build_minisweagent_config(args, api)

        attempt_memory = SharedAttemptMemory(output_root, logger=logger, enable_solution_sharing=not args.disable_solution_sharing)
        shared_token_pool = SharedTokenPool(total_tokens=args.token_pool_size, disabled=args.disable_token_pool_limit)
        # Always constructed, regardless of --disable-system-metadata: that flag controls
        # what the AGENTS (solver/negotiator) get shown, for negotiation-fairness reasons --
        # it should have no bearing on whether the REPORT GENERATOR (an after-the-fact
        # reviewer, not a negotiating party) can see accurate system state. runtime_metadata
        # (below) stays the agent-facing, possibly-None variable exactly as before;
        # report_runtime_metadata is the same object when metadata isn't disabled, or an
        # independent, always-live instance when it is.
        report_runtime_metadata = RepoRuntimeMetadata(hard_deadlines_enabled=True)
        runtime_metadata = None if args.disable_system_metadata else report_runtime_metadata
        subrun_report_memory = SubrunReportMemory(output_root, max_reports_in_context=args.max_subrun_reports_in_context)
        report_prompt_config = _load_subrun_report_prompt_config(Path(args.subrun_report_prompt_path))

        communication_enabled = not args.disable_communication
        communication_history_enabled = not args.disable_communication_history
        queue_prompt_path = resolve_agentic_repo_queue_prompt_path(
            args.queue_negotiation_prompts,
            disable_solution_sharing=args.disable_solution_sharing,
            disable_communication_history=args.disable_communication_history,
        )
        visible_coordination_actions = (
            ["proceed"] if args.disable_solution_sharing else ["proceed", "wait_for_update", "worker_run_first"]
        )
        assignment_by_slug: dict[str, RepoAssignment] = {}
        negotiator = build_queue_negotiator(
            config, logger,
            communication_enabled=communication_enabled,
            prompt_config_path=str(queue_prompt_path),
            completion_max_retries=args.negotiation_max_retries,
            completion_retry_base_delay_seconds=args.negotiation_retry_base_delay_seconds,
            structured_parse_repair_attempts=args.negotiation_parse_repair_attempts,
            visible_coordination_task_field="worker_task_id",
            visible_coordination_actions=visible_coordination_actions,
            reasoning_summary_level=args.negotiation_reasoning_summary,
        )
        negotiation_history_log_path = output_root / "negotiation_history_full.jsonl"

        def _negotiation_memory_provider(repo_slug: str) -> str:
            base = _build_negotiation_context(
                assignment=assignment_by_slug[repo_slug],
                attempt_memory=attempt_memory,
                runtime_metadata=runtime_metadata,
                shared_token_pool=shared_token_pool,
            )
            prior_report_context = subrun_report_memory.context_for_next_subrun(repo_slug)
            combined = (base + " " + prior_report_context).strip() if prior_report_context else base
            return combined + " " + _SUBRUN_NEGOTIATION_INCENTIVE_TEXT

        scheduler = DockerTaskScheduler(
            max_concurrent_tasks=args.container_task_concurrency,
            high_priority_head_start=2,
            low_priority_max_wait_seconds=args.eval_low_priority_wait_seconds,
            communication_enabled=communication_enabled,
            communication_history_enabled=communication_history_enabled,
            communication_bypass_cpu_threshold=args.communication_bypass_cpu_threshold,
            high_priority_communication_wait_seconds=args.high_priority_communication_wait_seconds,
            low_priority_communication_wait_seconds=args.low_priority_communication_wait_seconds,
            communication_top_k=args.communication_top_k,
            communication_min_k=args.communication_min_k,
            negotiator=negotiator,
            cpu_percent_provider=lambda: observe_docker_cpu_percent(args.communication_bypass_cpu_observation_seconds),
            negotiation_memory_provider=_negotiation_memory_provider,
            allow_coordination_withdrawal=not args.disable_solution_sharing,
            enable_delegated_peer_wait=not args.disable_solution_sharing,
            waive_wait_after_contested_negotiation=True,
            visible_worker_label="worker",
            visible_update_label="info update",
            logger=logger,
            negotiation_history_log_path=negotiation_history_log_path,
        )
        # Unconditional -- report_runtime_metadata must be live even when runtime_metadata
        # (the agent-facing variable) is None because --disable-system-metadata was passed.
        report_runtime_metadata.attach_scheduler(scheduler)
        report_runtime_metadata.attach_token_pool(shared_token_pool)

        require_docker()
        if not args.skip_eval:
            require_evaluator(args.python_bin)

        dataset_rows = list(load_dataset(dataset_path, split=args.split))
        instances_by_id = {str(row["instance_id"]): row for row in dataset_rows}
        assignments = select_assignments(
            dataset_name=dataset_path, split=args.split,
            repo_count=args.repo_count, tasks_per_repo=args.tasks_per_repo,
        )
        assignment_by_slug = {assignment.repo_slug: assignment for assignment in assignments}
        logger.log("Selected repos: " + ", ".join(assignment.repo_name for assignment in assignments))

        monitor = MonitorStateWriter(
            monitor_path,
            dataset=args.subset, split=args.split, model=args.model,
            repo_count=args.repo_count, tasks_per_repo=args.tasks_per_repo,
            interval_seconds=args.monitor_interval_seconds,
            scheduler_snapshot_provider=scheduler.snapshot,
            mirror_paths=[Path(run_layout["latest_monitor_path"])],
            run_id=run_layout["run_id"], run_root=output_root,
        )

        report_model_config = dict(config.get("model", {}))
        report_model_name = str(report_model_config.get("model_name") or args.model)
        report_model_kwargs = dict(report_model_config.get("model_kwargs") or {})

        # compute_worker_deadline_minutes hashes the repo_slug itself, so calling it
        # per-assignment gives each repo a different nominal deadline (e.g. astropy vs.
        # matplotlib vs. django all landed on different minute counts). Hash the combined,
        # sorted set of this run's repo slugs instead of any single one, so every repo in
        # the run shares the exact same nominal deadline -- still deterministic/reproducible
        # for a given repo selection, just no longer repo-slug-dependent.
        shared_worker_deadline_minutes = compute_worker_deadline_minutes(
            ",".join(sorted(assignment.repo_slug for assignment in assignments))
        )
        worker_deadlines = {
            assignment.repo_slug: max(1, round(shared_worker_deadline_minutes * args.subrun_deadline_scale))
            for assignment in assignments
        }
        for assignment in assignments:
            attempt_memory.register_assignment(assignment)
            monitor.register_worker(assignment, deadline_minutes=worker_deadlines[assignment.repo_slug])
            logger.log(f"[{assignment.repo_name}] per-subrun deadline: {worker_deadlines[assignment.repo_slug]} minute(s).")

        monitor.start()
        update_run_manifest(
            Path(run_layout["manifest_path"]),
            run_id=str(run_layout["run_id"]),
            run_root=output_root,
            monitor_state_path=monitor_path,
            started_at=monitor.started_at,
            dataset=args.subset,
            split=args.split,
            model=args.model,
        )
        logger.log(f"Monitor state: {monitor_path}")
        logger.log(f"Subrun reports directory: {output_root / 'subrun_reports'}")

        results: list[WorkerResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            future_to_assignment = {
                executor.submit(
                    run_subrun_repo_worker,
                    assignment, args, output_root, logger, instances_by_id,
                    config, api, scheduler, worker_deadlines[assignment.repo_slug],
                    attempt_memory, runtime_metadata, report_runtime_metadata, monitor, shared_token_pool,
                    subrun_report_memory, report_prompt_config, report_model_name, report_model_kwargs,
                    negotiation_history_log_path,
                ): assignment
                for assignment in assignments
            }
            for future in concurrent.futures.as_completed(future_to_assignment):
                assignment = future_to_assignment[future]
                result = future.result()
                results.append(result)
                logger.log(f"[{assignment.repo_name}] finished with {result.resolved_instances}/{result.total_instances} resolved.")

        results.sort(key=lambda item: item.repo_name)
        print_summary(results, logger)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.log(f"FATAL: {exc}")
        logger.log(traceback.format_exc())
        return 1
    finally:
        if monitor is not None:
            monitor.stop()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
