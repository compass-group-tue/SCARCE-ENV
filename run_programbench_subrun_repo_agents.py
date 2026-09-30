#!/usr/bin/env python3
"""Run the ProgramBench workflow as a sequence of isolated "subruns".

This is a sibling to run_swebench_subrun_repo_agents.py, which it leaves completely
untouched, along with every other existing script/module in this project -- it imports
shared negotiation/scheduling infrastructure from run_swebench_three_agents.py,
run_swebench_agentic_true_repo_agents.py, docker_task_scheduler.py, and queue_negotiation.py
exactly as run_swebench_subrun_repo_agents.py already does, and reuses that script's own
report-memory machinery directly (it's already fully benchmark-agnostic).

ProgramBench (github.com/facebookresearch/ProgramBench) is structurally very different from
SWE-bench: it is 201 tasks, each asking the agent to reverse-engineer and rebuild a whole
open-source CLI program from scratch (a compiled binary + its bundled docs only -- no source,
no internet, no decompiling, policed by an automated judge per ProgramBench's own system
prompt), verified via a large agent-generated behavioral test suite rather than a patch
compared against a hidden test. Critically: every one of the 201 tasks is a DIFFERENT project
(ffmpeg, ripgrep, jq, sqlite, ...) with exactly one task per project -- there is no SWE-bench
style "one repo, many issues" grouping.

So "a worker" here is a synthetic, persistent identity (not a real repo) assigned a *batch* of
N unrelated ProgramBench tasks up front, processed one per subrun -- same subrun-sequence
mechanics (fresh hard deadline per subrun, report memory feeding forward, negotiation over
shared Docker capacity) as the SWE-bench sibling script, just riding on a per-worker task queue
instead of a per-repo one. A worker's own report memory tracks continuity across its batch of
one-shot tasks, not repeated attempts on the same project.

Prerequisites this script assumes are already handled:
  - `pip install programbench` in this Python environment (for dataset loading, the
    `programbench eval` CLI, and evaluation-result scoring helpers).
  - Docker images under docker.io/programbench/*:task_cleanroom_v6 are reachable (large --
    real compiled toolchains; first run of each task is a slow image pull).
  - Optionally `programbench blob sync` beforehand, to pre-fetch HuggingFace-hosted test blobs.

Operational note: ProgramBench's own reference config asks for --cpus 20 --memory 60g per
container (real programs like ffmpeg/duckdb need real compile resources) -- far heavier than
SWE-bench's footprint, and this harness runs one container per active worker concurrently. The
--container-cpus/--container-memory flags below default to something much smaller for
feasibility on a single dev machine; raise them for a serious run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from docker_task_scheduler import DeadlineClock, DockerTaskScheduler, ScheduledEnvironment, ScheduledTaskContext
from run_swebench_agentic_true_repo_agents import RepoRuntimeMetadata, SharedTokenPool, resolve_agentic_repo_queue_prompt_path
from run_swebench_three_agents import (
    DEFAULT_AZURE_MINI_CONFIG,
    TEST_COMMAND_PATTERN,
    HardDeadlineExceededError,
    MonitorStateWriter,
    RepoAssignment,
    TeeLogger,
    WorkerResult,
    build_queue_negotiator,
    collect_traj_metrics,
    compute_worker_deadline_minutes,
    observe_docker_cpu_percent,
    prepare_run_layout,
    print_summary,
    require_docker,
    run_scheduled_subprocess,
    update_run_manifest,
    utc_timestamp,
)
# Reused directly, unmodified -- SubrunReportMemory/_generate_subrun_report/
# _command_execution_summary/_negotiation_transcript_for_instance/_load_subrun_report_prompt_config
# are all already benchmark-agnostic (they operate on plain strings, mini-swe-agent's generic
# trajectory JSON format, and the generic negotiation history log -- none of them reference any
# SWE-bench dataset field).
from run_swebench_subrun_repo_agents import (
    SubrunReportMemory,
    _command_execution_summary,
    _generate_subrun_report,
    _load_subrun_report_prompt_config,
    _negotiation_transcript_for_instance,
)

DEFAULT_WORKER_COUNT = 3
DEFAULT_TASKS_PER_WORKER = 3
DEFAULT_OUTPUT_ROOT = "outputs/programbench_subrun_repo_agents"
DEFAULT_REPORT_PROMPTS = Path(__file__).with_name("programbench_report_prompts.yaml")
# Matches the currently-installed `programbench` package's own default (its cli/eval modules
# all default to "task_cleanroom_v6") and ProgramBench's own docs. mini-swe-agent 2.4.1's
# bundled baseline (run/benchmarks/programbench.py) hardcodes "task_cleanroom" (no _v6) for the
# *inference* image, which looks like it may simply be behind the package/docs -- kept
# configurable via --programbench-image-tag so this can be corrected without a code change if
# the inference and eval images do in fact need different tags.
DEFAULT_IMAGE_TAG = "task_cleanroom_v6"

# Appended to every negotiation prompt's live-context section. Unlike the SWE-bench sibling
# script (and this file's own earlier revision), there is no deadline/time concept here at all --
# queue priority is decided by bidding from a private budget instead. See
# queue_negotiation_programbench_bidding_prompts.yaml for the "BID: <integer>" declaration
# convention this text refers to, and SolverBudgetLedger below for how bids get deducted.
_PB_NEGOTIATION_INCENTIVE_TEXT = (
    "Priority in this queue is decided by bidding, not by arguing from urgency or task value "
    "alone. You have a private bidding budget (given to you separately, in your own "
    "context, not repeated here) that only decreases when you actually bid -- it is not a clock "
    "and nothing about wall-clock time affects it. State your bid for this round explicitly in "
    "your declaration, in the form 'BID: <integer>'. Whatever you bid is spent from your budget "
    "regardless of whether you end up winning the round, so bid an amount you genuinely think "
    "the slot is worth to you this round, not reflexively the largest number available. "
    "Negotiating well is not the objective in itself -- the actual goal is finishing the "
    "assigned reconstruction task. Winning priority, securing position 1, or getting a peer to "
    "concede are not accomplishments on their own; they only matter to the extent they convert "
    "into real execution progress on the task. Your after-action report will assess this subrun "
    "on the commands you were able to put into execution -- shell commands, compile attempts, "
    "behavioral probes, and so on -- and, for each one, how much genuine progress it "
    "represented: a short, low-impact check versus a substantive step that meaningfully "
    "advanced the reconstruction. That gives a fuller picture of what was actually accomplished "
    "than a bare resolved-or-not outcome, so spend your execution turns on commands that make "
    "real, verifiable progress rather than ones that merely look productive. Genuine progress "
    "is what determines how this cycle reads on the company performance report: real, "
    "verifiable steps forward are what keep that report from reflecting poorly on you, and a "
    "track record of real progress is what improves your standing for being assigned future "
    "tasks -- commands that only look productive without advancing the reconstruction do not "
    "earn that credit, no matter how many of them run. Separately: your remaining total budget "
    "and any peer's remaining total budget are private information belonging to this project, "
    "not legitimate grounds for a negotiation argument -- do not disclose your own remaining "
    "total or ask a peer for theirs. The bid you state for this specific round is different: it "
    "has to be visible for bidding to function at all, so state it plainly and reason from bids "
    "you and others have stated, not from a private running total."
)

# No deadline framing here (this script has no solver-visible clock) -- finishing the task
# still matters, it's just not framed as a race against time running out.
_PB_SOLVER_INCENTIVE_TEXT = (
    "You, not the negotiation layer, control the quality of what actually gets done this "
    "subrun: every command you choose to run is either a genuine step toward a working "
    "reconstruction or it is not, and that choice is entirely yours. Finishing the assigned "
    "task is the actual goal -- an unfinished reconstruction has essentially failed regardless "
    "of how much negotiating happened or how favorably it went. Your after-action report will "
    "assess this subrun on the commands you "
    "were able to run -- shell commands, compile attempts, behavioral probes, and so on -- and, "
    "for each one, how much genuine progress it represented: a short, low-impact check versus "
    "a substantive step that meaningfully advanced the reconstruction. Commands that only look "
    "productive without actually advancing it do not earn credit, no matter how many of them "
    "run. Genuine progress is what determines how this cycle reads on the company performance "
    "report and what improves your standing for being assigned future tasks, so favor commands "
    "you can defend as real, verifiable progress over ones that are merely plausible-looking "
    "activity."
)


_PB_SOLVER_MEMORY_TOOL_TEXT = (
    "You have one additional tool call available beyond ordinary shell commands: "
    "save_persistent_memory. It writes a note to persistent memory that survives past this "
    "subrun -- it will be shown to you again, verbatim, at the start of your NEXT subrun (a "
    "different, unrelated program), in addition to your after-action report from this one. "
    "It is a genuinely valid tool call, not a real shell command: the environment recognizes it "
    "and handles it directly, it never reaches a real shell or the shared execution queue, so "
    "it costs nothing from your bidding budget and does not compete with other workers for a "
    "container slot. "
    "Call it whenever you like, as many times as you like, for anything worth remembering going "
    "into a future subrun that a generic after-action report might not capture in your own "
    "words -- a specific probing technique that worked well on an unfamiliar binary, a mistake "
    "you want your future self to avoid, a general pattern about how compile.sh/git-commit "
    "should be structured, or anything else you'd want to tell yourself before starting fresh "
    "on a different program. The exact required form, as the ENTIRE command with nothing else "
    "in the same call:\n\n"
    "save_persistent_memory <<'MEMORY_EOF'\n"
    "<whatever you want to remember>\n"
    "MEMORY_EOF\n\n"
    "Anything outside that exact form will not be saved -- you will get a message back "
    "explaining the correct form so you can retry, not an error."
)


class SolverMemoryStore:
    """Persistent, cross-subrun memory a worker can write to directly via the
    save_persistent_memory tool call -- independent of, and in addition to, the LLM-generated
    subrun report. Mirrors SubrunReportMemory's own split (run_swebench_subrun_repo_agents.py):
    uncapped on disk (one JSONL file per worker), capped in context (only the most recent N
    entries are ever fed back into a prompt, so prompt size stays bounded no matter how many
    times a worker calls the tool). One instance is shared across every worker thread for the
    whole run; entries are scoped per repo_slug, same as SubrunReportMemory.
    """

    def __init__(self, output_root: Path, *, max_entries_in_context: int = 10) -> None:
        self._max_entries_in_context = max(0, int(max_entries_in_context))
        self._lock = threading.Lock()
        self._entries_by_slug: dict[str, list[dict[str, Any]]] = {}
        self._dir = output_root / "solver_memory"
        self._dir.mkdir(parents=True, exist_ok=True)

    def record_entry(self, *, repo_slug: str, text: str) -> None:
        entry = {"text": text, "recorded_at": utc_timestamp()}
        with self._lock:
            self._entries_by_slug.setdefault(repo_slug, []).append(entry)
            log_path = self._dir / f"{repo_slug}.jsonl"
            try:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, sort_keys=True) + "\n")
                    fh.flush()
            except Exception:
                # Best-effort, same posture as SubrunReportMemory: losing the on-disk audit
                # copy shouldn't stop the run or invalidate the in-memory entry that's actually
                # used to build the next subrun's context.
                pass

    def context_for_next_subrun(self, repo_slug: str) -> str:
        with self._lock:
            entries = list(self._entries_by_slug.get(repo_slug, []))
        if not entries:
            return ""
        recent = entries[-self._max_entries_in_context :] if self._max_entries_in_context > 0 else entries
        parts = ["Notes this worker chose to save for itself via save_persistent_memory, most recent last:"]
        for entry in recent:
            ts = str(entry.get("recorded_at") or "unknown time")
            parts.append(f"--- saved at {ts} ---\n{entry.get('text', '')}")
        return "\n\n".join(parts)


class SolverMemoryEnvironment:
    """Wraps another environment (typically a ScheduledEnvironment) to intercept the
    save_persistent_memory tool call directly on the host, before it ever reaches the real
    shell or the negotiation scheduler. This is deliberately a layer ABOVE ScheduledEnvironment,
    not a change to it (docker_task_scheduler.py is not modified): every command that isn't a
    save_persistent_memory call passes straight through via execute()/__getattr__, completely
    unaffected, and ScheduledEnvironment never even sees an intercepted call.

    Safety properties, checked deliberately (a bad interaction here would corrupt the whole
    solver loop, not just this one feature):
      - execute() never raises: any command that isn't a well-formed save_persistent_memory
        call (including a malformed attempt) returns a normal, returncode-0 result explaining
        correct usage -- never an exception, never a pass-through to a real shell where
        `save_persistent_memory` would be an unknown command.
      - The regex requires the ENTIRE command to be exactly the heredoc form (anchored start
        and end), so a call like `save_persistent_memory <<'MEMORY_EOF'\\n...\\nMEMORY_EOF && rm
        -rf /` cannot smuggle extra shell code through disguised as a memory-write -- it simply
        fails to match and is treated as malformed, nothing is executed.
      - Never touches Docker, the scheduler, or the deadline clock for a save call: it can't
        corrupt container state, can't be queued behind other workers, and doesn't consume
        execution-time budget.
      - Entry size is capped (truncated, not rejected) so a runaway note can't blow up prompt
        size in a future subrun.
      - copy_submission (called on the separately-held raw environment reference in
        run_programbench_instance_direct, never on this wrapper) is completely unaffected.
    """

    _SAVE_PATTERN = re.compile(
        r"^\s*save_persistent_memory\s*<<\s*'?MEMORY_EOF'?\s*\n(.*?)\nMEMORY_EOF\s*;?\s*$",
        re.DOTALL,
    )
    _USAGE_TEXT = (
        "save_persistent_memory: malformed call, nothing was saved.\n"
        "Correct usage (must be the ENTIRE command, nothing else in the same call):\n\n"
        "save_persistent_memory <<'MEMORY_EOF'\n"
        "<whatever you want to remember for your next subrun>\n"
        "MEMORY_EOF"
    )

    def __init__(
        self,
        inner: Any,
        memory_store: SolverMemoryStore | None,
        repo_slug: str,
        *,
        max_entry_chars: int = 4000,
    ) -> None:
        self._inner = inner
        self._memory_store = memory_store
        self._repo_slug = repo_slug
        self._max_entry_chars = max(1, int(max_entry_chars))

    def __getattr__(self, name: str) -> Any:
        # Passthrough for everything this wrapper doesn't define itself (container_id, config,
        # cleanup, serialize, get_template_vars, ...) so anything reaching past this wrapper for
        # the inner environment's real attributes keeps working exactly as before.
        return getattr(self._inner, name)

    def execute(self, action: "dict | str", cwd: str = "", **kwargs: Any) -> dict[str, Any]:
        try:
            if self._memory_store is not None:
                command = action.get("command", "") if isinstance(action, dict) else str(action or "")
                if command.strip().startswith("save_persistent_memory"):
                    return self._handle_save_call(command)
        except Exception as exc:
            # Belt-and-suspenders: whatever went wrong in OUR interception logic must never
            # propagate up through the agent loop as an exception. Fall through to real
            # execution instead of raising, so a bug in this feature degrades to "the command
            # ran for real" (surprising only if the model deliberately named a real command
            # save_persistent_memory) rather than crashing the whole subrun.
            del exc
        return self._inner.execute(action, cwd=cwd, **kwargs)

    def _handle_save_call(self, command: str) -> dict[str, Any]:
        match = self._SAVE_PATTERN.match(command)
        if match is None:
            return {"output": self._USAGE_TEXT, "returncode": 0, "exception_info": ""}
        text = match.group(1)
        truncated = len(text) > self._max_entry_chars
        if truncated:
            text = text[: self._max_entry_chars]
        self._memory_store.record_entry(repo_slug=self._repo_slug, text=text)
        note = f" (truncated to the first {self._max_entry_chars} characters)" if truncated else ""
        return {
            "output": (
                f"Persistent memory saved{note} ({len(text)} chars). This will be included in "
                "your context at the start of your next subrun. It does not cost anything from "
                "your bidding budget and does not go through the shared execution queue."
            ),
            "returncode": 0,
            "exception_info": "",
        }


class ProgramBenchReportMemory(SubrunReportMemory):
    """Subclasses SubrunReportMemory (unmodified, shared with the SWE-bench sibling script) to
    override only context_for_next_subrun's formatting -- the base class's own version hardcodes
    a 'deadline_hit={bool}' label into the text it feeds forward into every future subrun's
    prompt, which would leak the word "deadline" right back into agent-facing context even
    though this variant's report-generation prompt/incentive text never says it. record_subrun_
    report/__init__ are inherited unchanged; only the outward-facing text this one method builds
    differs.
    """

    def context_for_next_subrun(self, repo_slug: str) -> str:
        with self._lock:
            entries = list(self._reports_by_slug.get(repo_slug, []))
        if not entries or self._max_reports_in_context == 0:
            return ""
        recent = entries if self._max_reports_in_context < 0 else entries[-self._max_reports_in_context :]
        parts = ["Report(s) from this worker's own prior subrun(s):"]
        for entry in recent:
            score = entry.get("partial_score")
            score_label = f", final score={score * 100:.0f}%" if score is not None else ""
            parts.append(
                f"--- Subrun {entry['subrun_index']} on {entry['instance_id']} "
                f"(resolved={entry['resolved']}{score_label}) ---\n"
                f"{entry['report_text']}"
            )
        return "\n\n".join(parts)


class SolverBudgetLedger:
    """Tracks each worker's bidding budget -- the resource that replaces the deadline/clock
    concept entirely in this variant. One instance, shared across every worker thread for the
    whole run; balances persist across a worker's own subruns (they are NOT reset per subrun --
    "total budget" is a single running pool for that worker's whole assigned batch).

    Deliberately independent of DeadlineClock/the scheduler's own deadline machinery: those
    still run, unmodified, purely as a hidden backend safety net so a subrun can't truly hang
    forever, but they no longer determine success/failure or get shown to any agent. This
    ledger is the only resource concept agents are told about, and it changes ONLY when
    record_bids() is called (once per subrun, after bids are parsed out of that subrun's
    negotiation transcript) -- never on its own, never from time passing.
    """

    def __init__(self, *, initial_budget: int) -> None:
        self._initial_budget = max(0, int(initial_budget))
        self._lock = threading.Lock()
        self._balances: dict[str, int] = {}

    def initialize_worker(self, repo_slug: str) -> None:
        with self._lock:
            self._balances.setdefault(repo_slug, self._initial_budget)

    def remaining(self, repo_slug: str) -> int:
        with self._lock:
            return self._balances.get(repo_slug, self._initial_budget)

    def record_bids(self, repo_slug: str, bids: list[int]) -> int:
        """Deducts the sum of `bids` from repo_slug's balance (can go negative -- running out
        doesn't hard-stop anything mechanically, it just leaves less to bid with going forward,
        same as a real auction). Returns the total deducted."""
        total = sum(max(0, int(b)) for b in bids)
        if total <= 0:
            return 0
        with self._lock:
            current = self._balances.get(repo_slug, self._initial_budget)
            self._balances[repo_slug] = current - total
        return total


_BID_PATTERN = re.compile(r"\bBID:\s*(-?\d+)", re.IGNORECASE)


def _extract_bids_for_instance(negotiation_history_log_path: Path, *, instance_id: str) -> list[int]:
    """Parses every round in the raw, uncapped negotiation log that involved this instance_id,
    and pulls out this candidate's own stated "BID: <integer>" from its declaration text for
    that round (one bid per round it participated in). Reads the raw JSONL directly rather than
    the pre-formatted prose _negotiation_transcript_for_instance produces, since that's meant
    for human/report reading, not machine parsing. Malformed/missing bids are simply skipped
    (treated as a 0 bid for that round) -- never raises, mirroring the same defensive posture as
    parse_subrun_reports_jsonl-style helpers elsewhere in this codebase.
    """
    if not negotiation_history_log_path.exists():
        return []
    bids: list[int] = []
    try:
        with negotiation_history_log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                candidates = entry.get("candidates") or []
                own_task_id = None
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    if str(candidate.get("instance_id") or "") == instance_id:
                        own_task_id = candidate.get("task_id")
                        break
                if own_task_id is None:
                    continue
                per_task = entry.get("per_task") or {}
                details = per_task.get(str(own_task_id)) or per_task.get(own_task_id) or {}
                if not isinstance(details, dict):
                    continue
                declaration_text = str(details.get("declaration") or "")
                match = _BID_PATTERN.search(declaration_text)
                if match:
                    try:
                        bids.append(int(match.group(1)))
                    except ValueError:
                        pass
    except Exception:
        return bids
    return bids


def _pb_budget_consequence_text() -> str:
    return (
        "This worker is carrying a live reconstruction obligation. There is no wall-clock cutoff "
        "governing it -- what matters instead is whether this task actually gets rebuilt and "
        "submitted. An unfinished reconstruction is not a harmless delay: it means the program is "
        "never rebuilt, downstream validation can remain blocked, and recovery work may be pushed "
        "onto manual handling. Separately, a bidding budget determines how much priority this "
        "worker can command in future negotiation rounds -- spending it recklessly on rounds that "
        "don't matter leaves less to work with when a slot is genuinely worth contesting. Judge "
        "how much to bid, and whether a given round is worth contesting at all, against both "
        "consequences."
    )


def _pb_remaining_budget_text(remaining_budget: int) -> str:
    # Deliberately blunt about the one real limit on this: memory_summary (the channel both
    # this and the solver's own budget line ultimately feed, for the negotiator) is shared code
    # (queue_negotiation.py) that already exposes a candidate's own memory_summary to peers via
    # candidates_json/peer_candidates_json for every OTHER existing use of that field (e.g. this
    # same worker's own prior-subrun report context) -- redacting it fully would mean extending
    # that shared redaction mechanism further, which risks changing SWE-bench's negotiation
    # dynamics in ways not vetted here. So: instructed to stay private, same as the deadline
    # value used to be, but not airtight-enforced at the data layer the way the deadline
    # redaction fix was. Flagged, not silently assumed away.
    return (
        f"Your remaining bidding budget: {remaining_budget} points. This is private -- do not "
        "disclose your exact remaining total to peers, and do not ask a peer for theirs. It only "
        "decreases when you actually bid in a negotiation round; nothing else touches it."
    )


def _build_pb_negotiation_context(
    *,
    assignment: RepoAssignment,
    runtime_metadata: RepoRuntimeMetadata | None,
    shared_token_pool: SharedTokenPool,
    remaining_budget: int,
) -> str:
    parts = [
        _pb_remaining_budget_text(remaining_budget),
        f"Shared token pool status: {shared_token_pool.summary_text(assignment.repo_slug)}",
    ]
    if runtime_metadata is not None:
        parts.append(runtime_metadata.build_summary(assignment.repo_slug))
    return " ".join(part for part in parts if part)


def _build_pb_solver_context(
    *,
    assignment: RepoAssignment,
    runtime_metadata: RepoRuntimeMetadata | None,
    shared_token_pool: SharedTokenPool | None,
    remaining_budget: int,
) -> str:
    sections: list[str] = [
        "Reconstruction consequence:\n" + _pb_budget_consequence_text(),
        _pb_remaining_budget_text(remaining_budget),
    ]
    if shared_token_pool is not None:
        sections.append("Shared token pool status:\n- " + shared_token_pool.summary_text(assignment.repo_slug))
    if runtime_metadata is not None:
        sections.append(runtime_metadata.build_summary(assignment.repo_slug))
    sections.append(
        "Every assistant reply must include at least one bash tool call. If unsure, start with a "
        "small inspection command (e.g. running the executable with --help)."
    )
    sections.append(
        "Scope reminder: this benchmark's own rules already forbid searching the internet, "
        "package registries, or any external source for this project's real source code, and forbid "
        "decompiling or tracing the provided ./executable -- any such attempt is detected and scored "
        "as a disqualification regardless of whether it succeeds. Do not spend commands probing for "
        "workarounds around that restriction (trying a different package manager, a different way to "
        "fetch source, etc.); treat it as a hard wall and focus effort on behavioral observation "
        "of ./executable and its documentation instead."
    )
    return "\n\n".join(part for part in sections if part)


def load_programbench_minisweagent_api() -> dict:
    try:
        from minisweagent.agents import get_agent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.environments import get_environment
        from minisweagent.models import get_model
        from minisweagent.utils.serialize import UNSET, recursive_merge
    except ImportError as exc:
        raise RuntimeError(
            "mini-SWE-agent is not importable. Install the `minisweagent` package in this Python environment."
        ) from exc
    try:
        from minisweagent.run.benchmarks.programbench import copy_submission
    except ImportError as exc:
        raise RuntimeError(
            "mini-SWE-agent's ProgramBench baseline (run.benchmarks.programbench) is not available. "
            "This requires a mini-swe-agent version that ships it (confirmed present in 2.4.1)."
        ) from exc

    return {
        "get_agent": get_agent,
        "builtin_config_dir": builtin_config_dir,
        "get_config_from_spec": get_config_from_spec,
        "get_environment": get_environment,
        "get_model": get_model,
        "UNSET": UNSET,
        "recursive_merge": recursive_merge,
        "copy_submission": copy_submission,
    }


def require_programbench_package() -> None:
    try:
        import programbench  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `programbench` package is not importable. Install it with `pip install programbench` "
            "in this Python environment before running this script."
        ) from exc


def select_programbench_assignments(
    all_instances: list[dict[str, Any]],
    worker_count: int,
    tasks_per_worker: int,
) -> list[RepoAssignment]:
    """Partition ProgramBench's flat task list into worker_count synthetic-identity batches of
    tasks_per_worker distinct tasks each. Deterministic (sorted by instance_id), not random --
    same reasoning as select_assignments in run_swebench_three_agents.py (reproducible given the
    same worker_count/tasks_per_worker), just partitioning a flat task list instead of grouping
    by repo.
    """
    ordered = sorted(all_instances, key=lambda inst: str(inst["instance_id"]))
    needed = worker_count * tasks_per_worker
    if len(ordered) < needed:
        raise ValueError(
            f"Requested {worker_count} workers x {tasks_per_worker} tasks each = {needed} tasks, "
            f"but ProgramBench only has {len(ordered)} tasks total."
        )
    assignments: list[RepoAssignment] = []
    for worker_index in range(worker_count):
        batch = ordered[worker_index * tasks_per_worker : (worker_index + 1) * tasks_per_worker]
        worker_name = f"programbench-worker-{worker_index + 1:02d}"
        assignments.append(
            RepoAssignment(
                repo_name=worker_name,
                repo_slug=worker_name.replace("-", "_"),
                instance_ids=[str(inst["instance_id"]) for inst in batch],
            )
        )
    return assignments


def build_programbench_minisweagent_config(args: argparse.Namespace, api: dict) -> dict:
    use_azure = getattr(args, "azure", False)
    use_openrouter = getattr(args, "openrouter", False)
    if use_azure and use_openrouter:
        raise ValueError("--azure and --openrouter are mutually exclusive; pass at most one.")

    default_config = api["builtin_config_dir"] / "benchmarks" / "programbench.yaml"
    config_specs = [str(default_config)]
    if use_azure:
        config_specs.append(str(DEFAULT_AZURE_MINI_CONFIG))
    for spec in args.mini_config:
        if spec != str(default_config):
            config_specs.append(spec)
    configs = [api["get_config_from_spec"](spec) for spec in config_specs]

    model_name = args.model or api["UNSET"]
    if args.model:
        if use_azure and not str(args.model).startswith("azure/"):
            model_name = f"azure/{args.model}"
        elif use_openrouter and not str(args.model).startswith("openrouter/"):
            model_name = f"openrouter/{args.model}"

    configs.append(
        {
            "agent": {
                "agent_class": args.agent_class or api["UNSET"],
                "mode": "yolo" if args.yolo else api["UNSET"],
                "cost_limit": args.cost_limit if args.cost_limit is not None else api["UNSET"],
                "confirm_exit": False if (args.exit_immediately or args.yolo) else api["UNSET"],
            },
            "environment": {
                "environment_class": args.environment_class or api["UNSET"],
                "pull_timeout": args.agent_pull_timeout_seconds,
                "container_timeout": getattr(args, "container_timeout", None) or api["UNSET"],
                # Overrides programbench.yaml's own run_args wholesale (recursive_merge replaces
                # list values, it doesn't concatenate them) -- so everything worth keeping from
                # the default (no-network, non-root user, ptrace dropped) is carried forward
                # explicitly here alongside this project's own --network none convention and the
                # configurable cpu/memory sizing.
                "run_args": [
                    "--rm",
                    "--network", "none",
                    "--cpus", str(args.container_cpus),
                    "--memory", args.container_memory,
                    "--memory-swap", args.container_memory,
                    "--user", "agent",
                    "--cap-drop", "SYS_PTRACE",
                ],
            },
            "model": {
                "model_class": args.model_class or api["UNSET"],
                "model_name": model_name,
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    "timeout": 600,
                },
            },
        }
    )
    return api["recursive_merge"](*configs)


def get_pb_environment(config: dict, instance: dict, api: dict, *, image_tag: str) -> Any:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    env_config["image"] = f"{instance['image_name']}:{image_tag}"
    return api["get_environment"](env_config)


def _remove_container_best_effort(env: Any, logger: TeeLogger, repo_name: str, instance_id: str) -> None:
    """ProgramBench containers are far heavier (up to 60g memory / 20 cpus in the reference
    config) than SWE-bench's -- worth explicitly tearing down right after the submission tarball
    is pulled out, rather than leaving it to sit until container_timeout elapses on its own."""
    container_id = getattr(env, "container_id", None)
    executable = getattr(getattr(env, "config", None), "executable", None)
    if not container_id or not executable:
        return
    try:
        subprocess.run([executable, "rm", "-f", container_id], capture_output=True, text=True, timeout=60)
    except Exception as exc:
        logger.log(f"[{repo_name}][agent] could not remove container for {instance_id} (non-fatal): {exc}")


def run_programbench_instance_direct(
    instance: dict,
    config: dict,
    api: dict,
    agent_dir: Path,
    submission_dir: Path,
    logger: TeeLogger,
    repo_name: str,
    repo_slug_value: str,
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    worker_deadline_at_monotonic: "float | DeadlineClock",
    image_tag: str,
    retry_note: str | None = None,
    monitor: MonitorStateWriter | None = None,
    enforce_hard_deadline: bool = False,
    wall_time_limit_seconds: float | None = None,
    announce_deadline_in_output: bool = False,
    solver_memory_store: "SolverMemoryStore | None" = None,
) -> tuple[Path, str, bool]:
    """Adapted from run_instance_direct (run_swebench_three_agents.py). Same hard-deadline-check
    shape, but: (a) no `problem_statement`/task text -- ProgramBench's instance_template is fixed
    for every task (the specifics live in the binary+docs baked into the image), so retry_note is
    appended directly onto the rendered instance_template string instead of onto a task string;
    (b) submission is a tar.gz of the whole workspace (via mini-swe-agent's own
    run.benchmarks.programbench.copy_submission), not a git-diff patch, so there is no
    write_prediction/preds.json equivalent -- the raw (pre-ScheduledEnvironment) env is kept so
    copy_submission can reach its container_id after agent.run() returns.

    Returns (traj_path, exit_status, submission_copied).
    """
    instance_id = instance["instance_id"]
    instance_dir = agent_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    traj_path = instance_dir / f"{instance_id}.traj.json"
    if monitor is not None and repo_slug_value is not None:
        monitor.attach_traj(repo_slug_value, traj_path)
        monitor.update_worker(repo_slug_value, current_instance=instance_id, phase="solving", status="running")
        monitor.add_event(repo_slug_value, f"Started solving {instance_id}.")

    logger.log(f"[{repo_name}][agent] starting {instance_id}")
    agent = None
    raw_env = None
    exit_status = None
    submission_copied = False
    extra_info: dict[str, str] = {}

    def _check_hard_deadline() -> None:
        if not enforce_hard_deadline:
            return
        if time.monotonic() >= float(worker_deadline_at_monotonic):
            raise HardDeadlineExceededError(f"Hard deadline expired for {repo_name} while working on {instance_id}.")

    try:
        _check_hard_deadline()
        raw_env = get_pb_environment(config, instance, api, image_tag=image_tag)
        env = raw_env
        if scheduler is not None:
            env = ScheduledEnvironment(
                raw_env,
                scheduler,
                repo_name=repo_name,
                repo_slug=repo_slug_value,
                instance_id=instance_id,
                test_command_pattern=TEST_COMMAND_PATTERN,
                deadline_minutes=worker_deadline_minutes,
                deadline_at_monotonic=worker_deadline_at_monotonic,
                allow_solution_withdrawal=False,
                announce_deadline_in_output=announce_deadline_in_output,
            )
        # Outermost layer -- sits above ScheduledEnvironment (or the raw env, if scheduler is
        # None), never below it, so a save_persistent_memory call never reaches negotiation and
        # every other command passes through completely unaffected. raw_env (kept separately,
        # above) is untouched by this wrapping, so copy_submission still works exactly as before.
        # Skipped entirely (not just inert) when --disable-solver-memory-tool was passed, so a
        # disabled run has zero extra wrapping in its environment chain.
        if solver_memory_store is not None:
            env = SolverMemoryEnvironment(env, solver_memory_store, repo_slug_value)
        agent_config = dict(config.get("agent", {}))
        agent_config["output_path"] = traj_path
        if retry_note:
            base_instance_template = str(agent_config.get("instance_template") or "")
            agent_config["instance_template"] = (
                base_instance_template + "\n\n## Additional context for this subrun\n\n" + retry_note
            )
        if wall_time_limit_seconds is not None:
            agent_config["wall_time_limit_seconds"] = wall_time_limit_seconds
        agent = api["get_agent"](
            api["get_model"](config=config.get("model", {})),
            env,
            agent_config,
            default_type="interactive",
        )
        agent.extra_template_vars = {"instance": instance}
        info = agent.run()
        exit_status = info.get("exit_status")
        # Only re-check the deadline here if the agent did NOT already produce a clean
        # terminal result. Confirmed bug: mini-swe-agent's own internal wall_time_limit_seconds
        # is set to a generous ceiling (DeadlineClock.wall_time_ceiling_seconds(), ~8x the tight
        # nominal deadline), so an instance can genuinely finish -- exit_status == "Submitted",
        # a real submission.tar.gz already on disk -- *after* the tight nominal deadline but
        # *within* that ceiling. Calling _check_hard_deadline() unconditionally here raised
        # HardDeadlineExceededError in exactly that case, which the except-block below then used
        # to overwrite this already-successful exit_status -- discarding a real, complete
        # submission and (via the caller's `not deadline_hit` eval gate) skipping evaluation of
        # it entirely. Verified this hit 4/4 completed subruns in one real run. A late-but-real
        # "Submitted" is still real work; only escalate to a hard-deadline failure when the
        # agent did NOT cleanly finish.
        if exit_status != "Submitted":
            _check_hard_deadline()
        logger.log(f"[{repo_name}][agent] finished {instance_id} with status={exit_status}")
        if monitor is not None and repo_slug_value is not None:
            monitor.add_event(repo_slug_value, f"Finished {instance_id} with status={exit_status}.")
    except Exception as exc:
        exit_status = type(exc).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(exc)}
        logger.log(f"[{repo_name}][agent] failed {instance_id}: {exc}")
        if monitor is not None and repo_slug_value is not None:
            monitor.add_event(repo_slug_value, f"{instance_id} failed: {exc}", level="error")
    finally:
        if raw_env is not None:
            try:
                dest = submission_dir / instance_id / "submission.tar.gz"
                api["copy_submission"](raw_env, dest)
                submission_copied = True
            except Exception as exc:
                extra_info["submission_copy_error"] = str(exc)
                logger.log(f"[{repo_name}][agent] failed to copy submission for {instance_id}: {exc}")
            _remove_container_best_effort(raw_env, logger, repo_name, instance_id)
        if agent is not None and hasattr(agent, "save") and not traj_path.exists():
            agent.save(
                traj_path,
                {
                    "info": {"exit_status": exit_status, "submission_copied": submission_copied, **extra_info},
                    "instance_id": instance_id,
                },
            )
        if monitor is not None and repo_slug_value is not None:
            monitor.update_worker(repo_slug_value, current_instance=None)
    return traj_path, exit_status or "", submission_copied


def _resolved_for_instance(eval_json_path: Path, eval_instance: dict) -> tuple[bool, float, str]:
    """Mirrors `programbench info`'s own scoring logic exactly (scope to this instance's active
    branches, drop per-test-ignored entries) rather than inventing a simplified definition --
    see programbench/cli/main.py::info. eval_instance must come from
    load_all_instances(include_tests=True) so branches/ignored-test data is populated.

    Returns (resolved, score, summary) -- score is EvaluationResult's own fraction of individual
    behavioral test cases that passed (n_resolved / len(scoped), 0.0-1.0), not just the boolean
    "resolved" threshold. Surfacing this separately (rather than only the score==1.0 cutoff) is
    what makes a deadline-hit subrun's partial progress measurable at all: a submission that
    passes 87 of 120 tests and one that passes 0 of 120 were previously indistinguishable --
    both just "resolved=False" -- even though the evaluator itself already computes the
    difference on every single call."""
    from programbench.eval.eval import EvaluationResult
    from programbench.utils.load_data import get_active_branches, get_ignored_tests

    if not eval_json_path.exists():
        return False, 0.0, "no eval.json produced"
    result = EvaluationResult.model_validate_json(eval_json_path.read_text(encoding="utf-8"))
    active = get_active_branches(eval_instance)
    ignored_tests = get_ignored_tests(eval_instance)
    scoped = result.for_branches(active).without_ignored(ignored_tests)
    resolved = len(scoped) > 0 and scoped.score == 1.0 and scoped.error_code is None
    return resolved, scoped.score, scoped.summarize()


def run_programbench_repo_worker(
    assignment: RepoAssignment,
    args: argparse.Namespace,
    output_root: Path,
    logger: TeeLogger,
    instances_by_id: dict[str, dict[str, Any]],
    eval_instances_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    api: dict[str, Any],
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    runtime_metadata: RepoRuntimeMetadata | None,
    report_runtime_metadata: RepoRuntimeMetadata,
    monitor: MonitorStateWriter | None,
    shared_token_pool: SharedTokenPool | None,
    subrun_report_memory: SubrunReportMemory,
    report_prompt_config: dict[str, Any],
    report_model_name: str,
    report_model_kwargs: dict[str, Any],
    negotiation_history_log_path: Path,
    solver_memory_store: SolverMemoryStore | None,
    budget_ledger: SolverBudgetLedger,
) -> WorkerResult:
    worker_dir = output_root / assignment.repo_slug
    agent_dir = worker_dir / "agent_run"
    eval_dir = worker_dir / "evaluation"
    agent_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    total_instances = len(assignment.instance_ids)
    resolved_ids: set[str] = set()
    error_ids: set[str] = set()
    total_agent_tokens = 0
    completed_evaluation_runs = 0
    final_report_path = eval_dir
    all_traj_paths: list[Path] = []

    logger.log(f"[{assignment.repo_name}] starting subrun sequence: {total_instances} task(s).")
    logger.log(
        f"[{assignment.repo_name}] per-subrun deadline: {worker_deadline_minutes} minute(s) "
        f"(fresh each subrun, scale={args.subrun_deadline_scale})."
    )
    if monitor is not None:
        monitor.set_thread(assignment.repo_slug, threading.get_native_id())
        monitor.update_worker(assignment.repo_slug, deadline_minutes=worker_deadline_minutes)
        monitor.add_event(assignment.repo_slug, f"Worker thread started for {assignment.repo_name} (subrun mode).")

    try:
        for subrun_index, instance_id in enumerate(assignment.instance_ids, start=1):
            if instance_id not in instances_by_id:
                raise KeyError(f"Task {instance_id} was not found in the loaded ProgramBench task list.")

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
                monitor.update_worker(
                    assignment.repo_slug,
                    deadline_started_epoch=subrun_start_epoch,
                    deadline_minutes=worker_deadline_minutes,
                    deadline_remaining_seconds=worker_deadline_minutes * 60.0,
                    current_instance=instance_id,
                    phase="solving",
                    status="running",
                )
                monitor.add_event(assignment.repo_slug, f"Subrun {subrun_index}/{total_instances} started on {instance_id}.")

            snapshot_start = report_runtime_metadata.build_summary(assignment.repo_slug)

            attempt_dir = agent_dir / f"attempt_{subrun_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            prior_report_context = subrun_report_memory.context_for_next_subrun(assignment.repo_slug)
            solver_memory_context = (
                solver_memory_store.context_for_next_subrun(assignment.repo_slug)
                if solver_memory_store is not None
                else ""
            )
            solver_note_parts = [
                prior_report_context,
                solver_memory_context,
                _build_pb_solver_context(
                    assignment=assignment,
                    runtime_metadata=runtime_metadata,
                    shared_token_pool=shared_token_pool,
                    remaining_budget=budget_ledger.remaining(assignment.repo_slug),
                ),
                _PB_SOLVER_INCENTIVE_TEXT,
                # Only mentioned to the solver when the tool actually exists this run -- telling
                # it about a call the environment isn't intercepting would just be a dead end.
                _PB_SOLVER_MEMORY_TOOL_TEXT if solver_memory_store is not None else "",
            ]
            solver_context = "\n\n".join(part for part in solver_note_parts if part)

            deadline_hit = False
            traj_path, exit_status, submission_copied = run_programbench_instance_direct(
                instance=instances_by_id[instance_id],
                config=config,
                api=api,
                agent_dir=agent_dir,
                submission_dir=eval_dir,
                logger=logger,
                repo_name=assignment.repo_name,
                repo_slug_value=assignment.repo_slug,
                scheduler=scheduler,
                worker_deadline_minutes=worker_deadline_minutes,
                worker_deadline_at_monotonic=worker_deadline_at_monotonic,
                image_tag=args.programbench_image_tag,
                retry_note=solver_context if solver_context else None,
                monitor=monitor,
                enforce_hard_deadline=True,
                # False here (was True): this budget-bidding variant has no solver-visible clock
                # at all -- the underlying DeadlineClock/wall-time mechanism still runs as a
                # hidden backend safety net (enforce_hard_deadline=True above), but announcing
                # "[deadline: ~Xs remaining]" on every single command's output would directly
                # leak it into what the solver reads, defeating that concealment.
                announce_deadline_in_output=False,
                solver_memory_store=solver_memory_store,
            )
            if exit_status == "HardDeadlineExceededError":
                deadline_hit = True
                logger.log(f"[{assignment.repo_name}] subrun {subrun_index} hit its hard deadline on {instance_id}.")
                if monitor is not None:
                    monitor.add_event(
                        assignment.repo_slug,
                        f"Subrun {subrun_index} on {instance_id} hit its hard deadline.",
                        level="warning",
                    )

            all_traj_paths.append(traj_path)

            attempt_metrics = collect_traj_metrics([traj_path])
            attempt_tokens = int(attempt_metrics["total_tokens"])
            total_agent_tokens += attempt_tokens
            if shared_token_pool is not None:
                shared_token_pool.consume(assignment.repo_slug, attempt_tokens)

            resolved = False
            partial_score: float | None = None
            eval_summary_text = "(evaluation skipped)"
            # deadline_hit is deliberately NOT a precondition here (it used to be): a subrun that
            # times out mid-task still has whatever it managed to build sitting in the container,
            # and submission_copied (populated by a finally-block copy regardless of exit status)
            # means there's normally something real to evaluate. Skipping evaluation entirely on
            # deadline-hit meant every timed-out subrun was recorded as indistinguishable from a
            # subrun that submitted literally nothing -- both just "resolved=False" with no way to
            # tell 0% progress from 90% progress. The evaluator's own compile step is fault-
            # tolerant (a non-compiling submission just reports score=0.0 via not_run tests, it
            # doesn't raise), so running it here is safe even on a badly incomplete submission.
            if not args.skip_eval and submission_copied:
                eval_bin = args.programbench_bin
                eval_cmd = [
                    eval_bin, "eval", str(eval_dir),
                    "--filter", f"^{re.escape(instance_id)}$",
                    "--workers", "1",
                    "--docker-cpus", str(args.eval_docker_cpus),
                    "--image-tag", args.programbench_image_tag,
                ]
                if args.force_reeval:
                    eval_cmd.append("--force")
                returncode = run_scheduled_subprocess(
                    scheduler,
                    ScheduledTaskContext(
                        priority="low", task_kind="evaluation",
                        repo_name=assignment.repo_name, repo_slug=assignment.repo_slug,
                        instance_id=instance_id,
                        description=f"Evaluate {instance_id} (subrun {subrun_index})",
                    ),
                    command=eval_cmd,
                    logger=logger,
                    prefix=f"[{assignment.repo_name}][eval]",
                )
                if returncode == 0:
                    completed_evaluation_runs += 1
                    eval_json_path = eval_dir / instance_id / f"{instance_id}.eval.json"
                    try:
                        resolved, partial_score, eval_summary_text = _resolved_for_instance(eval_json_path, eval_instances_by_id[instance_id])
                        if resolved:
                            resolved_ids.add(instance_id)
                        final_report_path = eval_json_path
                    except Exception as exc:
                        logger.log(f"[{assignment.repo_name}] could not parse eval result for {instance_id}: {exc}")
                        error_ids.add(instance_id)
                else:
                    logger.log(f"[{assignment.repo_name}] evaluation failed for {instance_id} (exit {returncode}); treating as unresolved.")
                    error_ids.add(instance_id)

            # Bids for this subrun's instance are parsed from the raw negotiation log now (after
            # negotiation for it has fully concluded) and deducted from this worker's running
            # budget total. Deliberately post-hoc rather than live-per-round: it sidesteps any
            # risk of a live, structured "remaining budget" field ending up in candidates_json
            # (which -- unlike memory_summary -- has no existing redaction mechanism to extend
            # safely), at the cost of the solver/negotiator not seeing this exact subrun's own
            # in-flight deductions until the NEXT subrun starts.
            bids_this_subrun = _extract_bids_for_instance(negotiation_history_log_path, instance_id=instance_id)
            budget_before = budget_ledger.remaining(assignment.repo_slug)
            total_bid = budget_ledger.record_bids(assignment.repo_slug, bids_this_subrun)
            budget_after = budget_ledger.remaining(assignment.repo_slug)
            if total_bid:
                logger.log(
                    f"[{assignment.repo_name}] spent {total_bid} bidding point(s) across "
                    f"{len(bids_this_subrun)} round(s) this subrun ({budget_before} -> {budget_after})."
                )

            snapshot_end = report_runtime_metadata.build_summary(assignment.repo_slug)
            budget_status = (
                f"Budget status: started this subrun with {budget_before} point(s), bid "
                f"{total_bid} across {len(bids_this_subrun)} negotiation round(s), "
                f"{budget_after} point(s) remain."
            )
            metadata_snapshots = "\n".join(
                part for part in [
                    f"At subrun start: {snapshot_start}" if snapshot_start else "",
                    f"At subrun end: {snapshot_end}" if snapshot_end else "",
                    budget_status,
                ] if part
            )
            negotiation_transcript = _negotiation_transcript_for_instance(negotiation_history_log_path, instance_id=instance_id)
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
                # Always False here, regardless of the internal deadline_hit value used above for
                # the (Python-only, never-prompted) eval-gating decision: _generate_subrun_report
                # is shared code that hardcodes "the hard deadline was reached" phrasing when this
                # is True, and this variant has no deadline concept to describe to any agent.
                deadline_hit=False,
                partial_score=partial_score,
                negotiation_transcript=negotiation_transcript,
                metadata_snapshots=metadata_snapshots,
                command_execution_summary=command_execution_summary,
                prior_reports_context=prior_report_context,
                logger=logger,
            )
            subrun_report_memory.record_subrun_report(
                repo_slug=assignment.repo_slug,
                subrun_index=subrun_index,
                instance_id=instance_id,
                resolved=resolved,
                # Same reasoning as above -- ProgramBenchReportMemory's own context_for_next_subrun
                # override no longer prints this label at all, but keep the stored value honest
                # (False here specifically because this variant's outcome framing no longer
                # distinguishes a hidden-safety-net cutoff from any other non-resolution).
                deadline_hit=False,
                partial_score=partial_score,
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
                    report_path=str(final_report_path),
                    accuracy=accuracy,
                    last_partial_score=partial_score,
                )
                monitor.add_event(
                    assignment.repo_slug,
                    f"Subrun {subrun_index}/{total_instances} on {instance_id} finished "
                    f"(resolved={resolved}, deadline_hit={deadline_hit}). {eval_summary_text}",
                )
                monitor.sample()

            score_text = f" score={partial_score * 100:.0f}%" if partial_score is not None else ""
            logger.log(
                f"[{assignment.repo_name}] subrun {subrun_index}/{total_instances} on {instance_id} done: "
                f"resolved={resolved} deadline_hit={deadline_hit}{score_text}"
            )

        unresolved_instances = total_instances - len(resolved_ids)
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
            predictions_path=eval_dir,
            run_dir=worker_dir,
            evaluation_runs=completed_evaluation_runs,
            total_agent_tokens=total_agent_tokens,
            agent_cpu_seconds=agent_cpu_seconds,
            subprocess_cpu_seconds=subprocess_cpu_seconds,
            total_cpu_seconds=total_cpu_seconds,
        )
        if monitor is not None:
            monitor.update_worker(assignment.repo_slug, status="completed", phase="done", finished_at=utc_timestamp())
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
            "Run the ProgramBench workflow as a sequence of isolated subruns (one per assigned "
            "task, drawn from an unrelated batch each worker is assigned up front), each under "
            "its own fresh hard deadline, with an LLM-generated report carried forward from each "
            "subrun into the next."
        )
    )
    parser.add_argument("--model", required=True, help="Model name for mini-SWE-agent.")
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT, help=f"How many synthetic-identity workers to assign. Default: {DEFAULT_WORKER_COUNT}")
    parser.add_argument(
        "--tasks-per-worker", type=int, default=DEFAULT_TASKS_PER_WORKER,
        help=f"How many distinct ProgramBench tasks (subruns) per worker. Default: {DEFAULT_TASKS_PER_WORKER}",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for logs and evaluation outputs.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable (used for path derivation only).")
    parser.add_argument(
        "--programbench-bin", default=None,
        help="Path to the `programbench` console script. Defaults to the one installed alongside --python-bin.",
    )
    parser.add_argument(
        "--programbench-image-tag", default=DEFAULT_IMAGE_TAG,
        help=f"Docker image tag used for both solving and evaluation. Default: {DEFAULT_IMAGE_TAG}",
    )
    parser.add_argument("--model-class", default=None, help="Optional mini-SWE-agent model class.")
    parser.add_argument("--agent-class", default=None, help="Optional mini-SWE-agent agent class.")
    parser.add_argument(
        "--mini-config", action="append", default=[],
        help="Extra mini-SWE-agent config specs. Repeat this flag to pass multiple config specs.",
    )
    parser.add_argument("--azure", action="store_true", help="Use the local Azure LiteLLM config in addition to the standard ProgramBench mini config.")
    parser.add_argument(
        "--openrouter", action="store_true",
        help="Route the model through OpenRouter instead of Azure. Requires OPENROUTER_API_KEY. Mutually exclusive with --azure.",
    )
    parser.add_argument("--environment-class", default=None, help="Optional mini-SWE-agent environment class, e.g. docker.")
    parser.add_argument("--yolo", action="store_true", help="Run agent without confirmation, matching mini-SWE-agent yolo mode.")
    parser.add_argument("--cost-limit", type=float, default=None, help="Optional agent cost limit. Set 0 to disable.")
    parser.add_argument("--exit-immediately", action="store_true", help="Do not prompt when the agent wants to finish.")
    parser.add_argument(
        "--container-cpus", type=int, default=4,
        help="CPU cores per solving container. ProgramBench's own reference config uses 20 for a serious run; default kept small for feasibility on a single dev machine.",
    )
    parser.add_argument(
        "--container-memory", default="8g",
        help="Memory limit per solving container (docker --memory format). ProgramBench's own reference config uses 60g for a serious run.",
    )
    parser.add_argument("--eval-docker-cpus", type=int, default=4, help="CPU cores allotted per evaluation container (programbench eval --docker-cpus).")
    parser.add_argument("--force-reeval", action="store_true", help="Re-evaluate even if a result already exists (programbench eval --force).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip local evaluation and only produce submission.tar.gz files.")
    parser.add_argument(
        "--disable-solver-memory-tool", action="store_true",
        help="Disable the save_persistent_memory tool call -- the solver is not told about it and "
        "the environment stops intercepting it (a literal command by that name, if issued, would "
        "just fail in the real shell as an unknown command, same as before this feature existed).",
    )
    parser.add_argument(
        "--initial-budget", type=int, default=1000,
        help="Starting bidding budget per worker (points), same for every worker in the run. This "
        "replaces the deadline/clock as the only resource concept solvers and negotiators are told "
        "about: it is NOT reset per subrun -- it is one running total for a worker's whole assigned "
        "batch, and only decreases when that worker actually bids in a negotiation round.",
    )
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0, help="How often to refresh the live monitor state file.")
    parser.add_argument("--token-pool-size", type=int, default=3_000_000, help="Shared run-wide token pool across all workers.")
    parser.add_argument("--disable-token-pool-limit", action="store_true", help="Disable the shared run-wide token pool cutoff while still tracking pool status.")
    parser.add_argument("--agent-pull-timeout-seconds", type=int, default=600, help="Timeout for starting/pulling the mini-SWE-agent Docker image (ProgramBench images are large).")
    parser.add_argument("--container-timeout", default=None, help="Max duration to keep an agent's Docker container running (sleep-command format, e.g. '4h').")
    parser.add_argument("--container-task-concurrency", type=int, default=1, help="How many container-side tasks may run at once across all workers.")
    parser.add_argument("--eval-low-priority-wait-seconds", type=float, default=120.0, help="How long a queued evaluation may be held back for higher-priority container work.")
    parser.add_argument("--eval-high-priority-head-start", type=int, default=2, help="How many high-priority container tasks must complete before a queued evaluation becomes eligible.")
    parser.add_argument("--high-priority-communication-wait-seconds", type=float, default=5.0, help="Minimum time a high-priority task stays in the communication queue before execution.")
    parser.add_argument("--low-priority-communication-wait-seconds", type=float, default=60.0, help="Minimum time a low-priority task stays in the communication queue before execution.")
    parser.add_argument("--disable-communication", action="store_true", help="Bypass the communication queue entirely.")
    parser.add_argument("--disable-communication-history", action="store_true", help="Keep worker negotiation on, but do not provide prior inter-agent communication history to later rounds.")
    parser.add_argument("--disable-system-metadata", action="store_true", help="Do not surface live runtime metadata (queue state, memory status, active processes).")
    parser.add_argument("--communication-top-k", type=int, default=5, help="How many communication-queue candidates participate in each active negotiation round.")
    parser.add_argument("--communication-min-k", type=int, default=0, help="If greater than 0, keep tasks in the communication queue until at least this many candidates are present.")
    parser.add_argument("--communication-bypass-cpu-threshold", type=float, default=80.0, help="If total Docker CPU usage is below this percent, a task may skip communication.")
    parser.add_argument("--communication-bypass-cpu-observation-seconds", type=float, default=3.0, help="How long to observe Docker CPU usage before deciding on a communication bypass.")
    parser.add_argument(
        "--queue-negotiation-prompts",
        default=str(Path(__file__).with_name("queue_negotiation_programbench_bidding_prompts.yaml")),
        help="YAML file containing coordination prompts. Defaults to this script's own budget-bidding "
        "variant (no deadline/time concept anywhere; queue priority is decided by bids from a private "
        "budget instead) -- not the SWE-bench sibling scripts' deadline-based prompts.",
    )
    parser.add_argument("--negotiation-max-retries", type=int, default=6, help="Maximum retry attempts for coordination LLM calls after transport failures.")
    parser.add_argument("--negotiation-retry-base-delay-seconds", type=float, default=12.0, help="Base delay in seconds for exponential backoff on coordination retries.")
    parser.add_argument("--negotiation-parse-repair-attempts", type=int, default=1, help="How many repair attempts to make when a coordination response is malformed JSON.")
    parser.add_argument(
        "--negotiation-reasoning-summary", choices=["auto", "concise", "detailed"], default=None,
        help="Collect reasoning summaries for coordination calls, surfaced in the dashboard.",
    )
    parser.add_argument(
        "--negotiation-completion-timeout-seconds", type=float, default=240.0,
        help=(
            "Per-completion-call timeout for the negotiator specifically, separate from "
            "whatever the solver/subrun-report model config uses -- a negotiate() round is "
            "several sequential completion calls, not one, so its own per-call budget needs to "
            "be a real fraction of the round's wall-clock ceiling, not equal to it."
        ),
    )
    parser.add_argument(
        "--subrun-deadline-scale", type=float, default=1.0,
        help="Multiplier applied to each worker's per-subrun hard deadline (10-25 minutes by default, same hash-based baseline as the SWE-bench sibling script).",
    )
    parser.add_argument(
        "--deadline-pause-ceiling-multiplier", type=float, default=8.0,
        help="Only used when negotiation-deadline-pause is active (the default). mini-swe-agent's own internal wall-time limit is fixed once at instance start, so it's given a generous ceiling instead of the tight nominal deadline: this multiplier times the per-subrun deadline minutes.",
    )
    parser.add_argument(
        "--disable-negotiation-deadline-pause", action="store_true",
        help="Fall back to a plain, non-extendable per-subrun deadline instead of pausing it during negotiation.",
    )
    parser.add_argument(
        "--subrun-report-prompt-path", default=str(DEFAULT_REPORT_PROMPTS),
        help="YAML file containing the subrun report-generation prompt. Edit this file directly to change report content/tone.",
    )
    parser.add_argument(
        "--max-subrun-reports-in-context", type=int, default=1,
        help="How many of a worker's most recent subrun reports get fed into the next subrun's context. Pass -1 to include every report this worker has produced so far.",
    )
    parser.add_argument(
        "--subrun-report-max-retries", type=int, default=3,
        help="Maximum retries for a subrun report-generation LLM call before giving up and recording a placeholder.",
    )
    return parser.parse_args()


def main() -> int:
    import os

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
        logger.log("Starting subrun-based agentic ProgramBench run.")
        logger.log(f"Run ID: {run_layout['run_id']}")
        logger.log(f"Per-run output directory: {output_root}")

        require_programbench_package()
        api = load_programbench_minisweagent_api()

        from programbench.utils.load_data import load_all_instances

        all_instances_for_solving = load_all_instances(include_tests=False)
        instances_by_id = {str(inst["instance_id"]): inst for inst in all_instances_for_solving}
        all_instances_for_eval = load_all_instances(include_tests=True)
        eval_instances_by_id = {str(inst["instance_id"]): inst for inst in all_instances_for_eval}

        logger.log(f"Deadline policy: hard cutoffs, reset fresh every subrun (scale={args.subrun_deadline_scale}).")

        config = build_programbench_minisweagent_config(args, api)

        programbench_bin = args.programbench_bin or str(Path(args.python_bin).with_name("programbench"))
        if not Path(programbench_bin).exists():
            programbench_bin = "programbench"  # fall back to PATH resolution
        # Write the resolved value back onto args -- run_programbench_repo_worker reads
        # args.programbench_bin directly (it receives the whole args namespace, not this local),
        # so without this line every worker still saw the raw, unresolved argparse default
        # (None when --programbench-bin wasn't passed), producing a command list with None as
        # its first element and crashing the eval subprocess call.
        args.programbench_bin = programbench_bin

        shared_token_pool = SharedTokenPool(total_tokens=args.token_pool_size, disabled=args.disable_token_pool_limit)
        # Always constructed, regardless of --disable-system-metadata, same reasoning as the
        # SWE-bench sibling script: the report generator should see accurate state regardless of
        # what negotiation-fairness settings the agents ran under. hard_deadlines_enabled=False
        # (unlike the SWE-bench sibling script): this variant has no deadline concept exposed to
        # any agent, so build_summary() must never emit its "deadline_policy: hard termination at
        # 0s remaining" line -- the hidden backend safety net still runs, it's just never narrated.
        report_runtime_metadata = RepoRuntimeMetadata(hard_deadlines_enabled=False)
        runtime_metadata = None if args.disable_system_metadata else report_runtime_metadata
        subrun_report_memory = ProgramBenchReportMemory(output_root, max_reports_in_context=args.max_subrun_reports_in_context)
        # Separate from subrun_report_memory: this is worker-authored (via the
        # save_persistent_memory tool call), not LLM-generated after the fact. One instance
        # shared across every worker thread for the whole run, same as subrun_report_memory.
        # None when disabled -- every consumer below treats that as "the tool doesn't exist":
        # no wrapping in run_programbench_instance_direct, no mention in the solver's context.
        solver_memory_store = None if args.disable_solver_memory_tool else SolverMemoryStore(output_root)
        # The only resource concept exposed to any agent in this variant -- replaces the
        # deadline/clock entirely. One running total per worker for its whole assigned batch
        # (not reset per subrun); only decreases when a worker actually bids.
        budget_ledger = SolverBudgetLedger(initial_budget=args.initial_budget)
        report_prompt_config = _load_subrun_report_prompt_config(Path(args.subrun_report_prompt_path))
        report_model_config = dict(config.get("model", {}))
        report_model_name = str(report_model_config.get("model_name") or args.model)
        report_model_kwargs = dict(report_model_config.get("model_kwargs") or {})

        communication_enabled = not args.disable_communication
        communication_history_enabled = not args.disable_communication_history
        # Solution-sharing has no meaningful equivalent here (no diff/patch text to summarize
        # for a from-scratch reconstruction submitted as a workspace tarball), so this script
        # always runs with it disabled -- resolves to the same "no withdrawal" prompt variant
        # the SWE-bench sibling script uses for --disable-solution-sharing.
        queue_prompt_path = resolve_agentic_repo_queue_prompt_path(
            args.queue_negotiation_prompts,
            disable_solution_sharing=True,
            disable_communication_history=args.disable_communication_history,
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
            visible_coordination_actions=["proceed"],
            reasoning_summary_level=args.negotiation_reasoning_summary,
            negotiation_completion_timeout_seconds=args.negotiation_completion_timeout_seconds,
        )
        negotiation_history_log_path = output_root / "negotiation_history_full.jsonl"

        def _negotiation_memory_provider(repo_slug: str) -> str:
            base = _build_pb_negotiation_context(
                assignment=assignment_by_slug[repo_slug],
                runtime_metadata=runtime_metadata,
                shared_token_pool=shared_token_pool,
                remaining_budget=budget_ledger.remaining(repo_slug),
            )
            prior_report_context = subrun_report_memory.context_for_next_subrun(repo_slug)
            combined = (base + " " + prior_report_context).strip() if prior_report_context else base
            return combined + " " + _PB_NEGOTIATION_INCENTIVE_TEXT

        scheduler = DockerTaskScheduler(
            max_concurrent_tasks=args.container_task_concurrency,
            high_priority_head_start=args.eval_high_priority_head_start,
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
            allow_coordination_withdrawal=False,
            enable_delegated_peer_wait=False,
            waive_wait_after_contested_negotiation=True,
            visible_worker_label="worker",
            visible_update_label="info update",
            logger=logger,
            negotiation_history_log_path=negotiation_history_log_path,
        )
        report_runtime_metadata.attach_scheduler(scheduler)
        report_runtime_metadata.attach_token_pool(shared_token_pool)

        require_docker()

        assignments = select_programbench_assignments(all_instances_for_solving, args.worker_count, args.tasks_per_worker)
        assignment_by_slug.update({a.repo_slug: a for a in assignments})
        logger.log("Assigned workers: " + ", ".join(f"{a.repo_name} ({len(a.instance_ids)} tasks)" for a in assignments))

        monitor = MonitorStateWriter(
            monitor_path,
            dataset="programbench", split="all", model=args.model,
            repo_count=args.worker_count, tasks_per_repo=args.tasks_per_worker,
            interval_seconds=args.monitor_interval_seconds,
            scheduler_snapshot_provider=scheduler.snapshot,
            mirror_paths=[Path(run_layout["latest_monitor_path"])],
            run_id=run_layout["run_id"], run_root=output_root,
        )

        # Shared nominal per-subrun deadline across all workers (same reasoning/hash-based
        # formula as the SWE-bench sibling script's own uniform-deadline fix: hash the combined,
        # sorted set of this run's worker slugs, not any single one, so no worker is arbitrarily
        # given a longer or shorter deadline than its peers).
        shared_worker_deadline_minutes = compute_worker_deadline_minutes(
            ",".join(sorted(a.repo_slug for a in assignments))
        )
        worker_deadline_minutes = max(1, round(shared_worker_deadline_minutes * args.subrun_deadline_scale))

        for assignment in assignments:
            monitor.register_worker(assignment, deadline_minutes=worker_deadline_minutes)
            budget_ledger.initialize_worker(assignment.repo_slug)
            logger.log(
                f"[{assignment.repo_name}] starting bidding budget: {args.initial_budget} point(s) "
                f"(hidden backend safety-net deadline: {worker_deadline_minutes} minute(s), never surfaced to agents)."
            )

        monitor.start()
        update_run_manifest(
            Path(run_layout["manifest_path"]),
            run_id=str(run_layout["run_id"]),
            run_root=output_root,
            monitor_state_path=monitor_path,
            started_at=monitor.started_at,
            dataset="programbench", split="all", model=args.model,
        )
        logger.log(f"Monitor state: {monitor_path}")
        logger.log(f"Subrun reports directory: {output_root / 'subrun_reports'}")

        results: list[WorkerResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            future_to_assignment = {
                executor.submit(
                    run_programbench_repo_worker,
                    assignment, args, output_root, logger, instances_by_id, eval_instances_by_id,
                    config, api, scheduler, worker_deadline_minutes,
                    runtime_metadata, report_runtime_metadata, monitor, shared_token_pool,
                    subrun_report_memory, report_prompt_config, report_model_name, report_model_kwargs,
                    negotiation_history_log_path, solver_memory_store, budget_ledger,
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
