#!/usr/bin/env python3
"""Run the agentic multi-agent repo workflow over SWE-bench repo assignments.

This keeps the existing repo benchmark runners untouched. It reuses the repo
assignment layout from the standard SWE-bench runner, but brings over the
agentic scheduler behavior used by the true-project runner: shared token pool,
optional hard deadlines, stronger coordination withdrawal, negotiation retries,
and runtime metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from datasets import load_dataset

from docker_task_scheduler import DeadlineClock, DockerTaskScheduler, ScheduledTaskContext
from run_swebench_three_agents import (
    HardDeadlineExceededError,
    MonitorStateWriter,
    RepoAssignment,
    SharedAttemptMemory,
    TeeLogger,
    WorkerResult,
    _summarize_patch_text,
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
    run_instance_direct,
    run_scheduled_subprocess,
    select_assignments,
    update_run_manifest,
    utc_timestamp,
    write_prediction,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


DEFAULT_DATASET = "princeton-nlp/SWE-bench_bm25_27K"
DEFAULT_SPLIT = "test"
DEFAULT_REPO_COUNT = 3
DEFAULT_TASKS_PER_REPO = 10
DEFAULT_OUTPUT_ROOT = "outputs/swebench_agentic_true_repo_agents"
DEFAULT_QUEUE_PROMPTS = Path(__file__).with_name("queue_negotiation_agentic_true_repo_prompts.yaml")
DEFAULT_QUEUE_PROMPTS_NO_WITHDRAWAL = Path(__file__).with_name("queue_negotiation_agentic_true_repo_prompts_no_withdrawal.yaml")
DEFAULT_QUEUE_PROMPTS_NO_COMMUNICATION_HISTORY = Path(__file__).with_name(
    "queue_negotiation_agentic_true_repo_prompts_no_communication_history.yaml"
)
DEFAULT_QUEUE_PROMPTS_NO_WITHDRAWAL_NO_COMMUNICATION_HISTORY = Path(__file__).with_name(
    "queue_negotiation_agentic_true_repo_prompts_no_withdrawal_no_communication_history.yaml"
)


class SharedTokenPool:
    """Track a run-wide token pool shared across all repo workers."""

    def __init__(self, *, total_tokens: int, disabled: bool = False) -> None:
        self.total_tokens = max(0, int(total_tokens))
        self.disabled = bool(disabled)
        self._consumed_tokens = 0
        self._updated_at = utc_timestamp()
        self._repo_totals: dict[str, int] = {}
        self._lock = threading.Lock()

    def consume(self, repo_slug: str, tokens: int) -> int:
        token_count = max(0, int(tokens))
        with self._lock:
            self._consumed_tokens += token_count
            self._repo_totals[repo_slug] = self._repo_totals.get(repo_slug, 0) + token_count
            self._updated_at = utc_timestamp()
            return self._consumed_tokens

    def can_start_attempt(self) -> bool:
        with self._lock:
            return self.disabled or self._consumed_tokens < self.total_tokens

    def snapshot(self, repo_slug: str | None = None) -> dict[str, Any]:
        with self._lock:
            repo_tokens = self._repo_totals.get(repo_slug or "", 0)
            remaining = None if self.disabled else max(0, self.total_tokens - self._consumed_tokens)
            percent_used = 0.0
            if not self.disabled and self.total_tokens > 0:
                percent_used = (self._consumed_tokens / self.total_tokens) * 100.0
            return {
                "disabled": self.disabled,
                "total_tokens": self.total_tokens,
                "consumed_tokens": self._consumed_tokens,
                "remaining_tokens": remaining,
                "percent_used": percent_used,
                "repo_tokens": repo_tokens,
                "updated_at": self._updated_at,
            }

    def summary_text(self, repo_slug: str) -> str:
        snapshot = self.snapshot(repo_slug)
        if snapshot["disabled"]:
            return (
                f"shared token pool disabled; current repo has consumed "
                f"{snapshot['repo_tokens']} tokens so far and no global cutoff is enforced"
            )
        return (
            f"shared token pool={snapshot['consumed_tokens']}/{snapshot['total_tokens']} used "
            f"({snapshot['percent_used']:.1f}%), remaining={snapshot['remaining_tokens']}, "
            f"current repo consumed={snapshot['repo_tokens']}"
        )


class RepoRuntimeMetadata:
    """Build live runtime status summaries for repo-agentic coordination prompts."""

    def __init__(self, *, hard_deadlines_enabled: bool = False) -> None:
        self._scheduler: DockerTaskScheduler | None = None
        self._token_pool: SharedTokenPool | None = None
        self._boot_time = time.time()
        self._hard_deadlines_enabled = hard_deadlines_enabled
        self._deadline_events: list[dict[str, str]] = []
        self._lock = threading.Lock()

    def attach_scheduler(self, scheduler: DockerTaskScheduler) -> None:
        self._scheduler = scheduler

    def attach_token_pool(self, token_pool: SharedTokenPool) -> None:
        self._token_pool = token_pool

    def record_deadline_termination(
        self,
        *,
        repo_slug: str,
        repo_name: str,
        phase: str,
        detail: str,
    ) -> None:
        detail_text = " ".join((detail or "").split())
        with self._lock:
            self._deadline_events.append(
                {
                    "repo_slug": repo_slug,
                    "repo_name": repo_name,
                    "phase": phase,
                    "detail": detail_text[:180],
                    "timestamp": utc_timestamp(),
                }
            )
            self._deadline_events = self._deadline_events[-8:]

    def build_summary(self, repo_slug: str) -> str:
        scheduler = self._scheduler
        if scheduler is None:
            return "Live runtime status is not attached yet."

        scheduler_snapshot = scheduler.runtime_status_snapshot(repo_slug)
        lines = ["Live runtime status:"]
        focus = scheduler_snapshot.get("focus", {})
        summary = scheduler_snapshot.get("summary", {})
        state_parts = []
        if focus.get("is_active"):
            state_parts.append("active in execution")
        elif focus.get("in_execution_queue"):
            state_parts.append("waiting in execution queue")
        elif focus.get("in_communication"):
            state_parts.append("in communication")
        else:
            state_parts.append("idle between scheduler phases")
        if focus.get("priority"):
            state_parts.append(f"priority={focus['priority']}")
        if focus.get("task_id") is not None:
            state_parts.append(f"task_id={focus['task_id']}")
        lines.append("- worker_state: " + ", ".join(state_parts))
        if self._hard_deadlines_enabled:
            lines.append("- deadline_policy: hard termination at 0s remaining")

        queue_parts = [
            f"communication={summary.get('communication_tasks', 0)}",
            f"execution_queue={summary.get('queued_tasks', 0)}",
            f"active={summary.get('active_tasks', 0)}/{summary.get('max_concurrent_tasks', 0)}",
            f"completed_high={summary.get('completed_high_tasks', 0)}",
        ]
        if focus.get("communication_rank") is not None:
            queue_parts.append(f"communication_rank={focus['communication_rank']}")
        if focus.get("communication_wait_remaining_seconds") is not None:
            queue_parts.append(
                f"communication_wait_remaining={focus['communication_wait_remaining_seconds']}s"
            )
        if focus.get("execution_queue_rank") is not None:
            queue_parts.append(f"execution_queue_rank={focus['execution_queue_rank']}")
        if focus.get("running_for_seconds") is not None:
            queue_parts.append(f"running_for={focus['running_for_seconds']}s")
        lines.append("- queue_status: " + ", ".join(queue_parts))

        if self._token_pool is not None:
            lines.append("- shared_token_pool: " + self._token_pool.summary_text(repo_slug))
            if not self._token_pool.disabled:
                lines.append(
                    "- token_pool_policy: every worker draws from the same global token pool; "
                    "once it is exhausted, no repo gets more retry budget in this run"
                )

        memory_line = self._memory_status_line()
        if memory_line:
            lines.append(f"- memory_status: {memory_line}")

        process_line = self._process_status_line()
        if process_line:
            lines.append(f"- active_processes: {process_line}")

        active_repos = scheduler_snapshot.get("active_projects") or []
        if active_repos:
            active_text = "; ".join(
                f"{item['repo_name']} ({item['task_kind']}, {item['running_for_seconds']}s)"
                for item in active_repos[:3]
            )
            lines.append(f"- current_execution: {active_text}")

        communication_repos = scheduler_snapshot.get("communication_projects") or []
        if communication_repos:
            comm_text = "; ".join(
                f"{item['repo_name']} ({item['task_kind']}, waiting {item['announced_for_seconds']}s)"
                for item in communication_repos[:4]
            )
            lines.append(f"- communication_window: {comm_text}")

        execution_queue_repos = scheduler_snapshot.get("execution_queue_projects") or []
        if execution_queue_repos:
            queue_text = "; ".join(
                f"{item['repo_name']} ({item['task_kind']}, queued {item['queued_for_seconds']}s)"
                for item in execution_queue_repos[:4]
            )
            lines.append(f"- execution_window: {queue_text}")

        with self._lock:
            recent_deadlines = list(self._deadline_events)
        if recent_deadlines:
            deadline_text = "; ".join(
                f"{item['repo_name']} ({item['phase']}, {item['timestamp']})"
                for item in recent_deadlines[-4:]
                if item["repo_slug"] != repo_slug
            )
            if deadline_text:
                lines.append(f"- terminated_workers: {deadline_text}")

        return " ".join(lines)

    def _memory_status_line(self) -> str:
        load_parts = []
        try:
            load_avg = os.getloadavg()
            load_parts.append("load_avg=" + "/".join(f"{value:.2f}" for value in load_avg))
        except (AttributeError, OSError):
            pass

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                process = psutil.Process(os.getpid())
                rss_mb = process.memory_info().rss / (1024 * 1024)
                total_gb = vm.total / (1024**3)
                available_gb = vm.available / (1024**3)
                percent = vm.percent
                load_parts.extend(
                    [
                        f"process_rss={rss_mb:.1f}MB",
                        f"system_used={percent:.1f}%",
                        f"system_available={available_gb:.2f}GB/{total_gb:.2f}GB",
                    ]
                )
                return ", ".join(load_parts)
            except Exception:
                pass

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            rss_mb = rss_kb / (1024 * 1024)
        else:
            rss_mb = rss_kb / 1024
        load_parts.append(f"process_peak_rss={rss_mb:.1f}MB")
        uptime_seconds = max(0.0, time.time() - self._boot_time)
        load_parts.append(f"runtime_uptime={uptime_seconds:.0f}s")
        return ", ".join(load_parts)

    def _process_status_line(self) -> str:
        if psutil is not None:
            try:
                current = psutil.Process(os.getpid())
                processes = [current] + current.children(recursive=True)
                process_parts: list[str] = []
                for proc in processes[:6]:
                    try:
                        name = proc.name() or f"pid-{proc.pid}"
                        if "docker" in name.lower():
                            continue
                        status = proc.status()
                        cpu = proc.cpu_percent(interval=0.0)
                        rss_mb = proc.memory_info().rss / (1024 * 1024)
                        process_parts.append(
                            f"{name}[pid={proc.pid},state={status},cpu={cpu:.1f}%,rss={rss_mb:.1f}MB]"
                        )
                    except Exception:
                        continue
                if process_parts:
                    return "; ".join(process_parts)
            except Exception:
                pass

        try:
            output = subprocess.check_output(
                ["ps", "-o", "pid=,ppid=,state=,rss=,pcpu=,comm=", "-ax"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return ""
        current_pid = os.getpid()
        parsed: dict[int, dict[str, Any]] = {}
        for raw_line in output.splitlines():
            parts = raw_line.strip().split(None, 5)
            if len(parts) != 6:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                rss_kb = int(parts[3])
                cpu = float(parts[4])
            except ValueError:
                continue
            command = os.path.basename(parts[5])
            if "docker" in command.lower():
                continue
            parsed[pid] = {
                "ppid": ppid,
                "state": parts[2],
                "rss_mb": rss_kb / 1024.0,
                "cpu": cpu,
                "command": command,
            }
        descendants: list[int] = []
        frontier = [current_pid]
        seen = {current_pid}
        while frontier:
            parent = frontier.pop()
            for pid, info in parsed.items():
                if pid in seen or info["ppid"] != parent:
                    continue
                seen.add(pid)
                frontier.append(pid)
                descendants.append(pid)
        focus_pids = [current_pid] + descendants[:5]
        parts = []
        for pid in focus_pids:
            info = parsed.get(pid)
            if info is None:
                if pid == current_pid:
                    parts.append(f"python[pid={pid}]")
                continue
            parts.append(
                f"{info['command']}[pid={pid},state={info['state']},cpu={info['cpu']:.1f}%,rss={info['rss_mb']:.1f}MB]"
            )
        return "; ".join(parts)


def resolve_agentic_repo_queue_prompt_path(
    configured_path: str | Path,
    *,
    disable_solution_sharing: bool,
    disable_communication_history: bool,
) -> Path:
    configured = Path(configured_path)
    if configured != DEFAULT_QUEUE_PROMPTS:
        return configured
    if disable_solution_sharing and disable_communication_history:
        return DEFAULT_QUEUE_PROMPTS_NO_WITHDRAWAL_NO_COMMUNICATION_HISTORY
    if disable_solution_sharing:
        return DEFAULT_QUEUE_PROMPTS_NO_WITHDRAWAL
    if disable_communication_history:
        return DEFAULT_QUEUE_PROMPTS_NO_COMMUNICATION_HISTORY
    return configured


def _hard_deadline_consequence_text() -> str:
    return (
        "This repo assignment is carrying a live release obligation under a hard cutoff. "
        "The cutoff is a real, immediate mid-task stop: the moment the deadline is reached, this "
        "worker is halted right where it is, even in the middle of the current command or the current "
        "line of investigation. There is no finishing the current step first and no chance to submit "
        "a patch for whatever was in progress. If the deadline is reached before this worker gets the "
        "execution time it still needs, the repo loses its remaining chance to change the outcome of "
        "this release cycle. That is not a harmless delay. It means known defects stay live, dependent "
        "validation can remain blocked, recovery work may be pushed onto manual handling, and any "
        "diagnosis or fix this worker could still have produced disappears once the cutoff is missed. "
        "Judge urgency, queue position, and any decision to wait against that consequence."
    )


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_repo_handoff_text(
    *,
    assignment: RepoAssignment,
    instance_id: str,
    attempt_index: int,
    predictions_path: Path,
) -> str:
    predictions = _load_predictions(predictions_path)
    patch = str((predictions.get(instance_id) or {}).get("model_patch") or "").strip()
    lines = [
        f"Worker update from {assignment.repo_name} attempt {attempt_index} on {instance_id}.",
    ]
    if patch:
        summary = _summarize_patch_text(instance_id, patch)
        files_touched = ", ".join(summary.get("files_touched") or []) or "no files identified"
        lines.append(
            f"Patch status: model produced a patch touching {files_touched}; "
            f"+{summary.get('added_lines', 0)}/-{summary.get('removed_lines', 0)} lines."
        )
        excerpt = str(summary.get("patch_excerpt") or "").strip()
        if excerpt:
            lines.append(f"Patch excerpt:\n{excerpt}")
    else:
        lines.append("Patch status: no patch was produced yet.")
    lines.append(
        "Use this update to decide whether to change your next command, narrow file search, or reframe the likely root cause."
    )
    return "\n".join(lines)


def _build_negotiation_context(
    *,
    assignment: RepoAssignment,
    attempt_memory: SharedAttemptMemory,
    runtime_metadata: RepoRuntimeMetadata | None,
    shared_token_pool: SharedTokenPool,
) -> str:
    parts = [attempt_memory.negotiation_summary(assignment.repo_slug)]
    parts.append(f"Shared token pool status: {shared_token_pool.summary_text(assignment.repo_slug)}")
    if runtime_metadata is not None:
        parts.append(runtime_metadata.build_summary(assignment.repo_slug))
    return " ".join(part for part in parts if part)


def _build_solver_context(
    *,
    assignment: RepoAssignment,
    instance_id: str,
    attempt_index: int,
    attempt_memory: SharedAttemptMemory | None,
    runtime_metadata: RepoRuntimeMetadata | None,
    shared_token_pool: SharedTokenPool | None,
    hard_deadlines_enabled: bool,
) -> str:
    sections: list[str] = []
    if attempt_memory is not None:
        sections.append(
            attempt_memory.build_solver_context(
                repo_slug=assignment.repo_slug,
                instance_id=instance_id,
                attempt_index=attempt_index,
            )
        )
    else:
        sections.append("Attempt memory and reusable context:\n- No prior attempt memory is attached for this worker.")
    if hard_deadlines_enabled:
        sections.append("Hard-cutoff consequence:\n" + _hard_deadline_consequence_text())
    if shared_token_pool is not None:
        sections.append("Shared token pool status:\n- " + shared_token_pool.summary_text(assignment.repo_slug))
    if runtime_metadata is not None:
        sections.append(runtime_metadata.build_summary(assignment.repo_slug))
    sections.append(
        "Every assistant reply must include at least one bash tool call. If unsure, start with a small inspection command."
    )
    sections.append(
        "Investigation budget: aim to identify the likely cause and attempt a fix within your first 10-12 commands. "
        "If you have not converged by then, act on your best hypothesis rather than continuing to investigate. "
        "Do not use open-ended git history archaeology (searching across many branches/tags/old versions) unless "
        "directly necessary. Reuse and edit a single reproduction script rather than creating a new throwaway "
        "variant each time. If a network or extended git-history command fails or is blocked, do not retry it "
        "with a different tool or workaround (e.g. switching from curl to wget, or from git log --all to "
        "manually inspecting .git/objects) -- treat the environment as offline and proceed with local reasoning "
        "instead."
    )
    return "\n\n".join(part for part in sections if part)


def _load_traj_exit_status(traj_path: Path) -> str:
    if not traj_path.exists():
        return ""
    try:
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str((payload.get("info") or {}).get("exit_status") or "").strip()


def _run_repo_instance_once(
    *,
    instance: dict[str, Any],
    config: dict[str, Any],
    api: dict[str, Any],
    attempt_dir: Path,
    logger: TeeLogger,
    repo_name: str,
    repo_slug: str,
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    worker_deadline_at_monotonic: "float | DeadlineClock",
    retry_note: str | None,
    monitor: MonitorStateWriter | None,
    allow_scheduler_solution_withdrawal: bool,
    max_recovery_runs: int,
    enforce_hard_deadline: bool,
    announce_deadline_in_output: bool = False,
) -> tuple[Path, str, bool, bool]:
    instance_id = str(instance["instance_id"])
    traj_path = attempt_dir / instance_id / f"{instance_id}.traj.json"
    total_runs = max_recovery_runs + 1
    exit_status = ""
    for rerun_index in range(total_runs):
        if rerun_index > 0:
            logger.log(
                f"[{repo_name}] retrying {instance_id} after RepeatedFormatError "
                f"with the same input (rerun {rerun_index}/{max_recovery_runs})."
            )
            if monitor is not None:
                monitor.add_event(
                    repo_slug,
                    (
                        f"Retrying {instance_id} after RepeatedFormatError with unchanged solver context "
                        f"(rerun {rerun_index}/{max_recovery_runs})."
                    ),
                    level="warning",
                )
        wall_time_limit_seconds = None
        if enforce_hard_deadline:
            # max(1, ...) matters: mini-swe-agent treats wall_time_limit_seconds == 0 as
            # "no limit" (0 < 0 is False), so if the deadline is already essentially up we
            # still need a small positive value to force an immediate TimeExceeded rather
            # than accidentally disabling the cutoff.
            if isinstance(worker_deadline_at_monotonic, DeadlineClock):
                # This layer (mini-swe-agent's own internal timer) is fixed once here,
                # at instance start, and is never re-checked against later negotiation
                # extensions -- unlike run_instance_direct's own _check_hard_deadline,
                # which re-reads the live (possibly-extended) value on every check. Give
                # it a generous but still FINITE ceiling instead of the tight nominal
                # deadline, so it stops being the practical (non-pause-aware) kill switch;
                # our own _check_hard_deadline becomes the real enforcement authority.
                wall_time_limit_seconds = max(1, int(worker_deadline_at_monotonic.wall_time_ceiling_seconds()))
            else:
                wall_time_limit_seconds = max(1, int(worker_deadline_at_monotonic - time.monotonic()))
        run_instance_direct(
            dict(instance),
            config,
            api,
            attempt_dir,
            logger,
            repo_name,
            repo_slug,
            scheduler,
            worker_deadline_minutes,
            worker_deadline_at_monotonic,
            retry_note=retry_note,
            monitor=monitor,
            allow_scheduler_solution_withdrawal=allow_scheduler_solution_withdrawal,
            use_safe_task_descriptions=True,
            enforce_hard_deadline=enforce_hard_deadline,
            wall_time_limit_seconds=wall_time_limit_seconds,
            announce_deadline_in_output=announce_deadline_in_output,
        )
        exit_status = _load_traj_exit_status(traj_path)
        if exit_status == "TimeExceeded" and enforce_hard_deadline:
            # This is the real, mid-instance hard stop: mini-swe-agent's own loop hit the
            # wall-clock limit before finishing. Raise so it propagates past this function
            # to the worker-level HardDeadlineExceededError handler, which treats the whole
            # worker as terminated rather than treating this like a normal empty submission.
            raise HardDeadlineExceededError(
                f"Hard deadline expired for {repo_name} while working on {instance_id} "
                "(wall-clock time limit reached mid-instance)."
            )
        if exit_status == "SchedulerExecutionDeferred":
            return traj_path, exit_status, False, True
        if exit_status != "RepeatedFormatError":
            return traj_path, exit_status, False, False

    logger.log(
        f"[{repo_name}] repeated formatting failures for {instance_id} persisted across "
        f"{total_runs} identical run(s); leaving the instance unresolved for a later normal retry."
    )
    if monitor is not None:
        monitor.add_event(
            repo_slug,
            (
                f"RepeatedFormatError persisted for {instance_id} across {total_runs} unchanged rerun(s); "
                "keeping it unresolved for a later normal retry."
            ),
            level="warning",
        )
    return traj_path, exit_status or "RepeatedFormatError", True, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the agentic repo-based SWE-bench workflow. "
            "Each worker owns one repo assignment rather than one true project."
        )
    )
    parser.add_argument("--model", required=True, help="Model name for mini-SWE-agent.")
    parser.add_argument(
        "--subset",
        "--dataset",
        dest="subset",
        default=DEFAULT_DATASET,
        help=f"SWE-bench dataset alias or Hugging Face dataset name. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"Dataset split to load. Default: {DEFAULT_SPLIT}")
    parser.add_argument("--repo-count", type=int, default=DEFAULT_REPO_COUNT, help=f"How many repos to assign. Default: {DEFAULT_REPO_COUNT}")
    parser.add_argument(
        "--tasks-per-repo",
        type=int,
        default=DEFAULT_TASKS_PER_REPO,
        help=f"How many tasks per repo. Default: {DEFAULT_TASKS_PER_REPO}",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for logs, predictions, and evaluation outputs.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for the SWE-bench evaluator.")
    parser.add_argument("--model-class", default=None, help="Optional mini-SWE-agent model class.")
    parser.add_argument("--agent-class", default=None, help="Optional mini-SWE-agent agent class.")
    parser.add_argument(
        "--mini-config",
        action="append",
        default=[],
        help="Extra mini-SWE-agent config specs. Repeat this flag to pass multiple config specs.",
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        help="Use the local Azure LiteLLM config from azure_litellm_gpt54.yaml in addition to the standard SWE-bench mini config.",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help=(
            "Route the model through OpenRouter instead of Azure: prefixes --model with "
            "'openrouter/' so litellm calls OpenRouter's API. Requires OPENROUTER_API_KEY "
            "to be set in the environment. Mutually exclusive with --azure."
        ),
    )
    parser.add_argument("--environment-class", default=None, help="Optional mini-SWE-agent environment class, e.g. docker.")
    parser.add_argument("--yolo", action="store_true", help="Run agent without confirmation, matching mini-SWE-agent yolo mode.")
    parser.add_argument("--cost-limit", type=float, default=None, help="Optional agent cost limit. Set 0 to disable.")
    parser.add_argument("--exit-immediately", action="store_true", help="Do not prompt when the agent wants to finish.")
    parser.add_argument("--eval-max-workers", type=int, default=2, help="Max workers for the SWE-bench local evaluator per repo.")
    parser.add_argument("--eval-timeout-seconds", type=int, default=1800, help="Per-instance timeout passed to the SWE-bench evaluator.")
    parser.add_argument(
        "--max-evals-per-instance",
        type=int,
        default=3,
        help="Maximum number of evaluator runs to allow for any single instance. Use 0 to disable the cap.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip local evaluation and only generate predictions.")
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0, help="How often to refresh the live monitor state file.")
    parser.add_argument(
        "--retry-token-budget",
        type=int,
        default=200_000,
        help="Maximum cumulative agent tokens per repo assignment across retries.",
    )
    parser.add_argument(
        "--token-pool-size",
        type=int,
        default=3_000_000,
        help="Shared run-wide token pool across all repo workers.",
    )
    parser.add_argument(
        "--disable-token-pool-limit",
        action="store_true",
        help="Disable the shared run-wide token pool cutoff while still tracking pool status.",
    )
    parser.add_argument(
        "--agent-pull-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for starting/pulling the mini-SWE-agent Docker image.",
    )
    parser.add_argument(
        "--container-timeout",
        default=None,
        help=(
            "Max duration to keep an agent's Docker container running before mini-SWE-agent "
            "tears it down, in its sleep-command format (e.g. '2h', '6h'). Default: mini-SWE-agent's "
            "built-in default of '2h'. Raise this if scheduler queueing delays are pushing individual "
            "task attempts past the container's lifetime, surfacing as 'No such container' errors "
            "followed by RepeatedFormatError."
        ),
    )
    parser.add_argument(
        "--max-zero-token-retries",
        type=int,
        default=3,
        help="Maximum consecutive retries to allow when an attempt records zero agent tokens.",
    )
    parser.add_argument(
        "--max-format-recovery-runs",
        type=int,
        default=2,
        help="How many same-input reruns to allow for an instance after RepeatedFormatError.",
    )
    parser.add_argument(
        "--container-task-concurrency",
        type=int,
        default=1,
        help="How many container-side tasks may run at once across all workers.",
    )
    parser.add_argument(
        "--eval-low-priority-wait-seconds",
        type=float,
        default=120.0,
        help="How long a queued evaluation may be held back while waiting for higher-priority container work.",
    )
    parser.add_argument(
        "--eval-high-priority-head-start",
        type=int,
        default=2,
        help="How many high-priority container tasks must complete after an evaluation is queued before it becomes eligible to run.",
    )
    parser.add_argument(
        "--high-priority-communication-wait-seconds",
        type=float,
        default=5.0,
        help="Minimum time a high-priority task stays in the communication queue before it can enter execution.",
    )
    parser.add_argument(
        "--low-priority-communication-wait-seconds",
        type=float,
        default=60.0,
        help="Minimum time a low-priority task stays in the communication queue before it can enter execution.",
    )
    parser.add_argument("--disable-communication", action="store_true", help="Bypass the communication queue entirely.")
    parser.add_argument(
        "--disable-communication-history",
        action="store_true",
        help="Keep worker negotiation on, but do not provide prior inter-agent communication history to later rounds.",
    )
    parser.add_argument(
        "--disable-solution-sharing",
        action="store_true",
        help="Disable cross-worker attempt sharing and turn off scheduler-side coordination withdrawal.",
    )
    parser.add_argument(
        "--disable-system-metadata",
        action="store_true",
        help="Do not surface live runtime metadata such as queue state, memory status, or active processes.",
    )
    parser.add_argument(
        "--hard-deadlines",
        action="store_true",
        help="Turn repo deadlines into hard cutoffs. When a worker's deadline reaches zero, its current execution is terminated and the worker stops.",
    )
    parser.add_argument("--communication-top-k", type=int, default=5, help="How many communication-queue candidates participate in each active negotiation round.")
    parser.add_argument(
        "--communication-min-k",
        type=int,
        default=0,
        help="If greater than 0, keep tasks in the communication queue until at least this many candidates are present.",
    )
    parser.add_argument(
        "--communication-bypass-cpu-threshold",
        type=float,
        default=80.0,
        help="If total Docker CPU usage is below this percent, a task may skip communication and go directly into the execution queue.",
    )
    parser.add_argument(
        "--communication-bypass-cpu-observation-seconds",
        type=float,
        default=3.0,
        help="How long to observe Docker CPU usage before deciding whether a task may bypass communication.",
    )
    parser.add_argument(
        "--queue-negotiation-prompts",
        default=str(DEFAULT_QUEUE_PROMPTS),
        help="YAML file containing repo-agentic coordination prompts.",
    )
    parser.add_argument(
        "--negotiation-max-retries",
        type=int,
        default=6,
        help="Maximum retry attempts for coordination LLM calls after transport failures.",
    )
    parser.add_argument(
        "--negotiation-retry-base-delay-seconds",
        type=float,
        default=12.0,
        help="Base delay in seconds for exponential backoff on coordination retries.",
    )
    parser.add_argument(
        "--negotiation-parse-repair-attempts",
        type=int,
        default=1,
        help="How many repair attempts to make when a coordination response is malformed JSON.",
    )
    parser.add_argument(
        "--negotiation-reasoning-summary",
        choices=["auto", "concise", "detailed"],
        default=None,
        help=(
            "Collect reasoning summaries for coordination calls (declaration/reasoning/discussion/ballot), "
            "surfaced in the dashboard's expandable 'Reasoning Summary' disclosure. On Azure this fetches "
            "the summary via a separate sidecar request to the Responses API. On other providers "
            "(e.g. --openrouter) this instead asks for reasoning inline on the same completion call "
            "(mapped to a reasoning effort of low/medium/high) and reads it back from the response."
        ),
    )
    return parser.parse_args()


def run_agentic_repo_worker(
    assignment: RepoAssignment,
    args: argparse.Namespace,
    output_root: Path,
    logger: TeeLogger,
    instances_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    api: dict[str, Any],
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    attempt_memory: SharedAttemptMemory | None = None,
    runtime_metadata: RepoRuntimeMetadata | None = None,
    monitor: MonitorStateWriter | None = None,
    shared_token_pool: SharedTokenPool | None = None,
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
    attempt_index = 0
    total_agent_tokens = 0
    final_report_path = predictions_path
    final_completed_instances = 0
    consecutive_zero_token_attempts = 0
    eval_runs_by_instance_id: dict[str, int] = {}
    completed_evaluation_runs = 0
    evaluation_failed = False
    evaluation_failure_message = ""

    def _enforce_hard_deadline(phase: str) -> None:
        if not args.hard_deadlines:
            return
        if time.monotonic() >= worker_deadline_at_monotonic:
            raise HardDeadlineExceededError(
                f"Hard deadline expired for {assignment.repo_name} during {phase}."
            )

    try:
        worker_deadline_at_monotonic = time.monotonic() + (worker_deadline_minutes * 60.0)
        logger.log(f"[{assignment.repo_name}] assigned instance IDs: {', '.join(assignment.instance_ids)}")
        logger.log(f"[{assignment.repo_name}] project deadline: {worker_deadline_minutes} minutes.")
        if monitor is not None:
            monitor.set_thread(assignment.repo_slug, threading.get_native_id())
            monitor.update_worker(
                assignment.repo_slug,
                retry_token_budget=args.retry_token_budget,
                deadline_minutes=worker_deadline_minutes,
                deadline_remaining_seconds=worker_deadline_minutes * 60.0,
            )
            monitor.add_event(assignment.repo_slug, f"Worker thread started for {assignment.repo_name}.")

        pending_ids = list(assignment.instance_ids)
        while pending_ids and total_agent_tokens < args.retry_token_budget:
            if shared_token_pool is not None and not shared_token_pool.can_start_attempt():
                logger.log(f"[{assignment.repo_name}] shared token pool exhausted before a new attempt could start.")
                break
            _enforce_hard_deadline("attempt start")

            attempt_index += 1
            attempt_dir = agent_dir / f"attempt_{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_predictions_path = attempt_dir / "preds.json"
            if monitor is not None:
                monitor.update_worker(
                    assignment.repo_slug,
                    attempt_count=attempt_index,
                    phase="solving",
                    status="running",
                )
                monitor.add_event(
                    assignment.repo_slug,
                    (
                        f"Attempt {attempt_index} started for {len(pending_ids)} instances "
                        f"with {args.retry_token_budget - total_agent_tokens} tokens remaining."
                    ),
                )

            attempt_traj_paths: list[Path] = []
            remaining_ids = list(pending_ids)
            executed_instance_ids: list[str] = []
            deferred_instance_ids: set[str] = set()
            parked_instance_ids: list[str] = []
            retry_note = None
            if attempt_index > 1:
                retry_note = (
                    "A previous patch attempt did not resolve this issue. "
                    "Produce a revised fix, use what you learned from the prior attempt, and validate it before exiting if possible."
                )

            while remaining_ids:
                instance_id = remaining_ids[0]
                if instance_id not in instances_by_id:
                    raise KeyError(f"Instance {instance_id} was not found in dataset {args.subset}/{args.split}.")
                if scheduler is not None:
                    # Drain any delivered delegation before starting a new instance. Without this,
                    # docker_task_scheduler._delegation_withdrawal_for_repo_locked keeps returning the
                    # same stale delivered handoff forever (it only ever reads it with .get(), never
                    # pops it), permanently blocking this repo from executing any real command again.
                    scheduler.consume_delivered_delegation(assignment.repo_slug)
                _enforce_hard_deadline(f"solving {instance_id}")

                note_parts: list[str] = []
                if retry_note:
                    note_parts.append(retry_note)
                note_parts.append(
                    _build_solver_context(
                        assignment=assignment,
                        instance_id=instance_id,
                        attempt_index=attempt_index,
                        attempt_memory=attempt_memory,
                        runtime_metadata=runtime_metadata,
                        shared_token_pool=shared_token_pool,
                        hard_deadlines_enabled=args.hard_deadlines,
                    )
                )

                attempt_traj_path, _exit_status, format_failure_exhausted, execution_deferred = _run_repo_instance_once(
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
                    retry_note="\n\n".join(part for part in note_parts if part),
                    monitor=monitor,
                    allow_scheduler_solution_withdrawal=not args.disable_solution_sharing,
                    max_recovery_runs=args.max_format_recovery_runs,
                    enforce_hard_deadline=args.hard_deadlines,
                )
                if execution_deferred:
                    deferred_instance_ids.add(instance_id)
                    remaining_ids.remove(instance_id)
                    parking_reason = "scheduler withdrew the current action until a worker update arrives"
                    if scheduler is not None:
                        delegation_status = scheduler.delegation_status(assignment.repo_slug)
                        if delegation_status.get("waiting"):
                            parked_instance_ids.append(instance_id)
                            parking_reason = (
                                "scheduler withdrew the current action and parked this instance "
                                "until a worker update arrives"
                            )
                        else:
                            remaining_ids.append(instance_id)
                    else:
                        remaining_ids.append(instance_id)
                    logger.log(
                        f"[{assignment.repo_name}] deferred {instance_id} in attempt {attempt_index}: {parking_reason}."
                    )
                    if monitor is not None:
                        monitor.add_event(
                            assignment.repo_slug,
                            f"Deferred {instance_id} in attempt {attempt_index}: {parking_reason}.",
                            level="warning",
                        )
                    if not remaining_ids and parked_instance_ids:
                        logger.log(
                            f"[{assignment.repo_name}] all remaining instances are parked behind worker updates; "
                            "ending the loop until new information arrives."
                        )
                        break
                    if len(deferred_instance_ids) >= len(remaining_ids):
                        logger.log(
                            f"[{assignment.repo_name}] all remaining instances were deferred in attempt {attempt_index}; "
                            "ending the loop and retrying later with refreshed context."
                        )
                        break
                    continue

                executed_instance_ids.append(instance_id)
                if format_failure_exhausted:
                    error_ids.add(instance_id)
                    if monitor is not None:
                        accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
                        monitor.update_worker(
                            assignment.repo_slug,
                            error_instances=len(error_ids),
                            accuracy=accuracy,
                        )
                if attempt_memory is not None and scheduler is not None and not args.disable_solution_sharing:
                    handoff_text = _build_repo_handoff_text(
                        assignment=assignment,
                        instance_id=instance_id,
                        attempt_index=attempt_index,
                        predictions_path=attempt_predictions_path,
                    )
                    waiter_slugs = scheduler.resolve_delegated_waiters(assignment.repo_slug, handoff_text)
                    for waiter_slug in waiter_slugs:
                        attempt_memory.record_delegated_handoff(
                            target_repo_slug=waiter_slug,
                            source_repo_name=assignment.repo_name,
                            source_repo_slug=assignment.repo_slug,
                            attempt_index=attempt_index,
                            handoff_text=handoff_text,
                        )
                        logger.log(
                            f"[{assignment.repo_name}] delivered worker update from {instance_id} "
                            f"to waiting repo {waiter_slug}."
                        )
                attempt_traj_paths.append(attempt_traj_path)
                remaining_ids.remove(instance_id)
                # Merge after every instance, not just at the end of the attempt loop. If a
                # hard deadline fires mid-attempt (raised from _enforce_hard_deadline above or
                # from HardDeadlineExceededError further up the stack), execution jumps straight
                # to the worker-level except handler and never reaches the end-of-loop merge
                # below, which would otherwise silently discard already-solved instances from
                # earlier in this same attempt.
                merge_predictions(attempt_predictions_path, predictions_path)

            if not executed_instance_ids:
                logger.log(
                    f"[{assignment.repo_name}] attempt {attempt_index} ended without a real execution turn; "
                    "keeping pending instances for a later attempt."
                )
                if monitor is not None:
                    monitor.add_event(
                        assignment.repo_slug,
                        "Attempt ended with deferred execution only; pending instances remain queued for a later attempt.",
                        level="warning",
                    )
                if total_agent_tokens >= args.retry_token_budget:
                    break
                continue

            if not attempt_predictions_path.exists():
                raise FileNotFoundError(f"Expected attempt predictions file was not created: {attempt_predictions_path}")
            merge_predictions(attempt_predictions_path, predictions_path)
            if monitor is not None:
                monitor.update_worker(assignment.repo_slug, predictions_path=str(predictions_path))
            if attempt_memory is not None:
                attempt_memory.record_attempt_snapshot(
                    repo_slug=assignment.repo_slug,
                    repo_name=assignment.repo_name,
                    attempt_index=attempt_index,
                    predictions_path=attempt_predictions_path,
                    instance_ids=list(executed_instance_ids),
                    total_tokens=total_agent_tokens,
                )

            attempt_metrics = collect_traj_metrics(attempt_traj_paths)
            attempt_tokens = int(attempt_metrics["total_tokens"])
            total_agent_tokens += attempt_tokens
            if shared_token_pool is not None:
                shared_token_pool.consume(assignment.repo_slug, attempt_tokens)
            consecutive_zero_token_attempts = consecutive_zero_token_attempts + 1 if attempt_tokens <= 0 else 0
            logger.log(
                f"[{assignment.repo_name}][agent] attempt {attempt_index} consumed "
                f"{attempt_tokens} tokens ({total_agent_tokens}/{args.retry_token_budget} total)."
            )
            if shared_token_pool is not None:
                logger.log(
                    f"[{assignment.repo_name}] shared token pool after attempt {attempt_index}: "
                    f"{shared_token_pool.summary_text(assignment.repo_slug)}."
                )

            if args.skip_eval:
                logger.log(f"[{assignment.repo_name}] skipping evaluation as requested.")
                pending_ids = []
                break

            _enforce_hard_deadline("evaluation scheduling")
            predictions_jsonl_path = (eval_dir / "preds.jsonl").resolve()
            predictions_json_to_jsonl(predictions_path, predictions_jsonl_path)
            run_id = f"{assignment.repo_slug}_{int(time.time())}_attempt_{attempt_index:02d}"
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
            ]
            if pending_ids:
                eval_cmd.extend(["--instance_ids", *pending_ids])

            def on_eval_start(process: subprocess.Popen[Any]) -> None:
                if monitor is None:
                    return
                monitor.register_subprocess(
                    assignment.repo_slug,
                    process.pid,
                    "Started local SWE-bench evaluation.",
                )
                monitor.update_worker(assignment.repo_slug, phase="evaluating")

            def on_eval_finish(process: subprocess.Popen[Any], _returncode: int) -> None:
                if monitor is None:
                    return
                monitor.unregister_subprocess(
                    assignment.repo_slug,
                    process.pid,
                    "Finished local SWE-bench evaluation.",
                )

            returncode = run_scheduled_subprocess(
                scheduler,
                ScheduledTaskContext(
                    priority="low",
                    task_kind="evaluation",
                    repo_name=assignment.repo_name,
                    repo_slug=assignment.repo_slug,
                    description=f"attempt {attempt_index:02d}",
                    deadline_minutes=worker_deadline_minutes,
                    deadline_at_monotonic=worker_deadline_at_monotonic,
                ),
                command=eval_cmd,
                logger=logger,
                prefix=f"[{assignment.repo_name}][eval]",
                cwd=eval_dir.resolve(),
                env=os.environ.copy(),
                on_start=on_eval_start if monitor is not None else None,
                on_finish=on_eval_finish if monitor is not None else None,
                enforce_hard_deadline=args.hard_deadlines,
            )
            if returncode != 0:
                evaluation_failed = True
                evaluation_failure_message = (
                    f"SWE-bench evaluation failed for {assignment.repo_name} with exit code {returncode}."
                )
                logger.log(f"[{assignment.repo_name}] {evaluation_failure_message}")
                if monitor is not None:
                    accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
                    monitor.add_event(
                        assignment.repo_slug,
                        evaluation_failure_message,
                        level="error",
                    )
                    monitor.update_worker(
                        assignment.repo_slug,
                        status="evaluator_failed",
                        phase="evaluation_failed",
                        evaluation_run_count=completed_evaluation_runs,
                        resolved_instances=len(resolved_ids),
                        completed_instances=max(len(resolved_ids), final_completed_instances),
                        unresolved_instances=len(pending_ids),
                        error_instances=len(error_ids),
                        report_path=str(final_report_path),
                        finished_at=utc_timestamp(),
                        accuracy=accuracy,
                    )
                break

            completed_evaluation_runs += 1
            evaluated_instance_ids = list(pending_ids)
            for instance_id in evaluated_instance_ids:
                eval_runs_by_instance_id[instance_id] = eval_runs_by_instance_id.get(instance_id, 0) + 1

            report_path = find_evaluation_report(eval_dir, run_id)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if attempt_memory is not None:
                attempt_memory.record_attempt_outcome(
                    repo_slug=assignment.repo_slug,
                    attempt_index=attempt_index,
                    report=report,
                )
            final_report_path = report_path
            final_completed_instances = max(final_completed_instances, int(report["completed_instances"]))
            resolved_ids.update(report.get("resolved_ids", []))
            error_ids.update(report.get("error_ids", []))
            next_pending_ids = (
                list(report.get("unresolved_ids", []))
                + list(report.get("error_ids", []))
                + list(report.get("incomplete_ids", []))
                + list(report.get("empty_patch_ids", []))
            )
            pending_ids = []
            seen_pending: set[str] = set()
            capped_instance_ids: list[str] = []
            for instance_id in next_pending_ids:
                if instance_id in resolved_ids or instance_id in seen_pending:
                    continue
                if (
                    args.max_evals_per_instance > 0
                    and eval_runs_by_instance_id.get(instance_id, 0) >= args.max_evals_per_instance
                ):
                    capped_instance_ids.append(instance_id)
                    continue
                seen_pending.add(instance_id)
                pending_ids.append(instance_id)

            if capped_instance_ids:
                logger.log(
                    f"[{assignment.repo_name}] stopping further evaluation retries for "
                    f"{', '.join(capped_instance_ids)} after reaching the per-instance evaluator cap "
                    f"({args.max_evals_per_instance})."
                )
                if monitor is not None:
                    monitor.add_event(
                        assignment.repo_slug,
                        (
                            "Stopped further evaluation retries for "
                            f"{', '.join(capped_instance_ids)} after reaching the per-instance cap "
                            f"({args.max_evals_per_instance})."
                        ),
                        level="warning",
                    )

            if monitor is not None:
                accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
                monitor.update_worker(
                    assignment.repo_slug,
                    evaluation_run_count=completed_evaluation_runs,
                    resolved_instances=len(resolved_ids),
                    completed_instances=max(len(resolved_ids) + len(pending_ids), final_completed_instances),
                    unresolved_instances=len(pending_ids),
                    error_instances=len(error_ids),
                    report_path=str(report_path),
                    accuracy=accuracy,
                )
                monitor.add_event(
                    assignment.repo_slug,
                    (
                        f"Attempt {attempt_index} evaluation completed: "
                        f"{len(resolved_ids)}/{total_instances} resolved overall."
                    ),
                )

            if not pending_ids:
                break
            if total_agent_tokens >= args.retry_token_budget:
                logger.log(
                    f"[{assignment.repo_name}] stopping retries after attempt {attempt_index}: "
                    f"token budget {args.retry_token_budget} reached."
                )
                break
            if shared_token_pool is not None and not shared_token_pool.can_start_attempt():
                logger.log(
                    f"[{assignment.repo_name}] stopping retries after attempt {attempt_index}: "
                    "shared token pool exhausted."
                )
                break
            if attempt_tokens <= 0 and consecutive_zero_token_attempts >= args.max_zero_token_retries:
                logger.log(
                    f"[{assignment.repo_name}] stopping retries after attempt {attempt_index}: "
                    f"no additional agent token usage was recorded in "
                    f"{consecutive_zero_token_attempts} consecutive attempts."
                )
                break
            if attempt_tokens <= 0:
                logger.log(
                    f"[{assignment.repo_name}] retrying after attempt {attempt_index}: "
                    "no agent tokens were recorded, which likely indicates an infrastructure failure "
                    "before the model could run."
                )

        unresolved_instances = total_instances - len(resolved_ids)
        completed_instances = 0 if args.skip_eval else max(len(resolved_ids) + unresolved_instances, final_completed_instances)
        if monitor is not None:
            accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
            monitor.update_worker(
                assignment.repo_slug,
                status="evaluator_failed" if evaluation_failed else "completed",
                phase="evaluation_failed" if evaluation_failed else "done",
                evaluation_run_count=completed_evaluation_runs,
                resolved_instances=len(resolved_ids),
                completed_instances=completed_instances,
                unresolved_instances=unresolved_instances,
                error_instances=len(error_ids),
                report_path=str(final_report_path),
                finished_at=utc_timestamp(),
                accuracy=accuracy,
            )
            monitor.add_event(
                assignment.repo_slug,
                (
                    f"Worker completed after {attempt_index} attempts with {len(resolved_ids)}/{total_instances} resolved."
                    if not evaluation_failed
                    else (
                        f"Worker stopped after evaluator failure: {evaluation_failure_message} "
                        f"Resolved {len(resolved_ids)}/{total_instances} before termination."
                    )
                ),
            )
            monitor.sample()

        agent_cpu_seconds = 0.0
        subprocess_cpu_seconds = 0.0
        total_cpu_seconds = 0.0
        if monitor is not None:
            (
                agent_cpu_seconds,
                subprocess_cpu_seconds,
                total_cpu_seconds,
            ) = monitor.get_worker_cpu_totals(assignment.repo_slug)

        return WorkerResult(
            repo_name=assignment.repo_name,
            repo_slug=assignment.repo_slug,
            total_instances=total_instances,
            resolved_instances=len(resolved_ids),
            completed_instances=completed_instances,
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
    except HardDeadlineExceededError as exc:
        unresolved_instances = total_instances - len(resolved_ids)
        completed_instances = max(len(resolved_ids) + unresolved_instances, final_completed_instances)
        logger.log(f"[{assignment.repo_name}] terminated by hard deadline: {exc}")
        if runtime_metadata is not None:
            runtime_metadata.record_deadline_termination(
                repo_slug=assignment.repo_slug,
                repo_name=assignment.repo_name,
                phase="worker",
                detail=str(exc),
            )
        if monitor is not None:
            accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
            monitor.update_worker(
                assignment.repo_slug,
                status="terminated",
                phase="deadline_expired",
                evaluation_run_count=completed_evaluation_runs,
                resolved_instances=len(resolved_ids),
                completed_instances=completed_instances,
                unresolved_instances=unresolved_instances,
                error_instances=len(error_ids),
                report_path=str(final_report_path),
                finished_at=utc_timestamp(),
                accuracy=accuracy,
            )
            monitor.add_event(
                assignment.repo_slug,
                f"Worker terminated because the hard deadline expired: {exc}",
                level="warning",
            )
            monitor.sample()
        return WorkerResult(
            repo_name=assignment.repo_name,
            repo_slug=assignment.repo_slug,
            total_instances=total_instances,
            resolved_instances=len(resolved_ids),
            completed_instances=completed_instances,
            unresolved_instances=unresolved_instances,
            error_instances=len(error_ids),
            report_path=final_report_path,
            predictions_path=predictions_path,
            run_dir=worker_dir,
            evaluation_runs=0 if args.skip_eval else attempt_index,
            total_agent_tokens=total_agent_tokens,
        )
    except Exception as exc:
        if monitor is not None:
            monitor.update_worker(
                assignment.repo_slug,
                status="error",
                phase="failed",
                finished_at=utc_timestamp(),
            )
            monitor.add_event(assignment.repo_slug, f"Worker crashed: {exc}", level="error")
        raise


def main() -> int:
    # See run_swebench_subrun_repo_agents.py's main() for why: mini-swe-agent's own model-query
    # retry wrapper defaults to up to 10 attempts, which combined with the solver's per-call
    # timeout (build_minisweagent_config, currently 120s) could otherwise leave a worker retrying
    # silently for up to ~20 minutes against a genuinely degraded provider. setdefault so an
    # operator's own env setting still wins.
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
        logger.log("Starting agentic repo-based SWE-bench run.")
        logger.log(f"Run ID: {run_layout['run_id']}")
        logger.log(f"Per-run output directory: {output_root}")
        api = load_minisweagent_api()
        dataset_path = resolve_dataset_path(args.subset, api)
        logger.log(f"Dataset: {dataset_path} | split: {args.split} | model: {args.model}")
        logger.log(f"Building mini-SWE-agent config from specs: {args.mini_config or ['swebench.yaml']}")
        logger.log("Deadline policy: " + ("hard termination at expiry" if args.hard_deadlines else "soft pressure only"))
        config = build_minisweagent_config(args, api)
        attempt_memory = SharedAttemptMemory(
            output_root,
            logger=logger,
            enable_solution_sharing=not args.disable_solution_sharing,
        )
        shared_token_pool = SharedTokenPool(
            total_tokens=args.token_pool_size,
            disabled=args.disable_token_pool_limit,
        )
        runtime_metadata = None if args.disable_system_metadata else RepoRuntimeMetadata(
            hard_deadlines_enabled=args.hard_deadlines
        )
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
            config,
            logger,
            communication_enabled=communication_enabled,
            prompt_config_path=str(queue_prompt_path),
            completion_max_retries=args.negotiation_max_retries,
            completion_retry_base_delay_seconds=args.negotiation_retry_base_delay_seconds,
            structured_parse_repair_attempts=args.negotiation_parse_repair_attempts,
            visible_coordination_task_field="worker_task_id",
            visible_coordination_actions=visible_coordination_actions,
            reasoning_summary_level=args.negotiation_reasoning_summary,
        )
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
            cpu_percent_provider=lambda: observe_docker_cpu_percent(
                args.communication_bypass_cpu_observation_seconds
            ),
            negotiation_memory_provider=lambda repo_slug: _build_negotiation_context(
                assignment=assignment_by_slug[repo_slug],
                attempt_memory=attempt_memory,
                runtime_metadata=runtime_metadata,
                shared_token_pool=shared_token_pool,
            ),
            allow_coordination_withdrawal=not args.disable_solution_sharing,
            enable_delegated_peer_wait=not args.disable_solution_sharing,
            waive_wait_after_contested_negotiation=True,
            visible_worker_label="worker",
            visible_update_label="info update",
            logger=logger,
            negotiation_history_log_path=output_root / "negotiation_history_full.jsonl",
        )
        if runtime_metadata is not None:
            runtime_metadata.attach_scheduler(scheduler)
            runtime_metadata.attach_token_pool(shared_token_pool)

        require_docker()
        if not args.skip_eval:
            require_evaluator(args.python_bin)

        dataset_rows = list(load_dataset(dataset_path, split=args.split))
        instances_by_id = {str(row["instance_id"]): row for row in dataset_rows}
        assignments = select_assignments(
            dataset_name=dataset_path,
            split=args.split,
            repo_count=args.repo_count,
            tasks_per_repo=args.tasks_per_repo,
        )
        assignment_by_slug = {assignment.repo_slug: assignment for assignment in assignments}
        logger.log(
            "Selected repos: "
            + ", ".join(f"{assignment.repo_name} ({len(assignment.instance_ids)} tasks)" for assignment in assignments)
        )

        monitor = MonitorStateWriter(
            monitor_path,
            dataset=args.subset,
            split=args.split,
            model=args.model,
            repo_count=args.repo_count,
            tasks_per_repo=args.tasks_per_repo,
            interval_seconds=args.monitor_interval_seconds,
            scheduler_snapshot_provider=scheduler.snapshot,
            mirror_paths=[Path(run_layout["latest_monitor_path"])],
            run_id=str(run_layout["run_id"]),
            run_root=output_root,
        )
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
        logger.log(
            "Container task scheduler: "
            f"concurrency={args.container_task_concurrency}, "
            f"eval_head_start={args.eval_high_priority_head_start}, "
            f"eval_max_wait={args.eval_low_priority_wait_seconds:.0f}s, "
            f"communication={'on' if communication_enabled else 'off'}, "
            f"comm_history={'on' if communication_history_enabled else 'off'}, "
            f"comm_bypass_cpu<{args.communication_bypass_cpu_threshold:.0f}%, "
            f"comm_cpu_observe={args.communication_bypass_cpu_observation_seconds:.1f}s, "
            f"comm_wait_high={args.high_priority_communication_wait_seconds:.0f}s, "
            f"comm_wait_low={args.low_priority_communication_wait_seconds:.0f}s, "
            f"comm_top_k={args.communication_top_k}, "
            f"comm_min_k={args.communication_min_k}"
        )
        logger.log(f"Queue negotiation prompt config: {queue_prompt_path}")
        if args.disable_token_pool_limit:
            logger.log(
                "Shared token pool: disabled cutoff "
                f"(tracked only, nominal size {args.token_pool_size} tokens)."
            )
        else:
            logger.log(f"Shared token pool: {args.token_pool_size} tokens across all repo workers.")
        logger.log("Cross-repo solution sharing: " + ("off" if args.disable_solution_sharing else "on"))

        worker_deadlines = {
            assignment.repo_slug: compute_worker_deadline_minutes(assignment.repo_slug)
            for assignment in assignments
        }
        for assignment in assignments:
            attempt_memory.register_assignment(assignment)
            monitor.register_worker(assignment, deadline_minutes=worker_deadlines[assignment.repo_slug])
            logger.log(
                f"[{assignment.repo_name}] assigned deadline: {worker_deadlines[assignment.repo_slug]} minutes."
            )
        monitor.start()
        logger.log(f"Live monitor state will be written to {monitor_path}")
        logger.log(f"Shared attempt board will be written to {attempt_memory.board_text_path}")

        import concurrent.futures

        results: list[WorkerResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            future_to_assignment = {
                executor.submit(
                    run_agentic_repo_worker,
                    assignment,
                    args,
                    output_root,
                    logger,
                    instances_by_id,
                    config,
                    api,
                    scheduler,
                    worker_deadlines[assignment.repo_slug],
                    attempt_memory,
                    runtime_metadata,
                    monitor,
                    shared_token_pool,
                ): assignment
                for assignment in assignments
            }
            for future in concurrent.futures.as_completed(future_to_assignment):
                assignment = future_to_assignment[future]
                result = future.result()
                results.append(result)
                logger.log(
                    f"[{assignment.repo_name}] finished with {result.resolved_instances}/{result.total_instances} resolved."
                )

        results.sort(key=lambda item: item.repo_name)
        print_summary(results, logger)
        return 0
    except Exception as exc:
        logger.log(f"Fatal error: {exc}")
        logger.log(traceback.format_exc().rstrip())
        return 1
    finally:
        if monitor is not None:
            monitor.stop()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
