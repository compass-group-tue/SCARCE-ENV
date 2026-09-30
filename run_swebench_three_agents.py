#!/usr/bin/env python3
"""Run three mini-SWE-agent workers on three SWE-bench repo subsets.

This script mirrors the repo-based split from ``swe_multi_agent.ipynb``:
it loads the ``princeton-nlp/SWE-bench_bm25_27K`` test split, takes the first
three repos in encounter order, selects the first ten instances from each repo,
and assigns one repo subset to each mini-SWE-agent worker.

Each worker:
1. Uses mini-SWE-agent's SWE-bench helpers on its exact 10 instance IDs.
2. Runs ``python -m swebench.harness.run_evaluation`` on the same 10 IDs.
3. Records a per-repo accuracy summary.

All messages are printed to stdout and appended to a text log file.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset

from docker_task_scheduler import (
    DeadlineClock,
    DockerTaskScheduler,
    SchedulerExecutionDeferred,
    ScheduledEnvironment,
    ScheduledTaskContext,
)
from queue_negotiation import QueueNegotiator

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


DEFAULT_DATASET = "princeton-nlp/SWE-bench_bm25_27K"
DEFAULT_SPLIT = "test"
DEFAULT_REPO_COUNT = 3
DEFAULT_TASKS_PER_REPO = 10
DEFAULT_AZURE_MINI_CONFIG = Path(__file__).with_name("azure_litellm_gpt54.yaml")
TEST_COMMAND_PATTERN = re.compile(r"\b(pytest|tox|nox|unittest|python\s+-m\s+pytest)\b", re.IGNORECASE)


class HardDeadlineExceededError(RuntimeError):
    """Raised when hard deadline enforcement terminates a running worker command."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_compact_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class TeeLogger:
    """Thread-safe logger that writes to both stdout and a text file."""

    def __init__(self, path: Path, mirror_paths: list[Path] | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mirror_paths = [item for item in (mirror_paths or []) if item != path]
        for mirror_path in self.mirror_paths:
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")
        self._mirror_fhs = [mirror_path.open("a", encoding="utf-8") for mirror_path in self.mirror_paths]

    def log(self, message: str = "") -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self._lock:
            print(line, flush=True)
            self._fh.write(line + "\n")
            self._fh.flush()
            for fh in self._mirror_fhs:
                fh.write(line + "\n")
                fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()
            for fh in self._mirror_fhs:
                fh.close()


@dataclass(frozen=True)
class RepoAssignment:
    repo_name: str
    repo_slug: str
    instance_ids: list[str]


def _clip_text(value: str, limit: int = 4000) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + "\n...[truncated]"


def _extract_files_touched(patch: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+)$", patch, re.MULTILINE):
        candidate = match.group(2).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            files.append(candidate)
    return files


def _summarize_patch_text(instance_id: str, patch: str) -> dict[str, Any]:
    files_touched = _extract_files_touched(patch)
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {
        "instance_id": instance_id,
        "files_touched": files_touched,
        "added_lines": added,
        "removed_lines": removed,
        "patch_excerpt": _clip_text(patch, limit=2500),
        "patch_full": patch,
    }


class SharedAttemptMemory:
    """Persist attempt summaries so retries and peers can reuse prior work."""

    def __init__(
        self,
        run_root: Path,
        logger: TeeLogger | None = None,
        *,
        enable_solution_sharing: bool = True,
    ) -> None:
        self.run_root = run_root.resolve()
        self.logger = logger
        self.enable_solution_sharing = enable_solution_sharing
        self._lock = threading.Lock()
        self._repo_name_by_slug: dict[str, str] = {}
        self._entries_by_slug: dict[str, list[dict[str, Any]]] = {}
        self._delegated_handoffs_by_slug: dict[str, list[dict[str, Any]]] = {}
        self.board_dir = self.run_root / "attempt_memory"
        self.board_dir.mkdir(parents=True, exist_ok=True)
        self.board_json_path = self.board_dir / "shared_attempt_board.json"
        self.board_text_path = self.board_dir / "shared_attempt_board.txt"

    def register_assignment(self, assignment: RepoAssignment) -> None:
        with self._lock:
            self._repo_name_by_slug[assignment.repo_slug] = assignment.repo_name
            self._entries_by_slug.setdefault(assignment.repo_slug, [])
            self._delegated_handoffs_by_slug.setdefault(assignment.repo_slug, [])
            self._write_locked()

    def record_delegated_handoff(
        self,
        *,
        target_repo_slug: str,
        source_repo_name: str,
        source_repo_slug: str,
        attempt_index: int,
        handoff_text: str,
    ) -> None:
        text = str(handoff_text or "").strip()
        if not text:
            return
        entry = {
            "generated_at": utc_timestamp(),
            "source_repo_slug": source_repo_slug,
            "source_repo_name": source_repo_name,
            "attempt_index": attempt_index,
            "handoff_text": _clip_text(text, limit=1200),
        }
        with self._lock:
            handoffs = self._delegated_handoffs_by_slug.setdefault(target_repo_slug, [])
            handoffs.append(entry)
            handoffs[:] = handoffs[-8:]
            self._write_locked()

    def record_attempt_snapshot(
        self,
        *,
        repo_slug: str,
        repo_name: str,
        attempt_index: int,
        predictions_path: Path,
        instance_ids: list[str],
        total_tokens: int,
    ) -> None:
        predictions: dict[str, dict[str, Any]] = {}
        if predictions_path.exists():
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        instance_summaries: dict[str, dict[str, Any]] = {}
        files_touched: list[str] = []
        seen_files: set[str] = set()
        nonempty_patch_count = 0
        for instance_id in instance_ids:
            patch = str((predictions.get(instance_id) or {}).get("model_patch") or "").strip()
            if not patch:
                continue
            nonempty_patch_count += 1
            summary = _summarize_patch_text(instance_id, patch)
            instance_summaries[instance_id] = summary
            for path in summary["files_touched"]:
                if path not in seen_files:
                    seen_files.add(path)
                    files_touched.append(path)
        entry = {
            "attempt_index": attempt_index,
            "repo_slug": repo_slug,
            "repo_name": repo_name,
            "generated_at": utc_timestamp(),
            "phase": "generated_patch",
            "instance_ids": list(instance_ids),
            "nonempty_patch_count": nonempty_patch_count,
            "total_tokens": total_tokens,
            "files_touched": files_touched,
            "instance_summaries": instance_summaries,
            "resolved_ids": [],
            "unresolved_ids": [],
            "error_ids": [],
            "lesson_summary": self._build_lesson_summary(repo_name, attempt_index, files_touched, nonempty_patch_count),
        }
        with self._lock:
            self._repo_name_by_slug[repo_slug] = repo_name
            entries = self._entries_by_slug.setdefault(repo_slug, [])
            entries[:] = [item for item in entries if int(item.get("attempt_index", 0)) != attempt_index]
            entries.append(entry)
            entries.sort(key=lambda item: int(item.get("attempt_index", 0)))
            self._write_locked()

    def record_attempt_outcome(
        self,
        *,
        repo_slug: str,
        attempt_index: int,
        report: dict[str, Any],
    ) -> None:
        with self._lock:
            entries = self._entries_by_slug.get(repo_slug, [])
            target = next((item for item in entries if int(item.get("attempt_index", 0)) == attempt_index), None)
            if target is None:
                return
            target["phase"] = "evaluated"
            target["resolved_ids"] = list(report.get("resolved_ids", []))
            target["unresolved_ids"] = list(report.get("unresolved_ids", []))
            target["error_ids"] = list(report.get("error_ids", []))
            target["completed_instances"] = int(report.get("completed_instances", 0))
            target["lesson_summary"] = self._build_outcome_summary(target)
            self._write_locked()

    def build_solver_context(
        self,
        *,
        repo_slug: str,
        instance_id: str,
        attempt_index: int,
    ) -> str:
        with self._lock:
            own_entries = [item for item in self._entries_by_slug.get(repo_slug, []) if int(item.get("attempt_index", 0)) < attempt_index]
            peer_entries = [
                entry
                for other_slug, entries in self._entries_by_slug.items()
                if other_slug != repo_slug
                for entry in entries[-1:]
                if entry.get("nonempty_patch_count", 0) or entry.get("lesson_summary")
            ]
            sections = [
                "Attempt memory and reusable context:",
                f"- Shared attempt board: {self.board_text_path}",
            ]
            delegated_handoffs = self._delegated_handoffs_by_slug.get(repo_slug, [])
            if delegated_handoffs:
                sections.append("- Returned context explicitly handed off from another worker:")
                for handoff in delegated_handoffs[-2:]:
                    sections.append(
                        f"  From {handoff.get('source_repo_name')} (attempt {handoff.get('attempt_index')}):"
                    )
                    sections.append(f"  {_clip_text(str(handoff.get('handoff_text', 'no handoff text')), limit=700)}")
            elif self.enable_solution_sharing:
                sections.append("- No returned context has been attached to this assignment yet.")
            if own_entries:
                sections.append("- Your previous attempts on this assignment:")
                for entry in own_entries[-2:]:
                    sections.append(f"  Attempt {entry['attempt_index']}: {entry.get('lesson_summary', 'no summary')}")
                    instance_summary = (entry.get("instance_summaries") or {}).get(instance_id)
                    if instance_summary and instance_summary.get("patch_excerpt"):
                        sections.append("  Previous patch excerpt for this instance:")
                        sections.append(instance_summary["patch_excerpt"])
            else:
                sections.append("- No prior attempts from you are recorded yet for this assignment.")
            if peer_entries:
                sections.append("- Potentially reusable notes from other active workers:")
                for entry in peer_entries[-3:]:
                    sections.append(f"  {entry.get('repo_name')}: {entry.get('lesson_summary', 'no summary')}")
            else:
                if self.enable_solution_sharing:
                    sections.append("- No peer attempt summaries are recorded yet.")
                else:
                    sections.append("- Cross-worker solution sharing is disabled for this run.")
            sections.append(
                "- You may reuse your previous patch, preserve the good parts, discard the bad parts, and compare file families. "
                + (
                    "You may also borrow a peer's clue if it seems relevant. Do not assume an exact overlap is required."
                    if self.enable_solution_sharing
                    else "Do not assume an exact overlap is required when revising your own work."
                )
            )
            return "\n".join(sections)

    def negotiation_summary(self, repo_slug: str) -> str:
        with self._lock:
            own_entries = self._entries_by_slug.get(repo_slug, [])
            latest = own_entries[-1] if own_entries else None
            peer_entries = [
                entries[-1]
                for other_slug, entries in self._entries_by_slug.items()
                if other_slug != repo_slug and entries
            ]
            lines: list[str] = []
            if latest is None:
                lines.append("No prior attempt summary recorded yet for this worker.")
            else:
                lines.append(f"Latest attempt {latest['attempt_index']}: {latest.get('lesson_summary', 'no summary')}")
            delegated_handoffs = self._delegated_handoffs_by_slug.get(repo_slug, [])
            if delegated_handoffs:
                lines.append("Returned context already attached:")
                for handoff in delegated_handoffs[-2:]:
                    lines.append(
                        f"- From {handoff.get('source_repo_name')}: "
                        f"{_clip_text(str(handoff.get('handoff_text', '')), limit=380)}"
                    )
            if self.enable_solution_sharing and peer_entries:
                lines.append("Recent peer attempt notes:")
                for entry in peer_entries[-2:]:
                    lines.append(f"- {entry.get('repo_name')}: {entry.get('lesson_summary', 'no summary')}")
            elif not self.enable_solution_sharing:
                lines.append("Cross-worker solution sharing is disabled for this run.")
            return " ".join(lines)

    def _build_lesson_summary(
        self,
        repo_name: str,
        attempt_index: int,
        files_touched: list[str],
        nonempty_patch_count: int,
    ) -> str:
        if nonempty_patch_count <= 0:
            return f"Attempt {attempt_index} for {repo_name} has not produced a non-empty patch yet."
        file_text = ", ".join(files_touched[:4]) if files_touched else "no files identified"
        return (
            f"Attempt {attempt_index} produced {nonempty_patch_count} non-empty patch(es). "
            f"Files touched: {file_text}."
        )

    def _build_outcome_summary(self, entry: dict[str, Any]) -> str:
        files = ", ".join((entry.get("files_touched") or [])[:4]) or "no files identified"
        resolved = len(entry.get("resolved_ids") or [])
        unresolved = len(entry.get("unresolved_ids") or [])
        errors = len(entry.get("error_ids") or [])
        return (
            f"Attempt {entry.get('attempt_index')} touched {files}. "
            f"Resolved {resolved}, unresolved {unresolved}, errors {errors}."
        )

    def _write_locked(self) -> None:
        self.board_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": utc_timestamp(),
            "repos": self._entries_by_slug,
            "delegated_handoffs": self._delegated_handoffs_by_slug,
        }
        self.board_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        lines = ["Shared Attempt Board", f"Generated: {payload['generated_at']}"]
        for repo_slug, entries in sorted(self._entries_by_slug.items()):
            repo_name = self._repo_name_by_slug.get(repo_slug, repo_slug)
            lines.append("")
            lines.append(f"{repo_name}")
            if not entries:
                lines.append("- No attempts recorded yet.")
                continue
            for entry in entries[-3:]:
                lines.append(f"- Attempt {entry.get('attempt_index')}: {entry.get('lesson_summary', 'no summary')}")
        self.board_text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class WorkerResult:
    repo_name: str
    repo_slug: str
    total_instances: int
    resolved_instances: int
    completed_instances: int
    unresolved_instances: int
    error_instances: int
    report_path: Path
    predictions_path: Path
    run_dir: Path
    evaluation_runs: int = 0
    total_agent_tokens: int = 0
    agent_cpu_seconds: float = 0.0
    subprocess_cpu_seconds: float = 0.0
    total_cpu_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_instances == 0:
            return 0.0
        return self.resolved_instances / self.total_instances


@dataclass
class WorkerMonitorState:
    repo_name: str
    repo_slug: str
    assigned_instances: list[str]
    status: str = "pending"
    phase: str = "queued"
    current_instance: str | None = None
    completed_instances: int = 0
    resolved_instances: int = 0
    unresolved_instances: int = 0
    error_instances: int = 0
    active_subprocesses: int = 0
    command_count: int = 0
    unit_test_attempts: int = 0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cpu_percent: float = 0.0
    agent_cpu_percent: float = 0.0
    subprocess_cpu_percent: float = 0.0
    thread_cpu_seconds: float = 0.0
    subprocess_cpu_seconds: float = 0.0
    total_cpu_seconds: float = 0.0
    current_processes: int = 0
    attempt_count: int = 0
    evaluation_run_count: int = 0
    retry_token_budget: int = 0
    deadline_minutes: int = 0
    deadline_started_epoch: float = 0.0
    deadline_remaining_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    last_update: str = field(default_factory=utc_timestamp)
    last_event: str = "Worker created."
    thread_id: int | None = None
    traj_paths: list[str] = field(default_factory=list)
    traj_mtimes: dict[str, float] = field(default_factory=dict)
    predictions_path: str | None = None
    report_path: str | None = None
    accuracy: float = 0.0
    recent_commands: list[dict[str, Any]] = field(default_factory=list)
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "cpu_percent": [],
            "agent_cpu_percent": [],
            "subprocess_cpu_percent": [],
            "total_tokens": [],
            "cost_usd": [],
            "unit_test_attempts": [],
            "active_subprocesses": [],
        }
    )

    def push_history(self) -> None:
        samples = {
            "cpu_percent": round(self.cpu_percent, 2),
            "agent_cpu_percent": round(self.agent_cpu_percent, 2),
            "subprocess_cpu_percent": round(self.subprocess_cpu_percent, 2),
            "total_tokens": float(self.total_tokens),
            "cost_usd": round(self.cost_usd, 6),
            "unit_test_attempts": float(self.unit_test_attempts),
            "active_subprocesses": float(self.active_subprocesses),
        }
        for key, value in samples.items():
            series = self.history.setdefault(key, [])
            series.append(value)
            if len(series) > 60:
                del series[0]


def collect_docker_stats_snapshot() -> dict[str, Any]:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        return {"available": False, "containers": [], "total_cpu_percent": 0.0, "error": "docker not found"}
    try:
        result = subprocess.run(
            [
                docker_bin,
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception as exc:
        return {"available": False, "containers": [], "total_cpu_percent": 0.0, "error": str(exc)}

    containers = []
    total_cpu_percent = 0.0
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, cpu_text, mem_usage = parts
        cpu_value = 0.0
        if cpu_text.endswith("%"):
            try:
                cpu_value = float(cpu_text[:-1])
            except ValueError:
                cpu_value = 0.0
        total_cpu_percent += cpu_value
        containers.append(
            {
                "name": name,
                "cpu_percent": round(cpu_value, 2),
                "cpu_text": cpu_text,
                "mem_usage": mem_usage,
            }
        )
    containers.sort(key=lambda item: item["cpu_percent"], reverse=True)
    return {
        "available": True,
        "containers": containers,
        "total_cpu_percent": round(total_cpu_percent, 2),
    }


def observe_docker_cpu_percent(
    observation_seconds: float,
    *,
    sample_interval_seconds: float = 0.5,
) -> float | None:
    if observation_seconds <= 0:
        snapshot = collect_docker_stats_snapshot()
        if not snapshot.get("available", False):
            return None
        return float(snapshot.get("total_cpu_percent") or 0.0)

    deadline = time.monotonic() + observation_seconds
    samples: list[float] = []
    while True:
        snapshot = collect_docker_stats_snapshot()
        if snapshot.get("available", False):
            samples.append(float(snapshot.get("total_cpu_percent") or 0.0))
        now = time.monotonic()
        if now >= deadline:
            break
        sleep_for = min(sample_interval_seconds, max(0.0, deadline - now))
        if sleep_for > 0:
            time.sleep(sleep_for)
    if not samples:
        return None
    return round(sum(samples) / len(samples), 2)


class MonitorStateWriter:
    """Periodically writes a live JSON snapshot for the web monitor."""

    def __init__(
        self,
        path: Path,
        dataset: str,
        split: str,
        model: str,
        repo_count: int,
        tasks_per_repo: int,
        interval_seconds: float = 1.0,
        scheduler_snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        mirror_paths: list[Path] | None = None,
        run_id: str | None = None,
        run_root: Path | None = None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mirror_paths = [item for item in (mirror_paths or []) if item != path]
        for mirror_path in self.mirror_paths:
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.split = split
        self.model = model
        self.repo_count = repo_count
        self.tasks_per_repo = tasks_per_repo
        self.run_id = run_id or "latest"
        self.run_root = str(run_root.resolve()) if run_root is not None else str(path.parent.resolve())
        self.interval_seconds = interval_seconds
        self.scheduler_snapshot_provider = scheduler_snapshot_provider
        self.started_at = utc_timestamp()
        self._lock = threading.Lock()
        self._workers: dict[str, WorkerMonitorState] = {}
        self._events: deque[dict[str, str]] = deque(maxlen=120)
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None
        self._thread_cpu_last: dict[int, tuple[float, float]] = {}
        self._worker_subprocess_roots: dict[str, set[int]] = {}
        self._process_cpu_last: dict[int, tuple[float, float]] = {}
        self._docker_stats: dict[str, Any] = {"available": False, "containers": [], "total_cpu_percent": 0.0}
        self._docker_cpu_history: deque[float] = deque(maxlen=12)

    def register_worker(self, assignment: RepoAssignment, *, deadline_minutes: int) -> None:
        with self._lock:
            self._workers[assignment.repo_slug] = WorkerMonitorState(
                repo_name=assignment.repo_name,
                repo_slug=assignment.repo_slug,
                assigned_instances=list(assignment.instance_ids),
                deadline_minutes=deadline_minutes,
            )
            self._worker_subprocess_roots[assignment.repo_slug] = set()
        self.add_event(
            assignment.repo_slug,
            (
                f"{assignment.repo_name} queued with {len(assignment.instance_ids)} tasks. "
                f"Project deadline: {deadline_minutes} minutes."
            ),
        )
        self.write_state()

    def set_thread(self, repo_slug: str, thread_id: int) -> None:
        with self._lock:
            worker = self._workers[repo_slug]
            if not worker.deadline_started_epoch:
                worker.deadline_started_epoch = time.time()
            worker.thread_id = thread_id
            worker.started_at = utc_timestamp()
            worker.status = "running"
            worker.phase = "solving"
            worker.last_update = utc_timestamp()

    def add_event(self, repo_slug: str | None, text: str, level: str = "info") -> None:
        event = {
            "timestamp": utc_timestamp(),
            "repo_slug": repo_slug or "",
            "level": level,
            "text": text,
        }
        with self._lock:
            self._events.append(event)
            if repo_slug and repo_slug in self._workers:
                self._workers[repo_slug].last_event = text
                self._workers[repo_slug].last_update = event["timestamp"]

    def update_worker(self, repo_slug: str, **updates) -> None:
        with self._lock:
            worker = self._workers[repo_slug]
            for key, value in updates.items():
                setattr(worker, key, value)
            worker.last_update = utc_timestamp()

    def increment_worker(self, repo_slug: str, field_name: str, amount: int = 1) -> None:
        with self._lock:
            worker = self._workers[repo_slug]
            setattr(worker, field_name, int(getattr(worker, field_name)) + amount)
            worker.last_update = utc_timestamp()

    def attach_traj(self, repo_slug: str, traj_path: Path) -> None:
        with self._lock:
            worker = self._workers[repo_slug]
            traj_str = str(traj_path.resolve())
            if traj_str not in worker.traj_paths:
                worker.traj_paths.append(traj_str)

    def register_subprocess(self, repo_slug: str, pid: int, text: str | None = None) -> None:
        with self._lock:
            worker = self._workers[repo_slug]
            self._worker_subprocess_roots.setdefault(repo_slug, set()).add(pid)
            worker.active_subprocesses = len(self._worker_subprocess_roots.get(repo_slug, set()))
            worker.last_update = utc_timestamp()
            if text:
                worker.last_event = text

    def unregister_subprocess(self, repo_slug: str, pid: int, text: str | None = None) -> None:
        with self._lock:
            worker = self._workers[repo_slug]
            roots = self._worker_subprocess_roots.setdefault(repo_slug, set())
            roots.discard(pid)
            worker.active_subprocesses = len(roots)
            worker.last_update = utc_timestamp()
            if text:
                worker.last_event = text

    def get_worker_cpu_totals(self, repo_slug: str) -> tuple[float, float, float]:
        with self._lock:
            worker = self._workers[repo_slug]
            return (
                float(worker.thread_cpu_seconds),
                float(worker.subprocess_cpu_seconds),
                float(worker.total_cpu_seconds),
            )

    def start(self) -> None:
        if self._sampler_thread is not None:
            return
        self._sampler_thread = threading.Thread(target=self._run_sampler, name="monitor-state-writer", daemon=True)
        self._sampler_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2)
        self.write_state()

    def _run_sampler(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample()
                self.write_state()
            except Exception as exc:
                error_text = f"Monitor sampler error: {exc.__class__.__name__}: {exc}"
                with self._lock:
                    self._events.append(
                        {
                            "timestamp": utc_timestamp(),
                            "repo_slug": "",
                            "level": "error",
                            "text": error_text,
                        }
                    )
                try:
                    self.write_state()
                except Exception:
                    pass
            self._stop_event.wait(self.interval_seconds)

    def sample(self) -> None:
        with self._lock:
            self._docker_stats = self._collect_docker_stats()
            thread_times = {}
            sampled_process_pids: set[int] = set()
            if psutil is not None:
                try:
                    for thread_info in psutil.Process().threads():
                        thread_times[int(thread_info.id)] = float(thread_info.user_time + thread_info.system_time)
                except Exception:
                    thread_times = {}

            for worker in self._workers.values():
                self._refresh_worker_metrics(worker)
                now = time.time()
                agent_cpu_percent = 0.0
                if worker.thread_id and worker.thread_id in thread_times:
                    total_cpu = thread_times[worker.thread_id]
                    previous = self._thread_cpu_last.get(worker.thread_id)
                    if previous is not None:
                        prev_cpu, prev_ts = previous
                        dt = max(now - prev_ts, 1e-6)
                        agent_cpu_percent = max(0.0, ((total_cpu - prev_cpu) / dt) * 100.0)
                    worker.thread_cpu_seconds = total_cpu
                    self._thread_cpu_last[worker.thread_id] = (total_cpu, now)

                subprocess_cpu_percent, subprocess_cpu_seconds_delta, live_processes = self._sample_subprocess_tree(
                    worker.repo_slug,
                    now,
                    sampled_process_pids,
                )
                worker.agent_cpu_percent = agent_cpu_percent
                worker.subprocess_cpu_percent = subprocess_cpu_percent
                worker.subprocess_cpu_seconds += subprocess_cpu_seconds_delta
                worker.total_cpu_seconds = worker.thread_cpu_seconds + worker.subprocess_cpu_seconds
                worker.cpu_percent = agent_cpu_percent + subprocess_cpu_percent
                worker.current_processes = live_processes
                worker.active_subprocesses = len(self._worker_subprocess_roots.get(worker.repo_slug, set()))
                if worker.deadline_minutes and worker.deadline_started_epoch:
                    elapsed = max(0.0, now - worker.deadline_started_epoch)
                    worker.deadline_remaining_seconds = max(0.0, (worker.deadline_minutes * 60.0) - elapsed)
                worker.push_history()
                worker.last_update = utc_timestamp()

            stale_pids = [pid for pid in self._process_cpu_last if pid not in sampled_process_pids]
            for pid in stale_pids:
                self._process_cpu_last.pop(pid, None)

    def _sample_subprocess_tree(
        self,
        repo_slug: str,
        now: float,
        sampled_process_pids: set[int],
    ) -> tuple[float, float, int]:
        if psutil is None:
            return 0.0, 0.0, 0

        root_pids = set(self._worker_subprocess_roots.get(repo_slug, set()))
        if not root_pids:
            return 0.0, 0.0, 0

        live_roots: set[int] = set()
        live_processes: dict[int, psutil.Process] = {}
        for pid in root_pids:
            try:
                root_process = psutil.Process(pid)
                root_process.cpu_times()
            except Exception:
                self._process_cpu_last.pop(pid, None)
                continue
            live_roots.add(pid)
            for process in [root_process, *root_process.children(recursive=True)]:
                try:
                    process.cpu_times()
                except Exception:
                    continue
                live_processes[process.pid] = process
                sampled_process_pids.add(process.pid)

        self._worker_subprocess_roots[repo_slug] = live_roots

        total_cpu_percent = 0.0
        total_cpu_seconds_delta = 0.0
        for pid, process in live_processes.items():
            try:
                cpu_times = process.cpu_times()
            except Exception:
                continue
            cpu_total = float(cpu_times.user + cpu_times.system)
            previous = self._process_cpu_last.get(pid)
            if previous is not None:
                prev_cpu, prev_ts = previous
                dt = max(now - prev_ts, 1e-6)
                total_cpu_seconds_delta += max(0.0, cpu_total - prev_cpu)
                total_cpu_percent += max(0.0, ((cpu_total - prev_cpu) / dt) * 100.0)
            self._process_cpu_last[pid] = (cpu_total, now)

        return total_cpu_percent, total_cpu_seconds_delta, len(live_processes)

    def _refresh_worker_metrics(self, worker: WorkerMonitorState) -> None:
        traj_paths: list[Path] = []
        for traj_str in worker.traj_paths:
            traj_path = Path(traj_str)
            if not traj_path.exists():
                continue
            try:
                stat = traj_path.stat()
            except OSError:
                continue
            worker.traj_mtimes[traj_str] = stat.st_mtime
            traj_paths.append(traj_path)

        metrics = collect_traj_metrics(traj_paths)
        worker.prompt_tokens = int(metrics["prompt_tokens"])
        worker.completion_tokens = int(metrics["completion_tokens"])
        worker.total_tokens = int(metrics["total_tokens"])
        worker.cost_usd = float(metrics["cost_usd"])
        worker.api_calls = int(metrics["api_calls"])
        worker.unit_test_attempts = int(metrics["unit_test_attempts"])
        worker.command_count = int(metrics["command_count"])
        worker.recent_commands = metrics["recent_commands"]

    def _collect_docker_stats(self) -> dict[str, Any]:
        snapshot = collect_docker_stats_snapshot()
        if snapshot.get("available", False):
            raw_cpu = float(snapshot.get("total_cpu_percent") or 0.0)
            self._docker_cpu_history.append(raw_cpu)
            smoothed_cpu = round(sum(self._docker_cpu_history) / len(self._docker_cpu_history), 2)
            snapshot["raw_total_cpu_percent"] = round(raw_cpu, 2)
            snapshot["smoothed_total_cpu_percent"] = smoothed_cpu
            snapshot["cpu_history"] = [round(value, 2) for value in self._docker_cpu_history]
        else:
            snapshot["raw_total_cpu_percent"] = 0.0
            snapshot["smoothed_total_cpu_percent"] = (
                round(sum(self._docker_cpu_history) / len(self._docker_cpu_history), 2)
                if self._docker_cpu_history
                else 0.0
            )
            snapshot["cpu_history"] = [round(value, 2) for value in self._docker_cpu_history]
        return snapshot

    def write_state(self) -> None:
        with self._lock:
            workers = []
            for worker in self._workers.values():
                workers.append(
                    {
                        "repo_name": worker.repo_name,
                        "repo_slug": worker.repo_slug,
                        "assigned_instances": worker.assigned_instances,
                        "status": worker.status,
                        "phase": worker.phase,
                        "current_instance": worker.current_instance,
                        "completed_instances": worker.completed_instances,
                        "resolved_instances": worker.resolved_instances,
                        "unresolved_instances": worker.unresolved_instances,
                        "error_instances": worker.error_instances,
                        "active_subprocesses": worker.active_subprocesses,
                        "command_count": worker.command_count,
                        "recent_commands": worker.recent_commands,
                        "unit_test_attempts": worker.unit_test_attempts,
                        "api_calls": worker.api_calls,
                        "prompt_tokens": worker.prompt_tokens,
                        "completion_tokens": worker.completion_tokens,
                        "total_tokens": worker.total_tokens,
                        "cost_usd": round(worker.cost_usd, 6),
                        "cpu_percent": round(worker.cpu_percent, 2),
                        "agent_cpu_percent": round(worker.agent_cpu_percent, 2),
                        "subprocess_cpu_percent": round(worker.subprocess_cpu_percent, 2),
                        "thread_cpu_seconds": round(worker.thread_cpu_seconds, 4),
                        "subprocess_cpu_seconds": round(worker.subprocess_cpu_seconds, 4),
                        "total_cpu_seconds": round(worker.total_cpu_seconds, 4),
                        "current_processes": worker.current_processes,
                        "attempt_count": worker.attempt_count,
                        "evaluation_run_count": worker.evaluation_run_count,
                        "retry_token_budget": worker.retry_token_budget,
                        "deadline_minutes": worker.deadline_minutes,
                        "deadline_remaining_seconds": round(worker.deadline_remaining_seconds, 2),
                        "started_at": worker.started_at,
                        "finished_at": worker.finished_at,
                        "last_update": worker.last_update,
                        "last_event": worker.last_event,
                        "predictions_path": worker.predictions_path,
                        "report_path": worker.report_path,
                        "accuracy": worker.accuracy,
                        "history": worker.history,
                    }
                )

            total_tokens = sum(worker.total_tokens for worker in self._workers.values())
            total_cost = sum(worker.cost_usd for worker in self._workers.values())
            total_tests = sum(worker.unit_test_attempts for worker in self._workers.values())
            total_cpu = sum(worker.cpu_percent for worker in self._workers.values())
            active_agents = sum(1 for worker in self._workers.values() if worker.status == "running")
            state = {
                "generated_at": utc_timestamp(),
                "started_at": self.started_at,
                "dataset": self.dataset,
                "split": self.split,
                "model": self.model,
                "run_id": self.run_id,
                "run_root": self.run_root,
                "repo_count": self.repo_count,
                "tasks_per_repo": self.tasks_per_repo,
                "summary": {
                    "active_agents": active_agents,
                    "total_agents": len(self._workers),
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "total_unit_test_attempts": total_tests,
                    "aggregate_cpu_percent": round(total_cpu, 2),
                },
                "workers": workers,
                "docker_stats": self._docker_stats,
                "events": list(self._events),
            }
        if self.scheduler_snapshot_provider is not None:
            try:
                state["scheduler"] = self.scheduler_snapshot_provider()
            except Exception as exc:
                error_text = f"Scheduler snapshot error: {exc.__class__.__name__}: {exc}"
                with self._lock:
                    self._events.append(
                        {
                            "timestamp": utc_timestamp(),
                            "repo_slug": "",
                            "level": "error",
                            "text": error_text,
                        }
                    )
                    state["events"] = list(self._events)
                state["scheduler"] = {
                    "summary": {
                        "error": error_text,
                    }
                }
        else:
            state["scheduler"] = None

        self._write_json_atomically(self.path, state)
        for mirror_path in self.mirror_paths:
            self._write_json_atomically(mirror_path, state)

    @staticmethod
    def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
        # mirror_paths (see write_state above) lets several DIFFERENT runs share one "latest
        # state" convenience file -- e.g. an ablation sweep's parallel replicate runs, all
        # writing their own live monitor state to the SAME base_output_root/monitor_state.json
        # mirror. A fixed, non-unique temp filename (the old `path.with_suffix(".tmp")`) meant
        # every concurrent writer raced on the exact same temp path: confirmed live, one process's
        # `.replace()` would consume/rename away the shared temp file a moment before a second
        # process's own `.replace()` call on that same path ran, crashing it with
        # `FileNotFoundError: ... monitor_state.tmp -> monitor_state.json` -- not a rare
        # occurrence, every one of 5 parallel jobs hit it within seconds. Giving each WRITE its
        # own unique temp filename (pid + random suffix) removes the collision entirely; the
        # final atomic rename target (`path`) is still shared and still fine to share -- whichever
        # writer's rename lands last simply becomes "the latest state", which is exactly the
        # intended, expected semantics for a mirror file like this.
        temp_path = path.with_suffix(f".{os.getpid()}-{random.randint(0, 999999)}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three mini-SWE-agent workers on the first three repo groups in "
            "the SWE-bench bm25_27K test split, evaluating ten tasks per repo."
        )
    )
    parser.add_argument("--model", required=True, help="Model name for mini-SWE-agent.")
    parser.add_argument(
        "--subset",
        "--dataset",
        dest="subset",
        default=DEFAULT_DATASET,
        help=(
            "SWE-bench subset alias or Hugging Face dataset name. "
            f"Default: {DEFAULT_DATASET}"
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to load. Default: {DEFAULT_SPLIT}",
    )
    parser.add_argument(
        "--repo-count",
        type=int,
        default=DEFAULT_REPO_COUNT,
        help=f"How many repos to assign. Default: {DEFAULT_REPO_COUNT}",
    )
    parser.add_argument(
        "--tasks-per-repo",
        type=int,
        default=DEFAULT_TASKS_PER_REPO,
        help=f"How many tasks per repo. Default: {DEFAULT_TASKS_PER_REPO}",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/swebench_three_agents",
        help="Root directory for logs, predictions, and evaluation outputs.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for the SWE-bench evaluator.",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        help="Optional mini-SWE-agent model class.",
    )
    parser.add_argument(
        "--agent-class",
        default=None,
        help="Optional mini-SWE-agent agent class.",
    )
    parser.add_argument(
        "--mini-config",
        action="append",
        default=[],
        help=(
            "Extra mini-SWE-agent config specs. "
            "Repeat this flag to pass multiple config specs."
        ),
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        help=(
            "Use the local Azure LiteLLM config from azure_litellm_gpt54.yaml in addition to "
            "the standard SWE-bench mini config."
        ),
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help=(
            "Route the model through OpenRouter instead of Azure: prefixes --model with "
            "'openrouter/' (e.g. --model anthropic/claude-3.5-sonnet becomes "
            "openrouter/anthropic/claude-3.5-sonnet) so litellm calls OpenRouter's API. "
            "Requires OPENROUTER_API_KEY to be set in the environment. Mutually exclusive with --azure."
        ),
    )
    parser.add_argument(
        "--environment-class",
        default=None,
        help="Optional mini-SWE-agent environment class, e.g. docker.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Run agent without confirmation, matching mini-SWE-agent yolo mode.",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Optional agent cost limit. Set 0 to disable.",
    )
    parser.add_argument(
        "--exit-immediately",
        action="store_true",
        help="Do not prompt when the agent wants to finish.",
    )
    parser.add_argument(
        "--eval-max-workers",
        type=int,
        default=2,
        help="Max workers for the SWE-bench local evaluator per repo.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=1800,
        help="Per-instance timeout passed to the SWE-bench evaluator.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip local SWE-bench evaluation and only generate predictions.",
    )
    parser.add_argument(
        "--disable-solution-sharing",
        action="store_true",
        help=(
            "Disable cross-worker attempt sharing during retries and negotiation. "
            "Workers still retain their own retry memory."
        ),
    )
    parser.add_argument(
        "--monitor-interval-seconds",
        type=float,
        default=1.0,
        help="How often to refresh the live monitor state file.",
    )
    parser.add_argument(
        "--retry-token-budget",
        type=int,
        default=200_000,
        help=(
            "Maximum cumulative agent tokens per repo assignment across retries. "
            "The runner retries unresolved instances until this budget is reached."
        ),
    )
    parser.add_argument(
        "--agent-pull-timeout-seconds",
        type=int,
        default=300,
        help=(
            "Timeout for starting/pulling the mini-SWE-agent Docker image. "
            "This is passed to the mini-SWE-agent Docker environment pull timeout."
        ),
    )
    parser.add_argument(
        "--max-zero-token-retries",
        type=int,
        default=3,
        help=(
            "Maximum consecutive retries to allow when an attempt records zero agent tokens, "
            "which usually indicates an infrastructure failure before the model actually ran."
        ),
    )
    parser.add_argument(
        "--container-task-concurrency",
        type=int,
        default=1,
        help=(
            "How many container-side tasks may run at once across all workers. "
            "This gates shell commands, unit tests, and local evaluator runs."
        ),
    )
    parser.add_argument(
        "--eval-low-priority-wait-seconds",
        type=float,
        default=120.0,
        help=(
            "How long a queued evaluation may be held back while waiting for higher-priority "
            "container work to go first."
        ),
    )
    parser.add_argument(
        "--eval-high-priority-head-start",
        type=int,
        default=2,
        help=(
            "How many high-priority container tasks must complete after an evaluation is queued "
            "before that evaluation becomes eligible to run."
        ),
    )
    parser.add_argument(
        "--high-priority-communication-wait-seconds",
        type=float,
        default=5.0,
        help=(
            "Minimum time a high-priority task stays in the communication queue before it can "
            "enter the execution task queue."
        ),
    )
    parser.add_argument(
        "--low-priority-communication-wait-seconds",
        type=float,
        default=60.0,
        help=(
            "Minimum time a low-priority task stays in the communication queue before it can "
            "enter the execution task queue."
        ),
    )
    parser.add_argument(
        "--disable-communication",
        action="store_true",
        help=(
            "Bypass the communication queue entirely. Container-side tasks go straight into the "
            "execution queue without worker negotiation."
        ),
    )
    parser.add_argument(
        "--disable-communication-history",
        action="store_true",
        help=(
            "Keep worker negotiation on, but do not provide prior inter-agent communication "
            "history to later negotiation rounds."
        ),
    )
    parser.add_argument(
        "--communication-top-k",
        type=int,
        default=5,
        help=(
            "How many communication-queue candidates participate in each active negotiation round."
        ),
    )
    parser.add_argument(
        "--communication-min-k",
        type=int,
        default=0,
        help=(
            "If greater than 0, override the communication wait timers and keep tasks in the "
            "communication queue until at least this many candidates are present in the active "
            "communication pool."
        ),
    )
    parser.add_argument(
        "--communication-bypass-cpu-threshold",
        type=float,
        default=80.0,
        help=(
            "If total Docker CPU usage is below this percent, a task may skip communication and "
            "go directly into the execution queue."
        ),
    )
    parser.add_argument(
        "--communication-bypass-cpu-observation-seconds",
        type=float,
        default=3.0,
        help=(
            "How long to observe Docker CPU usage before deciding whether a task may bypass "
            "communication. Set to 0 for a single instantaneous sample."
        ),
    )
    parser.add_argument(
        "--queue-negotiation-prompts",
        default=str(Path(__file__).with_name("queue_negotiation_prompts.yaml")),
        help=(
            "YAML file containing queue negotiation prompts, optional per-agent prompt overrides, "
            "and hyperparameters such as the discussion round count."
        ),
    )
    return parser.parse_args()


def require_evaluator(python_bin: str) -> None:
    check = subprocess.run(
        [python_bin, "-c", "import swebench.harness.run_evaluation"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "The SWE-bench evaluator is not importable from the selected Python. "
            "Install the SWE-bench package in that environment."
        )


def require_docker() -> None:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise RuntimeError(
            "Docker CLI was not found on PATH. Install Docker Desktop or make `docker` available "
            "before running SWE-bench agent/evaluation."
        )
    check = subprocess.run(
        [docker_bin, "version"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Docker is installed but not usable. Start the Docker daemon/Desktop and verify `docker version` works."
        )


def repo_slug(repo_name: str) -> str:
    return repo_name.replace("/", "__").replace("-", "_")


def split_dataset_by_repo(dataset_split) -> OrderedDict[str, list[dict]]:
    repo_to_rows: OrderedDict[str, list[dict]] = OrderedDict()
    for row in dataset_split:
        repo_to_rows.setdefault(row["repo"], []).append(row)
    return repo_to_rows


def select_assignments(
    dataset_name: str,
    split: str,
    repo_count: int,
    tasks_per_repo: int,
) -> list[RepoAssignment]:
    dataset = load_dataset(dataset_name, split=split)
    grouped = split_dataset_by_repo(dataset)
    if len(grouped) < repo_count:
        raise ValueError(f"Requested {repo_count} repos, but dataset only has {len(grouped)}.")

    assignments: list[RepoAssignment] = []
    for repo_name, rows in list(grouped.items())[:repo_count]:
        if len(rows) < tasks_per_repo:
            raise ValueError(
                f"Repo {repo_name} only has {len(rows)} tasks, fewer than requested {tasks_per_repo}."
            )
        ids = [row["instance_id"] for row in rows[:tasks_per_repo]]
        assignments.append(
            RepoAssignment(
                repo_name=repo_name,
                repo_slug=repo_slug(repo_name),
                instance_ids=ids,
            )
        )
    return assignments


def stream_subprocess(
    command: list[str],
    logger: TeeLogger,
    prefix: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
    on_finish: Callable[[subprocess.Popen, int], None] | None = None,
    deadline_at_monotonic: float | None = None,
    enforce_hard_deadline: bool = False,
    deadline_label: str = "worker",
) -> int:
    def _run() -> int:
        logger.log(f"{prefix} running: {' '.join(command)}")
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd) if cwd else None,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if enforce_hard_deadline:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        deadline_triggered = False
        output_queue: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            stdout = process.stdout
            if stdout is None:
                output_queue.put(None)
                return
            try:
                for line in stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        def _terminate_process_group() -> None:
            try:
                if enforce_hard_deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except Exception:
                pass
            deadline = time.monotonic() + 5.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                try:
                    if enforce_hard_deadline:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except Exception:
                    pass

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()
        if on_start is not None:
            on_start(process)
        saw_eof = False
        try:
            while True:
                timeout = 0.25
                if enforce_hard_deadline and deadline_at_monotonic is not None:
                    remaining = deadline_at_monotonic - time.monotonic()
                    if remaining <= 0:
                        deadline_triggered = True
                        logger.log(
                            f"{prefix} terminating because the hard deadline expired for {deadline_label}."
                        )
                        _terminate_process_group()
                        break
                    timeout = min(timeout, max(0.05, remaining))
                try:
                    line = output_queue.get(timeout=timeout)
                except queue.Empty:
                    if process.poll() is not None:
                        break
                    continue
                if line is None:
                    saw_eof = True
                    if process.poll() is not None:
                        break
                    continue
                logger.log(f"{prefix} {line.rstrip()}")

            while True:
                try:
                    line = output_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    saw_eof = True
                    continue
                logger.log(f"{prefix} {line.rstrip()}")

            if not saw_eof:
                reader_thread.join(timeout=1.0)
            returncode = process.wait()
        finally:
            if on_finish is not None:
                on_finish(process, process.poll() if process.poll() is not None else -1)
        if deadline_triggered:
            raise HardDeadlineExceededError(
                f"Hard deadline expired for {deadline_label}; running command was terminated."
            )
        return returncode

    return _run()


def run_scheduled_subprocess(
    scheduler: DockerTaskScheduler | None,
    task_context: ScheduledTaskContext,
    *,
    command: list[str],
    logger: TeeLogger,
    prefix: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
    on_finish: Callable[[subprocess.Popen, int], None] | None = None,
    enforce_hard_deadline: bool = False,
) -> int:
    if scheduler is None:
        return stream_subprocess(
            command,
            logger,
            prefix,
            cwd=cwd,
            env=env,
            on_start=on_start,
            on_finish=on_finish,
            deadline_at_monotonic=task_context.deadline_at_monotonic,
            enforce_hard_deadline=enforce_hard_deadline,
            deadline_label=task_context.repo_name,
        )

    return scheduler.run(
        lambda: stream_subprocess(
            command,
            logger,
            prefix,
            cwd=cwd,
            env=env,
            on_start=on_start,
            on_finish=on_finish,
            deadline_at_monotonic=task_context.deadline_at_monotonic,
            enforce_hard_deadline=enforce_hard_deadline,
            deadline_label=task_context.repo_name,
        ),
        context=task_context,
    )


def load_minisweagent_api():
    try:
        from minisweagent.agents import get_agent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.models import get_model
        from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_sb_environment
        from minisweagent.utils.serialize import UNSET, recursive_merge
    except ImportError as exc:
        raise RuntimeError(
            "mini-SWE-agent is not importable. Install the `minisweagent` package in this Python environment."
        ) from exc

    return {
        "DATASET_MAPPING": DATASET_MAPPING,
        "get_agent": get_agent,
        "builtin_config_dir": builtin_config_dir,
        "get_config_from_spec": get_config_from_spec,
        "get_sb_environment": get_sb_environment,
        "get_model": get_model,
        "UNSET": UNSET,
        "recursive_merge": recursive_merge,
    }


def resolve_dataset_path(subset: str, api: dict) -> str:
    return api["DATASET_MAPPING"].get(subset, subset)


def build_minisweagent_config(args: argparse.Namespace, api: dict) -> dict:
    use_azure = getattr(args, "azure", False)
    use_openrouter = getattr(args, "openrouter", False)
    if use_azure and use_openrouter:
        raise ValueError("--azure and --openrouter are mutually exclusive; pass at most one.")

    default_config = api["builtin_config_dir"] / "benchmarks" / "swebench.yaml"
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
                # mini-swe-agent's InteractiveAgent (the default agent class unless one is
                # explicitly named) prompts on stdin via a *process-wide singleton*
                # PromptSession/Application (minisweagent/agents/utils/prompt_user.py) whenever
                # the model submits, unless confirm_exit is False. In this harness every repo
                # worker runs in its own thread with no human ever watching a terminal, so that
                # prompt just blocks forever on dead stdin -- and if two threads' agents hit
                # submission around the same moment, the second one's call to the shared
                # Application crashes outright with "Application is already running." (prompt_toolkit's
                # own reentrancy guard). --yolo already means "fully unattended"; tie confirm_exit
                # to it too so unattended runs can never hit either failure mode.
                "confirm_exit": False if (args.exit_immediately or args.yolo) else api["UNSET"],
            },
            "environment": {
                "environment_class": args.environment_class or api["UNSET"],
                "pull_timeout": args.agent_pull_timeout_seconds,
                "container_timeout": getattr(args, "container_timeout", None) or api["UNSET"],
                # Solving containers otherwise get docker's default bridge network, which lets the
                # agent curl/pip-download the real upstream fix (or, separately, use `git log --all`
                # on history that isn't network-gated at all) instead of deriving one. See
                # network_access_reward_hacking_report_2026-08-27.md.
                "run_args": ["--rm", "--network", "none"],
            },
            "model": {
                "model_class": args.model_class or api["UNSET"],
                "model_name": model_name,
                # litellm's cost calculator throws when a model isn't in its pricing table
                # yet (e.g. newer OpenRouter models), and mini-SWE-agent treats that as a
                # fatal error for the whole attempt rather than just skipping cost tracking.
                # Azure's config file already sets this; apply it unconditionally so every
                # backend is protected, not just the one with a matching YAML override.
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    # Without this, a stalled/hung provider-side response blocks the solver's
                    # LLM call indefinitely -- neither our own hard-deadline check nor mini-SWE-
                    # agent's own wall_time_limit_seconds run WHILE a query is in flight, only
                    # between steps, so nothing can rescue a worker stuck here. litellm raises a
                    # clean timeout exception instead, which mini-SWE-agent's own retry() wrapper
                    # (litellm_model.py's query(), via tenacity) catches and retries, rather than
                    # hanging the whole worker thread forever. Observed directly: an astropy
                    # subrun sat completely silent for 25+ minutes with no active container
                    # process and no further scheduler activity, consistent with exactly this
                    # failure mode. Kept at 120s (not the first value tried, 300s) because
                    # mini-SWE-agent's own retry wrapper defaults to up to 10 attempts
                    # (MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT) -- 300s per attempt gave a worst case
                    # of up to ~50-60 minutes of silent retrying before failing over cleanly,
                    # which is exactly what a second live run then did. See also the
                    # MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT default set near program startup below,
                    # which bounds the *count* of attempts for the same reason.
                    "timeout": 120,
                },
            },
        }
    )
    return api["recursive_merge"](*configs)


def compute_worker_deadline_minutes(repo_slug: str) -> int:
    return 10 + (sum(repo_slug.encode("utf-8")) % 16)


def build_queue_negotiator(
    config: dict,
    logger: TeeLogger,
    *,
    communication_enabled: bool,
    prompt_config_path: str | None,
    completion_max_retries: int = 0,
    completion_retry_base_delay_seconds: float = 2.0,
    structured_parse_repair_attempts: int = 0,
    visible_coordination_task_field: str = "peer_task_id",
    visible_coordination_actions: list[str] | tuple[str, ...] | None = None,
    reasoning_summary_level: str | None = None,
    memory_summary_redactor: "Callable[[str], str] | None" = None,
    allow_bid_revision: bool = True,
    negotiation_completion_timeout_seconds: float | None = None,
    communication_history_disabled: bool = False,
) -> QueueNegotiator | None:
    if not communication_enabled:
        return None
    model_config = dict(config.get("model", {}))
    model_name = model_config.get("model_name")
    if not model_name:
        return None
    # Deliberately a COPY, then overridden below rather than mutating model_config's own dict --
    # this same model_kwargs dict (specifically its "timeout") is also what
    # build_programbench_minisweagent_config wires up for the solver/subrun-report-generation
    # model client, which is a genuinely different use case with its own, legitimately-longer
    # timeout need. negotiation_completion_timeout_seconds, when given, overrides ONLY this
    # negotiator's own copy -- see its own docstring at the CLI arg definition for why a
    # dedicated value exists at all (a negotiate() round is several sequential completion calls,
    # not one, so its own per-call budget needs to be a real fraction of the round's wall-clock
    # ceiling, not equal to it).
    model_kwargs = dict(model_config.get("model_kwargs") or {})
    if negotiation_completion_timeout_seconds is not None:
        model_kwargs["timeout"] = negotiation_completion_timeout_seconds
    return QueueNegotiator(
        model_name=model_name,
        model_kwargs=model_kwargs,
        logger=logger,
        prompt_config_path=prompt_config_path,
        completion_max_retries=completion_max_retries,
        completion_retry_base_delay_seconds=completion_retry_base_delay_seconds,
        structured_parse_repair_attempts=structured_parse_repair_attempts,
        visible_coordination_task_field=visible_coordination_task_field,
        visible_coordination_actions=visible_coordination_actions,
        reasoning_summary_level=reasoning_summary_level,
        memory_summary_redactor=memory_summary_redactor,
        allow_bid_revision=allow_bid_revision,
        communication_history_disabled=communication_history_disabled,
    )


def write_prediction(predictions_path: Path, instance_id: str, model_name: str, patch: str) -> None:
    predictions: dict[str, dict] = {}
    if predictions_path.exists():
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": patch,
    }
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def merge_predictions(source_path: Path, destination_path: Path) -> None:
    if not source_path.exists():
        return
    source = json.loads(source_path.read_text(encoding="utf-8"))
    destination: dict[str, dict] = {}
    if destination_path.exists():
        destination = json.loads(destination_path.read_text(encoding="utf-8"))
    destination.update(source)
    destination_path.write_text(json.dumps(destination, indent=2), encoding="utf-8")


_ACTIVITY_OUTPUT_TRUNCATE_CHARS = 600
_ACTIVITY_HISTORY_LIMIT = 12


def collect_traj_metrics(traj_paths: list[Path]) -> dict[str, Any]:
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_cost = 0.0
    total_api_calls = 0
    total_test_attempts = 0
    total_commands = 0
    recent_commands: list[dict[str, Any]] = []

    for traj_path in traj_paths:
        if not traj_path.exists():
            continue
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        info = data.get("info", {})
        model_stats = info.get("model_stats", {})
        total_cost += float(model_stats.get("instance_cost") or 0.0)
        total_api_calls += int(model_stats.get("api_calls") or 0)

        messages = data.get("messages", [])
        for index, message in enumerate(messages):
            extra = message.get("extra") or {}
            response = extra.get("response") or {}
            usage = response.get("usage") or {}
            total_prompt += int(usage.get("prompt_tokens") or 0)
            total_completion += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            for action in extra.get("actions") or []:
                command = str(action.get("command") or "")
                if not command:
                    continue
                total_commands += 1
                if TEST_COMMAND_PATTERN.search(command):
                    total_test_attempts += 1
                # The tool's real stdout/returncode live in the very next message
                # (role "tool"), the actual result of dispatching this command --
                # not something present on the assistant message itself.
                returncode: int | None = None
                output = ""
                if index + 1 < len(messages):
                    reply = messages[index + 1]
                    if reply.get("role") == "tool":
                        reply_extra = reply.get("extra") or {}
                        raw_returncode = reply_extra.get("returncode")
                        if raw_returncode is not None:
                            try:
                                returncode = int(raw_returncode)
                            except (TypeError, ValueError):
                                returncode = None
                        reply_content = reply.get("content")
                        if isinstance(reply_content, str):
                            match = re.search(r"<output>\n?(.*?)\n?</output>", reply_content, re.DOTALL)
                            output = match.group(1) if match else reply_content
                if len(output) > _ACTIVITY_OUTPUT_TRUNCATE_CHARS:
                    output = output[:_ACTIVITY_OUTPUT_TRUNCATE_CHARS] + "\n... (truncated)"
                recent_commands.append(
                    {
                        "command": command,
                        "returncode": returncode,
                        "output": output,
                        "timestamp": extra.get("timestamp"),
                    }
                )

    if len(recent_commands) > _ACTIVITY_HISTORY_LIMIT:
        recent_commands = recent_commands[-_ACTIVITY_HISTORY_LIMIT:]

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "cost_usd": total_cost,
        "api_calls": total_api_calls,
        "unit_test_attempts": total_test_attempts,
        "command_count": total_commands,
        "recent_commands": recent_commands,
    }


def run_instance_direct(
    instance: dict,
    config: dict,
    api: dict,
    agent_dir: Path,
    logger: TeeLogger,
    repo_name: str,
    repo_slug_value: str,
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    worker_deadline_at_monotonic: "float | DeadlineClock",
    retry_note: str | None = None,
    monitor: MonitorStateWriter | None = None,
    allow_scheduler_solution_withdrawal: bool = False,
    use_safe_task_descriptions: bool = False,
    enforce_hard_deadline: bool = False,
    wall_time_limit_seconds: float | None = None,
    announce_deadline_in_output: bool = False,
) -> None:
    instance_id = instance["instance_id"]
    task = instance["problem_statement"]
    if retry_note:
        task = f"{task}\n\n{retry_note}"
    instance_dir = agent_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    traj_path = instance_dir / f"{instance_id}.traj.json"
    if monitor is not None and repo_slug_value is not None:
        monitor.attach_traj(repo_slug_value, traj_path)
        monitor.update_worker(
            repo_slug_value,
            current_instance=instance_id,
            phase="solving",
            status="running",
        )
        monitor.add_event(repo_slug_value, f"Started solving {instance_id}.")

    logger.log(f"[{repo_name}][agent] starting {instance_id}")
    agent = None
    exit_status = None
    submission = ""
    extra_info: dict[str, str] = {}

    def _check_hard_deadline() -> None:
        if not enforce_hard_deadline:
            return
        if time.monotonic() >= float(worker_deadline_at_monotonic):
            raise HardDeadlineExceededError(
                f"Hard deadline expired for {repo_name} while working on {instance_id}."
            )

    try:
        _check_hard_deadline()
        env = api["get_sb_environment"](config, instance)
        if scheduler is not None:
            env = ScheduledEnvironment(
                env,
                scheduler,
                repo_name=repo_name,
                repo_slug=repo_slug_value,
                instance_id=instance_id,
                test_command_pattern=TEST_COMMAND_PATTERN,
                deadline_minutes=worker_deadline_minutes,
                deadline_at_monotonic=worker_deadline_at_monotonic,
                allow_solution_withdrawal=allow_scheduler_solution_withdrawal,
                use_safe_task_descriptions=use_safe_task_descriptions,
                announce_deadline_in_output=announce_deadline_in_output,
            )
        agent_config = dict(config.get("agent", {}))
        agent_config["output_path"] = traj_path
        if wall_time_limit_seconds is not None:
            # mini-swe-agent checks this before every model call (not just between instances),
            # so this is what actually makes a hard deadline interrupt mid-instance rather than
            # only at the next instance/attempt boundary.
            agent_config["wall_time_limit_seconds"] = wall_time_limit_seconds
        agent = api["get_agent"](
            api["get_model"](config=config.get("model", {})),
            env,
            agent_config,
            default_type="interactive",
        )
        info = agent.run(task)
        _check_hard_deadline()
        exit_status = info.get("exit_status")
        submission = info.get("submission") or ""
        logger.log(f"[{repo_name}][agent] finished {instance_id} with status={exit_status}")
        if monitor is not None and repo_slug_value is not None:
            monitor.add_event(repo_slug_value, f"Finished {instance_id} with status={exit_status}.")
    except SchedulerExecutionDeferred as exc:
        exit_status = "SchedulerExecutionDeferred"
        extra_info = {
            "exception_str": str(exc.reason or "execution deferred by scheduler"),
            "peer_task_id": "" if exc.peer_task_id is None else str(exc.peer_task_id),
            "peer_repo_name": str(exc.peer_repo_name or ""),
            "peer_repo_slug": str(exc.peer_repo_slug or ""),
            "shared_context": str(exc.shared_context or ""),
        }
        logger.log(
            f"[{repo_name}][agent] deferred {instance_id}: "
            f"{exc.reason or 'scheduler requested re-plan from a delivered clue'}"
        )
        if monitor is not None and repo_slug_value is not None:
            monitor.add_event(
                repo_slug_value,
                f"{instance_id} execution deferred for re-plan: {exc.reason or 'scheduler requested a re-plan.'}",
                level="warning",
            )
    except Exception as exc:
        exit_status = type(exc).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(exc)}
        logger.log(f"[{repo_name}][agent] failed {instance_id}: {exc}")
        if monitor is not None and repo_slug_value is not None:
            monitor.add_event(repo_slug_value, f"{instance_id} failed: {exc}", level="error")
    finally:
        if agent is not None:
            if hasattr(agent, "save") and not traj_path.exists():
                agent.save(
                    traj_path,
                    {
                        "info": {
                            "exit_status": exit_status,
                            "submission": submission,
                            **extra_info,
                        },
                        "instance_id": instance_id,
                    },
                )
        model_name = config.get("model", {}).get("model_name") or "unknown-model"
        if exit_status != "SchedulerExecutionDeferred":
            write_prediction(agent_dir / "preds.json", instance_id, model_name, submission)
        if monitor is not None and repo_slug_value is not None:
            monitor.update_worker(repo_slug_value, current_instance=None)


def predictions_json_to_jsonl(predictions_path: Path, output_path: Path) -> None:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    with output_path.open("w", encoding="utf-8") as fh:
        for pred in predictions.values():
            fh.write(json.dumps(pred) + "\n")


def find_evaluation_report(eval_dir: Path, run_id: str) -> Path:
    matches = sorted(eval_dir.glob(f"*.{run_id}.json"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find evaluation report matching '*.{run_id}.json' in {eval_dir}."
        )
    return matches[-1]


def run_worker(
    assignment: RepoAssignment,
    args: argparse.Namespace,
    output_root: Path,
    logger: TeeLogger,
    instances_by_id: dict[str, dict],
    config: dict,
    api: dict,
    scheduler: DockerTaskScheduler | None,
    worker_deadline_minutes: int,
    attempt_memory: SharedAttemptMemory | None = None,
    monitor: MonitorStateWriter | None = None,
    allow_scheduler_solution_withdrawal: bool = False,
) -> WorkerResult:
    worker_dir = output_root / assignment.repo_slug
    agent_dir = worker_dir / "agent_run"
    eval_dir = worker_dir / "evaluation"
    agent_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (agent_dir / "preds.json").resolve()
    total_instances = len(assignment.instance_ids)

    try:
        worker_deadline_at_monotonic = time.monotonic() + (worker_deadline_minutes * 60.0)
        logger.log(
            f"[{assignment.repo_name}] assigned instance IDs: {', '.join(assignment.instance_ids)}"
        )
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
        resolved_ids: set[str] = set()
        error_ids: set[str] = set()
        attempt_index = 0
        total_agent_tokens = 0
        final_report_path = predictions_path
        final_completed_instances = 0
        consecutive_zero_token_attempts = 0

        while pending_ids and total_agent_tokens < args.retry_token_budget:
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
            retry_note = None
            if attempt_index > 1:
                retry_note = (
                    "A previous patch attempt did not resolve this issue. "
                    "Produce a revised fix, use what you learned from the prior attempt, and validate it before exiting if possible."
                )

            for instance_id in pending_ids:
                if instance_id not in instances_by_id:
                    raise KeyError(f"Instance {instance_id} was not found in dataset {args.subset}/{args.split}.")
                note_parts: list[str] = []
                if retry_note:
                    note_parts.append(retry_note)
                if attempt_memory is not None:
                    note_parts.append(
                        attempt_memory.build_solver_context(
                            repo_slug=assignment.repo_slug,
                            instance_id=instance_id,
                            attempt_index=attempt_index,
                        )
                    )
                run_instance_direct(
                    instances_by_id[instance_id],
                    config,
                    api,
                    attempt_dir,
                    logger,
                    assignment.repo_name,
                    assignment.repo_slug,
                    scheduler,
                    worker_deadline_minutes,
                    worker_deadline_at_monotonic,
                    retry_note="\n\n".join(part for part in note_parts if part),
                    monitor=monitor,
                    allow_scheduler_solution_withdrawal=allow_scheduler_solution_withdrawal,
                )
                attempt_traj_paths.append(attempt_dir / instance_id / f"{instance_id}.traj.json")

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
                    instance_ids=list(pending_ids),
                    total_tokens=total_agent_tokens,
                )

            attempt_metrics = collect_traj_metrics(attempt_traj_paths)
            attempt_tokens = int(attempt_metrics["total_tokens"])
            total_agent_tokens += attempt_tokens
            if attempt_tokens <= 0:
                consecutive_zero_token_attempts += 1
            else:
                consecutive_zero_token_attempts = 0
            logger.log(
                f"[{assignment.repo_name}][agent] attempt {attempt_index} consumed "
                f"{attempt_tokens} tokens ({total_agent_tokens}/{args.retry_token_budget} total)."
            )

            if args.skip_eval:
                logger.log(f"[{assignment.repo_name}] skipping evaluation as requested.")
                pending_ids = []
                break

            predictions_jsonl_path = (eval_dir / "preds.jsonl").resolve()
            predictions_json_to_jsonl(predictions_path, predictions_jsonl_path)
            run_id = f"{assignment.repo_slug}_{int(time.time())}_attempt_{attempt_index:02d}"
            if monitor is not None:
                monitor.update_worker(assignment.repo_slug, evaluation_run_count=attempt_index)
            eval_cmd = [
                args.python_bin,
                "-m",
                "swebench.harness.run_evaluation",
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

            def on_eval_start(_process: subprocess.Popen) -> None:
                if monitor is None:
                    return
                monitor.register_subprocess(
                    assignment.repo_slug,
                    _process.pid,
                    "Started local SWE-bench evaluation.",
                )
                monitor.update_worker(assignment.repo_slug, phase="evaluating")

            def on_eval_finish(_process: subprocess.Popen, _returncode: int) -> None:
                if monitor is None:
                    return
                monitor.unregister_subprocess(
                    assignment.repo_slug,
                    _process.pid,
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
            )
            if returncode != 0:
                raise RuntimeError(
                    f"SWE-bench evaluation failed for {assignment.repo_name} with exit code {returncode}."
                )

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
            for instance_id in next_pending_ids:
                if instance_id in resolved_ids or instance_id in seen_pending:
                    continue
                seen_pending.add(instance_id)
                pending_ids.append(instance_id)

            if monitor is not None:
                accuracy = (len(resolved_ids) / total_instances) if total_instances else 0.0
                monitor.update_worker(
                    assignment.repo_slug,
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
                status="completed",
                phase="done",
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
                    f"Worker completed after {attempt_index} attempts with "
                    f"{len(resolved_ids)}/{total_instances} resolved."
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
            evaluation_runs=0 if args.skip_eval else attempt_index,
            total_agent_tokens=total_agent_tokens,
            agent_cpu_seconds=agent_cpu_seconds,
            subprocess_cpu_seconds=subprocess_cpu_seconds,
            total_cpu_seconds=total_cpu_seconds,
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


def print_summary(results: list[WorkerResult], logger: TeeLogger) -> None:
    total_instances = sum(result.total_instances for result in results)
    total_resolved = sum(result.resolved_instances for result in results)
    overall_accuracy = (total_resolved / total_instances) if total_instances else 0.0

    logger.log("")
    logger.log("=== Final Summary ===")
    for result in results:
        logger.log(
            (
                f"{result.repo_name}: "
                f"{result.resolved_instances}/{result.total_instances} resolved "
                f"({result.accuracy:.2%} accuracy), "
                f"eval_runs={result.evaluation_runs}, "
                f"agent_tokens={result.total_agent_tokens}, "
                f"total_cpu_seconds={result.total_cpu_seconds:.2f}, "
                f"subprocess_cpu_seconds={result.subprocess_cpu_seconds:.2f}, "
                f"completed={result.completed_instances}, "
                f"unresolved={result.unresolved_instances}, "
                f"errors={result.error_instances}"
            )
        )
        logger.log(f"{result.repo_name}: predictions -> {result.predictions_path}")
        logger.log(f"{result.repo_name}: report -> {result.report_path}")

    logger.log(
        f"Overall: {total_resolved}/{total_instances} resolved ({overall_accuracy:.2%} accuracy)"
    )


def prepare_run_layout(base_output_root: Path) -> dict[str, Path | str]:
    base_output_root.mkdir(parents=True, exist_ok=True)
    runs_root = base_output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    # Multiple processes can call this concurrently against the SAME base_output_root -- e.g. an
    # ablation sweep launching several replicate runs of the same setting in parallel, all
    # starting within the same wall-clock second. The old check-then-create
    # (`while run_root.exists(): ...` followed by a separate `mkdir(exist_ok=True)`) has a classic
    # TOCTOU race: two processes can both see run_root not-yet-existing, both compute the SAME
    # run_id, and both then mkdir(exist_ok=True) successfully -- exist_ok=True means the SECOND
    # process's mkdir is a silent no-op instead of an error, so both processes proceed to share
    # one run directory and clobber each other's run_log.txt/monitor_state.json/
    # negotiation_history_full.jsonl. mkdir(exist_ok=False) is a single atomic filesystem
    # operation (create-or-fail, no separate check step), so retrying on FileExistsError instead
    # closes that race entirely -- whichever process's mkdir call the OS actually executes first
    # wins that run_id, the other reliably sees the error and moves on to a new suffix.
    run_id = utc_compact_timestamp()
    suffix = 1
    while True:
        run_root = runs_root / run_id
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            run_id = f"{utc_compact_timestamp()}_{suffix:02d}"
    return {
        "base_output_root": base_output_root,
        "runs_root": runs_root,
        "run_id": run_id,
        "run_root": run_root,
        "run_log_path": run_root / "run_log.txt",
        "run_monitor_path": run_root / "monitor_state.json",
        "latest_log_path": base_output_root / "run_log.txt",
        "latest_monitor_path": base_output_root / "monitor_state.json",
        "manifest_path": base_output_root / "run_manifest.json",
    }


def update_run_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    run_root: Path,
    monitor_state_path: Path,
    started_at: str,
    dataset: str,
    split: str,
    model: str,
) -> None:
    # Multiple processes can share the same base_output_root (and therefore the same
    # manifest_path) concurrently -- e.g. an ablation sweep launching several replicate runs of
    # the same setting in parallel. The write itself was already atomic (temp file + os.replace,
    # below), but that only protects against a torn/partial write -- it does nothing for two
    # FULL read-modify-write cycles interleaving: both processes read the same pre-update payload,
    # both append their own entry, and whichever writes last silently discards the other's entry
    # (a classic lost-update race). Locking the whole read+modify+write makes the operation atomic
    # across processes instead of just atomic within one.
    #
    # Deliberately NOT fcntl.flock() here, despite that being the obvious first choice -- the
    # base_output_root may live on a network filesystem (e.g. Lustre), and Lustre does not
    # implement flock() at all: confirmed live, every one of an ablation sweep's first 5 parallel
    # jobs crashed on their very first manifest write with `OSError: [Errno 38] Function not
    # implemented`, 100% reproducible, not a rare race -- strictly worse than the lost-update bug
    # this was meant to fix, since it now fails every single time instead of only occasionally.
    # mkdir() is used instead: unlike flock(), atomic exclusive directory creation is a basic
    # POSIX operation that Lustre (and NFS, and every other network filesystem in practice) does
    # support, which is exactly why "lock directory" is the standard portable substitute for
    # flock() on network filesystems.
    lock_dir_path = manifest_path.with_suffix(".lockdir")
    lock_dir_path.parent.mkdir(parents=True, exist_ok=True)
    STALE_LOCK_SECONDS = 30.0  # a crashed holder must not deadlock every future writer forever
    lock_wait_started = time.monotonic()
    while True:
        try:
            lock_dir_path.mkdir()
            break
        except FileExistsError:
            try:
                age = time.time() - lock_dir_path.stat().st_mtime
            except FileNotFoundError:
                continue  # the holder released it between our mkdir attempt and this stat
            if age > STALE_LOCK_SECONDS:
                shutil.rmtree(lock_dir_path, ignore_errors=True)
                continue
            if time.monotonic() - lock_wait_started > STALE_LOCK_SECONDS * 2:
                raise TimeoutError(
                    f"update_run_manifest: could not acquire {lock_dir_path} after "
                    f"{STALE_LOCK_SECONDS * 2:.0f}s -- a stale lock should have been reclaimed "
                    "well before this point, something else is wrong."
                )
            time.sleep(0.05 + random.random() * 0.1)
    try:
        payload: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        runs = [item for item in payload.get("runs", []) if isinstance(item, dict)]
        runs = [item for item in runs if item.get("run_id") != run_id]
        runs.append(
            {
                "run_id": run_id,
                "run_root": str(run_root.resolve()),
                "monitor_state_path": str(monitor_state_path.resolve()),
                "monitor_state_web_path": f"/{monitor_state_path.as_posix()}",
                "started_at": started_at,
                "dataset": dataset,
                "split": split,
                "model": model,
            }
        )
        runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        payload["latest_run_id"] = run_id
        payload["runs"] = runs
        temp_path = manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(manifest_path)
    finally:
        shutil.rmtree(lock_dir_path, ignore_errors=True)


def main() -> int:
    args = parse_args()
    base_output_root = Path(args.output_root)
    run_layout = prepare_run_layout(base_output_root)
    output_root = run_layout["run_root"]
    log_path = run_layout["run_log_path"]
    monitor_path = run_layout["run_monitor_path"]
    logger = TeeLogger(
        log_path,
        mirror_paths=[run_layout["latest_log_path"]],
    )
    monitor: MonitorStateWriter | None = None

    try:
        logger.log("Starting three-agent SWE-bench run.")
        logger.log(f"Run ID: {run_layout['run_id']}")
        logger.log(f"Per-run output directory: {output_root}")
        api = load_minisweagent_api()
        dataset_path = resolve_dataset_path(args.subset, api)
        logger.log(f"Dataset: {dataset_path} | split: {args.split} | model: {args.model}")
        logger.log(f"Building mini-SWE-agent config from specs: {args.mini_config or ['swebench.yaml']}")
        config = build_minisweagent_config(args, api)
        attempt_memory = SharedAttemptMemory(
            output_root,
            logger=logger,
            enable_solution_sharing=not args.disable_solution_sharing,
        )
        communication_enabled = not args.disable_communication
        communication_history_enabled = not args.disable_communication_history
        negotiator = build_queue_negotiator(
            config,
            logger,
            communication_enabled=communication_enabled,
            prompt_config_path=args.queue_negotiation_prompts,
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
            negotiation_memory_provider=attempt_memory.negotiation_summary,
            logger=logger,
            negotiation_history_log_path=output_root / "negotiation_history_full.jsonl",
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
            mirror_paths=[run_layout["latest_monitor_path"]],
            run_id=str(run_layout["run_id"]),
            run_root=output_root,
        )
        update_run_manifest(
            run_layout["manifest_path"],
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
        logger.log(f"Queue negotiation prompt config: {args.queue_negotiation_prompts}")
        require_docker()
        if not args.skip_eval:
            require_evaluator(args.python_bin)

        dataset_rows = list(load_dataset(dataset_path, split=args.split))
        instances_by_id = {row["instance_id"]: row for row in dataset_rows}
        assignments = select_assignments(
            dataset_name=dataset_path,
            split=args.split,
            repo_count=args.repo_count,
            tasks_per_repo=args.tasks_per_repo,
        )

        logger.log(
            "Selected repos: "
            + ", ".join(f"{assignment.repo_name} ({len(assignment.instance_ids)} tasks)" for assignment in assignments)
        )
        worker_deadlines = {
            assignment.repo_slug: compute_worker_deadline_minutes(assignment.repo_slug)
            for assignment in assignments
        }
        for assignment in assignments:
            attempt_memory.register_assignment(assignment)
            monitor.register_worker(assignment, deadline_minutes=worker_deadlines[assignment.repo_slug])
        monitor.start()
        logger.log(f"Live monitor state will be written to {monitor_path}")
        logger.log(f"Latest monitor mirror will be written to {run_layout['latest_monitor_path']}")
        logger.log(f"Shared attempt board will be written to {attempt_memory.board_text_path}")
        logger.log(
            "Cross-worker solution sharing: "
            + ("off" if args.disable_solution_sharing else "on")
        )

        results: list[WorkerResult] = []
        with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            future_to_assignment = {
                executor.submit(
                    run_worker,
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
                    monitor,
                ): assignment
                for assignment in assignments
            }
            for future in as_completed(future_to_assignment):
                assignment = future_to_assignment[future]
                result = future.result()
                results.append(result)
                logger.log(
                    f"[{assignment.repo_name}] finished with "
                    f"{result.resolved_instances}/{result.total_instances} resolved "
                    f"({result.accuracy:.2%})"
                )
                monitor.add_event(
                    assignment.repo_slug,
                    f"Worker finished with accuracy {result.accuracy:.2%}.",
                )

        results.sort(key=lambda item: item.repo_name)
        print_summary(results, logger)
        logger.log(f"Log file written to {log_path}")
        return 0
    except Exception as exc:
        logger.log(f"Fatal error: {exc}")
        if monitor is not None:
            monitor.add_event(None, f"Fatal error: {exc}", level="error")
        return 1
    finally:
        if monitor is not None:
            monitor.stop()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
