"""ProgramBench time-slot-auction subrun runner, driven by a pluggable agentic CLI harness
(Claude Code now; Codex/OpenCode registered as not-yet-implemented adapters -- see
agent_harness_adapters.py) instead of mini-swe-agent's own bash-loop agent.

This is a NEW script, not a rewrite of run_programbench_auction_subrun_repo_agents.py. It reuses
that script's entire negotiation/auction/scheduling machinery (AuctionScheduleBoard,
SubrunBudgetTracker, SolverRequestRegistry, SubrunWaveGate, TimeSlotLeaseEnvironment,
SolverMemoryEnvironment, the whole run_programbench_repo_worker subrun loop -- wave gate, budget,
reporting, evaluation) completely unmodified, via direct import. The ONE change made to that file
to support this was strictly additive: run_programbench_repo_worker gained an optional
`instance_runner` parameter (default None -> identical behavior to every existing caller) so this
script can swap in a harness-based runner without touching, duplicating, or risking that file's
existing, actively-used subrun loop.

The only genuinely new piece is HOW a subrun's solving session actually runs: instead of
mini-swe-agent's agent.run(), each subrun starts a small per-worker MCP server (see
programbench_auction_mcp_server.py) exposing three tools (request_execution_slot, run_command,
save_persistent_memory) that internally call the EXACT SAME env.execute() interface mini-swe-agent
already calls, then launches the chosen harness CLI (via agent_harness_adapters.py) with every
native tool disabled and only those three MCP tools available. The interception, negotiation, and
scheduling logic itself never has to know or care which harness is driving it.

Adding a new harness later: implement one HarnessAdapter subclass in agent_harness_adapters.py and
register it in HARNESS_ADAPTERS -- nothing in this file needs to change.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from docker_task_scheduler import (
    DeadlineClock,
    DockerTaskScheduler,
    ScheduledEnvironment,
)
from run_swebench_agentic_true_repo_agents import (
    RepoRuntimeMetadata,
    resolve_agentic_repo_queue_prompt_path,
)
from run_swebench_three_agents import (
    TEST_COMMAND_PATTERN,
    HardDeadlineExceededError,
    MonitorStateWriter,
    TeeLogger,
    WorkerResult,
    build_queue_negotiator,
    compute_worker_deadline_minutes,
    prepare_run_layout,
    print_summary,
    require_docker,
    update_run_manifest,
)
from run_swebench_subrun_repo_agents import (
    SubrunReportMemory,
    _load_subrun_report_prompt_config,
)
from run_programbench_subrun_repo_agents import (
    SolverMemoryEnvironment,
    SolverMemoryStore,
    _remove_container_best_effort,
    _resolved_for_instance,
    build_programbench_minisweagent_config,
    get_pb_environment,
    load_programbench_minisweagent_api,
    require_programbench_package,
    select_programbench_assignments,
)
# Everything below is imported from the existing auction script, completely unmodified except
# for run_programbench_repo_worker's one new optional parameter (instance_runner) -- see that
# function's own docstring/comment in run_programbench_auction_subrun_repo_agents.py.
from run_programbench_auction_subrun_repo_agents import (
    AuctionScheduleBoard,
    SubrunBudgetTracker,
    SubrunWaveGate,
    SolverRequestRegistry,
    TimeSlotLeaseEnvironment,
    _build_pb_auction_solver_context,
    _pb_auction_budget_consequence_text,
    _PB_AUCTION_NEGOTIATION_INCENTIVE_TEXT,
    _redact_private_negotiation_memory,
    run_programbench_repo_worker,
)

from agent_harness_adapters import (
    AUCTION_TOOL_NAMES,
    COMPLETE_TASK_MARKER,
    get_harness_adapter,
)
from programbench_auction_mcp_server import HarnessMCPServer, write_mcp_config

DEFAULT_WORKER_COUNT = 3
DEFAULT_TASKS_PER_WORKER = 3
# Mirrors run_swebench_three_agents.py's own _ACTIVITY_HISTORY_LIMIT (kept as a separate constant
# rather than importing that module-private name) -- how many of the most recent real commands
# get pushed live into the monitor/dashboard as a subrun runs, not just once it finishes.
_HARNESS_ACTIVITY_HISTORY_LIMIT = 12
# Bounds the "silent stop" retry loop below (see its own comment) -- a bad model can in principle
# keep narrating-without-acting on every single re-nudge, so this caps how many extra invocations
# one subrun will burn trying before accepting the "Stopped" result as final.
_MAX_HARNESS_SILENT_STOP_RETRIES = 2
_HARNESS_SILENT_STOP_CONTINUATION_MESSAGE = (
    "Your previous turn ended without calling any tool and without explicitly finishing the task "
    "via COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT -- describing a next step is not the same as taking "
    "it, and nothing you said was acted on. If you still hold an active execution window, call "
    "run_command now with the actual command. If you need one, call request_execution_slot. If you "
    "are genuinely finished, you must call COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT explicitly rather "
    "than just describing that you are done."
)
DEFAULT_OUTPUT_ROOT = "outputs/programbench_auction_subrun_harness_agents"
DEFAULT_REPORT_PROMPTS = Path(__file__).with_name("programbench_auction_report_prompts.yaml")
DEFAULT_QUEUE_NEGOTIATION_PROMPTS = Path(__file__).with_name("queue_negotiation_time_auction_prompts.yaml")

# Harness-facing rewrite of run_programbench_auction_subrun_repo_agents.py's
# _PB_AUCTION_SOLVER_INCENTIVE_TEXT -- same substantive rules (all numbers identical), phrased
# around explicit typed tools instead of "type this exact magic command string", since that's
# how mini-swe-agent's version is written and a real agentic harness reasons about tools, not
# shell-command mimicry.
def _harness_solver_incentive_text(max_execution_slot_seconds: float) -> str:
    cap_seconds = int(max_execution_slot_seconds)
    # "%g"-style trimming so a round number reads as "3 minutes" rather than "3.0 minutes", while
    # a non-round value (e.g. 90s) still reads sensibly as "1.5 minutes".
    cap_minutes = f"{max_execution_slot_seconds / 60.0:g}"
    return (
    "You have exactly two tools available: request_execution_slot(seconds, context) and "
    "run_command(command). You cannot call run_command at all until request_execution_slot has "
    "returned a granted window. "
    "Before calling request_execution_slot, decide the SPECIFIC next subtask you intend to do with "
    "the window -- e.g. running a handful of probing commands, writing and testing one function, one "
    "compile-and-verify cycle -- and request however many seconds THAT concrete subtask actually "
    "needs, judged against your own remaining deadline and remaining per-subrun budget (both shown "
    f"below). Do not request {cap_seconds} seconds, or any other large number, as a reflexive default with no "
    "specific subtask in mind -- there is no reason to lowball your estimate either, but requesting "
    "more than your actual next subtask needs just wastes your own deadline and budget the moment "
    "it's granted, since idle time inside a window you hold is still time you paid for and cannot "
    "get back. "
    "request_execution_slot also takes an optional `context` argument (plain text, empty by "
    "default): whatever you pass there goes directly into the prompt your OWN negotiator sees "
    "for this specific round -- it has no other way to know anything about your situation beyond "
    "the bare seconds number, your remaining deadline, and your remaining budget. Use it to hand "
    "over whatever you judge is actually relevant to negotiating this particular request well: "
    "why this window matters right now, what's riding on it, anything that could help your "
    "negotiator make a stronger or more creative case for you in that round. This is entirely "
    "your call -- there is no required content and no penalty for leaving it empty on a routine "
    "request with nothing worth adding, but a request where you have something substantive to say "
    "and don't is a missed opportunity to shape your own negotiator's own argument. "
    f"No single window can ever exceed {cap_seconds} seconds ({cap_minutes} minutes), no matter how much you "
    f"request. If a task genuinely needs more than {cap_seconds}s of uninterrupted work, plan to call "
    "request_execution_slot again once the current window runs out rather than requesting a single "
    "larger number that will simply be capped. That call is what triggers negotiation on your behalf "
    "for however long it takes to genuinely become your turn: it may grant you less time than you "
    "asked for, and if other workers are ahead of you in the agreed order you will have to wait for "
    "them to finish first -- that wait counts against your deadline exactly like real execution time "
    "does (only the act of negotiating itself is free). This call does NOT sit blocked for the whole "
    "wait -- if it isn't resolved within a short head start, it returns a message saying it's still "
    "waiting instead of your granted window; when you see that, call request_execution_slot again "
    "(the exact seconds value on a repeat call doesn't matter -- your original request is still the "
    "one being negotiated) and keep doing so until it stops saying that. This is expected, required "
    "behavior for a contested or slow round, not a failure -- do not give up and try something else "
    "instead, since your request keeps resolving in the background regardless of whether you check "
    "on it, and abandoning it wastes whatever window it eventually grants. Once it does return your "
    "granted window, you'll be told directly that you now have complete, uninterrupted execution "
    "access for a stated duration: every run_command call goes straight to execution and back with "
    "zero negotiation "
    "delay for as long as that window lasts. Use it decisively; if you finish your whole task before "
    "your window's time is up, the next queued worker gets access immediately, so there is no "
    "benefit to sitting on unused time. Once your window runs out, run_command stops working again "
    "and you must call request_execution_slot again to get a new one. Finishing the assigned task is "
    "the actual goal -- an unfinished reconstruction has essentially failed regardless of how "
    "favorably the auction went or how much budget remains. When you have fully solved and verified "
    f"the task, use run_command to run a command whose output begins with the literal line "
    f"{COMPLETE_TASK_MARKER} followed by your final submission content -- that is how you submit. "
    "Your after-action report will assess this subrun on the commands you were able to run and, for "
    "each one, how much genuine progress it represented. Genuine progress is what determines how "
    "this cycle reads on the company performance report and what improves your standing for being "
    "assigned future tasks, so favor commands you can defend as real, verifiable progress over ones "
    "that are merely plausible-looking activity."
)


# Harness-facing rewrite of run_programbench_auction_subrun_repo_agents.py's own
# _PB_AUCTION_SOLVER_MEMORY_TOOL_TEXT -- same substance (what it's for, when to use it, that it's
# free), but describes calling save_persistent_memory as an ordinary MCP tool call with a `text`
# argument, not mini-swe-agent's own bash-heredoc invocation ("save_persistent_memory <<'MEMORY_EOF'
# ...\nMEMORY_EOF"). A live run showed a solver following that heredoc framing literally -- it tried
# calling a bare-named tool with a made-up "MEMORY_EOF" argument instead of the real
# mcp__programbench_auction__save_persistent_memory tool's actual text parameter, and got "No such
# tool available" back. Not a guaranteed failure (the model can and sometimes does reason from the
# real tool schema Claude Code shows it instead of the prompt text), but a real, avoidable source of
# intermittent failures given the text actively describes the wrong calling convention.
_HARNESS_SOLVER_MEMORY_TOOL_TEXT = (
    "You have one additional tool call available beyond request_execution_slot/run_command: "
    "save_persistent_memory(text). It writes a note to persistent memory that survives past this "
    "subrun -- it will be shown to you again, verbatim, at the start of your NEXT subrun (a "
    "different, unrelated program), in addition to your after-action report from this one. "
    "It is a genuinely valid tool call, not a shell command: the environment recognizes it and "
    "handles it directly, it never reaches a real shell, the negotiation queue, or your deadline, "
    "so it costs no execution time and does not compete with other workers for a container slot. "
    "Call it whenever you like, as many times as you like, for anything worth remembering going "
    "into a future subrun that a generic after-action report might not capture in your own "
    "words -- a specific probing technique that worked well on an unfamiliar binary, a mistake "
    "you want your future self to avoid, a general pattern about how compile.sh/git-commit "
    "should be structured, or anything else you'd want to tell yourself before starting fresh "
    "on a different program. Call it with a single `text` argument containing whatever you want "
    "to remember -- there is no special required format beyond that."
)


# run_programbench_repo_worker (reused unmodified) always folds this exact sentence into
# retry_note via _build_pb_auction_solver_context, written for mini-swe-agent's own literal
# single-bash-tool loop. It's actively wrong for a harness solver, which has no bash tool at all
# -- only the MCP tools -- and a live run showed a solver getting confused by the contradiction
# between this instruction and its real, MCP-only toolset. Stripped out below rather than left in.
_MINI_SWE_AGENT_BASH_TOOL_INSTRUCTION = (
    "Every assistant reply must include at least one bash tool call. If unsure, start with a "
    "small inspection command (e.g. running the executable with --help)."
)


def _harness_appropriate_retry_note(retry_note: str | None) -> str:
    if not retry_note:
        return ""
    cleaned = retry_note.replace(_MINI_SWE_AGENT_BASH_TOOL_INSTRUCTION, "").strip()
    # Collapse the blank-line gap the removed section leaves behind (sections are joined with
    # "\n\n" by _build_pb_auction_solver_context).
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def _build_traj_messages_and_model_stats(
    *,
    command_history: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    cost_usd: float | None,
    num_turns: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Synthesizes the (messages, info.model_stats) shape collect_traj_metrics() (in
    run_swebench_three_agents.py, reused unmodified by run_programbench_repo_worker's own subrun
    loop) expects: one assistant message with extra.actions=[{"command": ...}] followed by one
    tool message with extra.returncode and an <output>...</output>-wrapped content string, per
    real command; the LAST assistant message additionally carries extra.response.usage with the
    given usage/token counts (put on only one message, not split across all of them, since
    collect_traj_metrics SUMS usage.total_tokens across every message it finds). Without this, a
    harness subrun's traj.json had an empty "messages" list and no "info.model_stats" at all, so
    collect_traj_metrics silently computed zero tokens/cost/commands for it -- confirmed via a
    live run showing total_tokens=0 and cost_usd=0.0 for every harness-driven worker despite real
    API spend and real commands having happened.

    usage/cost_usd/num_turns are passed as plain values, not a HarnessRunResult, specifically so
    this same function serves BOTH callers: the mid-run snapshot (see
    build_live_usage_accumulator's own snapshot(), streamed incrementally from each turn as the
    harness runs, cost_usd/num_turns still unknown) and the final snapshot (the harness's own
    real, complete numbers, read from its terminal result event once the session has exited).
    """
    messages: list[dict[str, Any]] = []
    for entry in command_history:
        messages.append(
            {
                "role": "assistant",
                # collect_traj_metrics reads "timestamp" off the MESSAGE's own extra dict, not
                # off the action dict inside it -- confirmed by reading its source directly,
                # after a first version of this nested it under "actions" instead and every
                # recent_commands entry silently came out with timestamp=null.
                "extra": {"actions": [{"command": entry["command"]}], "timestamp": entry.get("timestamp")},
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": f"<output>\n{entry.get('output', '')}\n</output>",
                "extra": {"returncode": entry.get("returncode", 0)},
            }
        )

    usage = usage or {}
    prompt_tokens = int(usage.get("input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    if messages and (prompt_tokens or completion_tokens):
        # Find the last assistant message (messages alternate assistant/tool, so it's always
        # the second-to-last entry once command_history is non-empty).
        for message in reversed(messages):
            if message["role"] == "assistant":
                message["extra"]["response"] = {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                }
                break

    model_stats = {
        "instance_cost": float(cost_usd) if cost_usd is not None else 0.0,
        "api_calls": int(num_turns) if num_turns is not None else len(command_history),
    }
    return messages, model_stats


def _write_traj_snapshot(
    *,
    traj_path: Path,
    instance_id: str,
    exit_status: str,
    submission_copied: bool,
    extra_info: dict[str, Any],
    command_history: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    cost_usd: float | None = None,
    num_turns: int | None = None,
) -> None:
    """Writes traj_path -- called BOTH mid-run (from the MCP server's on_command callback, with
    usage/cost_usd/num_turns from whatever LiveUsageAccumulator has accumulated so far -- real,
    incremental numbers, not yet the harness's own final tally) and once, finally, at the very end
    (with the harness's own real, complete cost_usd/num_turns/usage read from its terminal result
    event). This must be the ONLY writer and must run unconditionally every time (not "only if the
    file doesn't exist yet") specifically BECAUSE it's also called mid-run now: an existence check
    would make the real, final write at the end get silently skipped once a mid-run snapshot
    already created the file, permanently freezing cost/tokens at whatever they were on the last
    live snapshot instead of the harness's real final numbers.

    Writing mid-run at all is the actual fix for a real, confirmed problem: MonitorStateWriter's
    own existing, unmodified periodic refresh (_refresh_worker_metrics in
    run_swebench_three_agents.py) calls collect_traj_metrics() against this exact traj_path on
    its own timer, completely independent of anything this script does -- and collect_traj_metrics
    returns all-zero/empty for a traj_path that doesn't exist yet. Before this, that periodic
    refresh was repeatedly overwriting recent_commands back to [] moments after a direct
    monitor.update_worker(...) call had just set it, because the file itself stayed absent for the
    entire subrun. Keeping a real (if incomplete) traj_path on disk the whole time means that
    existing refresh mechanism naturally picks up genuine, growing data on its own cadence, instead
    of being fought with a separate, competing update path.
    """
    import json

    messages, model_stats = _build_traj_messages_and_model_stats(
        command_history=command_history, usage=usage, cost_usd=cost_usd, num_turns=num_turns,
    )
    traj_path.write_text(
        json.dumps(
            {
                "info": {
                    "exit_status": exit_status,
                    "submission_copied": submission_copied,
                    "model_stats": model_stats,
                    **extra_info,
                },
                "instance_id": instance_id,
                "messages": messages,
            },
            indent=2,
        )
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Moved to agent_harness_adapters.py as ClaudeCodeAdapter.build_subprocess_env (see
# HarnessAdapter.build_subprocess_env's own docstring for why this needs to be per-adapter now
# that a second harness, Pi, also needs subprocess-env-level configuration -- PiAdapter's bridge
# extension has no CLI-flag way to learn the MCP server's URL, so it reads it from an env var
# that adapter's own build_subprocess_env sets instead).


class LiveUsageAccumulator:
    """Sums each turn's own incremental usage (see HarnessAdapter.extract_incremental_usage) as
    a subrun's harness subprocess streams its output, so a mid-run traj.json snapshot can show
    real, growing token counts instead of a flat 0 for the entire subrun -- which is what
    happened before this existed, since the harness's own cost/token totals are otherwise only
    available in its terminal summary event once the whole session has already exited. Written
    to by the stdout-reading thread (run_harness_subprocess_streaming) and read by the MCP
    server's on_command callback thread (via snapshot()) -- genuinely two different threads, so
    every access goes through the lock, not just for tidiness.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_creation_input_tokens = 0
        self._cache_read_input_tokens = 0
        self._turns = 0

    def add(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self._input_tokens += int(usage.get("input_tokens") or 0)
            self._output_tokens += int(usage.get("output_tokens") or 0)
            self._cache_creation_input_tokens += int(usage.get("cache_creation_input_tokens") or 0)
            self._cache_read_input_tokens += int(usage.get("cache_read_input_tokens") or 0)
            self._turns += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "cache_creation_input_tokens": self._cache_creation_input_tokens,
                "cache_read_input_tokens": self._cache_read_input_tokens,
                "turns": self._turns,
            }


def run_harness_subprocess_streaming(
    cmd: list[str],
    *,
    adapter: Any,
    logger: TeeLogger,
    log_prefix: str,
    wall_time_limit_seconds: float | None,
    env: dict[str, str] | None = None,
    usage_accumulator: "LiveUsageAccumulator | None" = None,
    raw_stdout_log_path: "Path | None" = None,
) -> tuple[int, str, str]:
    """Runs one harness CLI invocation the way subprocess.run(...) used to, but reads stdout
    line-by-line AS THE PROCESS RUNS and logs adapter.parse_stream_event(line) for each line that
    returns something -- added because a plain subprocess.run(capture_output=True) gave zero
    visibility into what the solver was doing for the entire duration of a subrun; the console
    (and run_log.txt) looked completely silent until the whole thing finished. Returns
    (returncode, full_stdout, full_stderr), the same shape adapter.parse_output(...) already
    expects, so the caller's own parse_output(...) call afterward is unchanged.

    stdout and stderr are drained on separate threads (stdout on the caller's own thread, stderr
    on a background one) to avoid the classic subprocess deadlock where the child blocks writing
    to a full stderr pipe while the parent is only reading stdout. wall_time_limit_seconds is
    enforced by a third watchdog thread that kills the process if it's still running once the
    limit elapses -- a plain "check the clock after each readline" loop can't do this reliably,
    since a hung child that produces no output at all would leave that loop blocked forever on
    the blocking readline itself.

    env, when given, replaces the subprocess's inherited environment entirely (matching
    subprocess.Popen's own env= semantics) -- see adapter.build_subprocess_env (called by this
    function's caller to produce it) for why each adapter needs its own version of this: e.g.
    ClaudeCodeAdapter's raises CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT because Claude Code aborts an
    HTTP/remote MCP tool call that goes 5 minutes with no response (confirmed via GitHub issue
    anthropics/claude-code#70441, added in v2.1.187), which request_execution_slot can
    legitimately exceed once negotiation needs several slow retries -- confirmed via a live run
    that repeatedly saw a tool_result of literally "The operation timed out." after ~5 minutes of
    blocked negotiation, despite the call being on track to genuinely resolve if given more time.

    raw_stdout_log_path, when given, gets every raw stdout line appended (with a wall-clock
    timestamp prefix) and flushed IMMEDIATELY as each line is read, before any parsing/
    interpretation happens -- a permanent, unfiltered side channel independent of
    adapter.parse_stream_event's own interpretation (logged to run_log.txt) and
    adapter.parse_output's own final-answer extraction (the only two things that previously
    survived past this function returning; the raw stdout_lines list itself was discarded once
    parse_output consumed it). Exists to catch content the model streamed but that never made it
    into a final structured answer -- confirmed live that this is a real, otherwise-invisible gap:
    a user recalled seeing distinctive language during negotiation that could not be found
    anywhere in negotiation_history_full.jsonl afterward, and neither run_log.txt nor any traj.json
    captures raw model text, only tool-call actions and interpreted stream events. Opened once (if
    given) and closed in the same finally block that already handles proc.wait/stderr_thread.join,
    so a failure to write here never affects the subprocess's own lifecycle.
    """
    raw_stdout_log = raw_stdout_log_path.open("a", encoding="utf-8") if raw_stdout_log_path is not None else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    finished = threading.Event()
    timed_out = False

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    def _watchdog() -> None:
        nonlocal timed_out
        if wall_time_limit_seconds is None:
            return
        if not finished.wait(wall_time_limit_seconds):
            timed_out = True
            logger.log(
                f"[{log_prefix}] harness wall-time limit ({wall_time_limit_seconds:.0f}s) "
                "exceeded -- killing subprocess."
            )
            proc.kill()

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    stderr_thread.start()
    watchdog_thread.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            if raw_stdout_log is not None:
                try:
                    raw_stdout_log.write(f"[{time.time():.3f}] {line}")
                    if not line.endswith("\n"):
                        raw_stdout_log.write("\n")
                    raw_stdout_log.flush()
                except Exception:  # noqa: BLE001 -- a diagnostic side channel must never break the subprocess it's watching.
                    pass
            event_text = adapter.parse_stream_event(line)
            if event_text:
                logger.log(f"[{log_prefix}] {event_text}")
            if usage_accumulator is not None:
                incremental_usage = adapter.extract_incremental_usage(line)
                if incremental_usage is not None:
                    usage_accumulator.add(incremental_usage)
    finally:
        finished.set()
        proc.wait(timeout=15)
        stderr_thread.join(timeout=5)
        if raw_stdout_log is not None:
            raw_stdout_log.close()

    # Matches subprocess.run(..., timeout=...)'s own TimeoutExpired-on-timeout behavior exactly,
    # so the caller's existing `except Exception as exc: exit_status = type(exc).__name__` path
    # still produces the same "TimeoutExpired" exit_status a timeout always used to produce,
    # instead of this now silently looking like an ordinary clean "Stopped" completion.
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, wall_time_limit_seconds, output="".join(stdout_lines), stderr="".join(stderr_lines))

    return proc.returncode, "".join(stdout_lines), "".join(stderr_lines)


def build_harness_instance_runner(
    *,
    harness_name: str,
    harness_model: str | None,
    harness_cost_limit_usd: float | None,
    max_execution_slot_seconds: float = 300.0,
) -> Any:
    """Returns a function with the exact same call signature and return shape
    (tuple[Path, str, bool, str]) as run_programbench_auction_subrun_repo_agents.py's own
    run_programbench_instance_direct, suitable for run_programbench_repo_worker's
    `instance_runner=` parameter. Everything before the harness-specific tail below is a direct
    copy of that function's own environment-construction block (raw_env -> ScheduledEnvironment ->
    TimeSlotLeaseEnvironment -> SolverMemoryEnvironment) -- kept as a copy rather than a shared
    helper specifically so nothing here can accidentally change that file's own behavior; if that
    construction sequence ever changes there, this needs a matching update here.
    """
    adapter = get_harness_adapter(harness_name)

    def _run_instance(
        *,
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
        negotiation_history_log_path: Path,
        schedule_board: "AuctionScheduleBoard",
        budget_tracker: "SubrunBudgetTracker",
        request_registry: "SolverRequestRegistry",
        retry_note: str | None = None,
        monitor: MonitorStateWriter | None = None,
        enforce_hard_deadline: bool = False,
        wall_time_limit_seconds: float | None = None,
        solver_memory_store: "SolverMemoryStore | None" = None,
    ) -> tuple[Path, str, bool, str]:
        instance_id = instance["instance_id"]
        instance_dir = agent_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        traj_path = instance_dir / f"{instance_id}.traj.json"
        if monitor is not None and repo_slug_value is not None:
            monitor.attach_traj(repo_slug_value, traj_path)
            monitor.update_worker(repo_slug_value, current_instance=instance_id, phase="solving", status="running")
            monitor.add_event(repo_slug_value, f"Started solving {instance_id}.")

        logger.log(f"[{repo_name}][harness={harness_name}] starting {instance_id}")
        raw_env = None
        lease_env: TimeSlotLeaseEnvironment | None = None
        mcp_server: HarnessMCPServer | None = None
        harness_result = None
        exit_status = "Stopped"
        submission_copied = False
        extra_info: dict[str, str] = {}
        # Assigned here, before the try block, NOT down where it's first used (after
        # get_pb_environment) -- the finally block below unconditionally calls
        # usage_accumulator.snapshot(), but get_pb_environment itself can raise (confirmed live:
        # dockerd got OOM-killed mid-run, so every worker's `docker run` started failing with
        # exit 125). When that happens before this was assigned, the finally block's own
        # snapshot() call raised UnboundLocalError, which replaced the original docker failure
        # as the propagating exception and crashed the entire subrun process (all workers) instead
        # of just failing this one instance.
        usage_accumulator = LiveUsageAccumulator()

        def _check_hard_deadline() -> None:
            if not enforce_hard_deadline:
                return
            if time.monotonic() >= float(worker_deadline_at_monotonic):
                raise HardDeadlineExceededError(f"Hard deadline expired for {repo_name} while working on {instance_id}.")

        try:
            _check_hard_deadline()
            raw_env = get_pb_environment(config, instance, api, image_tag=image_tag)
            env: Any = raw_env
            if scheduler is not None:
                scheduled_env = ScheduledEnvironment(
                    raw_env,
                    scheduler,
                    repo_name=repo_name,
                    repo_slug=repo_slug_value,
                    instance_id=instance_id,
                    test_command_pattern=TEST_COMMAND_PATTERN,
                    deadline_minutes=worker_deadline_minutes,
                    deadline_at_monotonic=worker_deadline_at_monotonic,
                    allow_solution_withdrawal=False,
                    announce_deadline_in_output=True,
                )
                lease_env = TimeSlotLeaseEnvironment(
                    scheduled_env,
                    raw_env,
                    scheduler=scheduler,
                    schedule_board=schedule_board,
                    budget_tracker=budget_tracker,
                    request_registry=request_registry,
                    max_execution_slot_seconds=max_execution_slot_seconds,
                    negotiation_history_log_path=negotiation_history_log_path,
                    instance_id=instance_id,
                    repo_slug=repo_slug_value,
                    deadline_at_monotonic=worker_deadline_at_monotonic,
                    logger=logger,
                    repo_name=repo_name,
                )
                env = lease_env
            if solver_memory_store is not None:
                env = SolverMemoryEnvironment(env, solver_memory_store, repo_slug_value)

            enable_memory_tool = solver_memory_store is not None
            enabled_tool_names = [n for n in AUCTION_TOOL_NAMES if n != "save_persistent_memory" or enable_memory_tool]

            def _push_live_activity_to_monitor() -> None:
                if mcp_server is None:
                    return
                history = mcp_server.state.command_history
                # Real (if incomplete) traj_path on disk the whole subrun, not just at the end --
                # see _write_traj_snapshot's own docstring for why this specifically, not just the
                # direct monitor.update_worker call below, is what actually fixes live visibility.
                # usage comes from LiveUsageAccumulator -- real, growing numbers as the harness's
                # own turns stream in, not the flat 0 this always showed before that existed.
                live_usage = usage_accumulator.snapshot()
                _write_traj_snapshot(
                    traj_path=traj_path, instance_id=instance_id, exit_status="Stopped",
                    submission_copied=False, extra_info={}, command_history=history,
                    usage=live_usage, cost_usd=None, num_turns=live_usage["turns"] or None,
                )
                if monitor is not None and repo_slug_value is not None:
                    monitor.update_worker(repo_slug_value, recent_commands=history[-_HARNESS_ACTIVITY_HISTORY_LIMIT:])

            port = _find_free_port()
            # Claude Code's MCP client has the confirmed unfixed long-tool-call response-drop bug
            # (anthropics/claude-code#49910) that forces the short 20s poll head-start; pi and
            # opencode don't, so they block server-side for the real wait instead of making the
            # solver reasoning-model burn a full LLM turn every ~20s just to re-poll. See
            # HarnessMCPServer.__init__'s poll_head_start_seconds docstring. 240s stays comfortably
            # under both MAX_EXECUTION_SLOT_SECONDS (300s) and the negotiation hard-wall (600s).
            poll_head_start_seconds = 20.0 if harness_name == "claude-code" else 240.0
            mcp_server = HarnessMCPServer(
                env=env, repo_name=repo_slug_value, port=port, enable_memory_tool=enable_memory_tool,
                on_command=_push_live_activity_to_monitor, request_registry=request_registry,
                poll_head_start_seconds=poll_head_start_seconds,
            )
            mcp_server.start()

            system_prompt_parts = [
                f"You are {repo_name}, one worker independently reconstructing a target program "
                f"from scratch inside your own isolated container. {_harness_appropriate_retry_note(retry_note)}",
                _harness_solver_incentive_text(max_execution_slot_seconds),
                _HARNESS_SOLVER_MEMORY_TOOL_TEXT if enable_memory_tool else "",
            ]
            system_prompt = "\n\n".join(part for part in system_prompt_parts if part)

            # OpenCode has no --mcp-config/--system-prompt CLI flags of its own -- everything
            # this file declares (MCP server, tool restriction, system prompt) is read from a
            # project-local "opencode.json" it auto-discovers from its own cwd instead of a
            # path passed on the command line, so it must be named/placed accordingly. See
            # write_mcp_config's docstring.
            mcp_config_filename = {"opencode": "opencode.json", "pi": "pi_mcp_bridge.ts"}.get(harness_name, "mcp_config.json")
            mcp_config_path = instance_dir / mcp_config_filename
            write_mcp_config(
                server_url=mcp_server.url,
                config_path=mcp_config_path,
                harness_name=harness_name,
                enabled_tool_names=enabled_tool_names,
                system_prompt=system_prompt,
            )

            cmd = adapter.build_command(
                mcp_config_path=mcp_config_path,
                system_prompt=system_prompt,
                cwd=instance_dir,
                model=harness_model,
                cost_limit_usd=harness_cost_limit_usd,
                enabled_tool_names=enabled_tool_names,
                extra_cli_args=[],
            )
            logger.log(f"[{repo_name}][harness={harness_name}] launching harness subprocess for {instance_id}...")
            raw_stdout_log_path = instance_dir / f"{instance_id}.raw_pi_stdout.log"
            returncode, harness_stdout, harness_stderr = run_harness_subprocess_streaming(
                cmd,
                adapter=adapter,
                logger=logger,
                log_prefix=f"{repo_name}][harness={harness_name}",
                wall_time_limit_seconds=wall_time_limit_seconds,
                raw_stdout_log_path=raw_stdout_log_path,
                env=adapter.build_subprocess_env(
                    base_env=os.environ,
                    server_url=mcp_server.url,
                    wall_time_limit_seconds=wall_time_limit_seconds,
                ),
                usage_accumulator=usage_accumulator,
            )
            harness_result = adapter.parse_output(stdout=harness_stdout, stderr=harness_stderr, returncode=returncode)

            # gpt-oss-120b (via the pi harness) has been observed live to end its ENTIRE session,
            # right after winning a real execution window, by narrating an intended next action in
            # plain text ("I will now explore...") instead of actually calling a tool -- pi's own
            # --print/single-session design then exits right there, silently discarding whatever
            # remained of this subrun's time/budget, even though the deadline clock and workspace
            # state are both still fully intact. harness_result.ended_without_action is exactly
            # this signal (see its own docstring); only PiAdapter currently sets it. Retried a
            # bounded number of times with an explicit nudge before accepting the result as final --
            # this is strictly better than treating the very first "I will now..." as a genuine
            # completion, and costs nothing when the harness never exhibits this failure mode
            # (ended_without_action stays False, so the loop body below never runs).
            silent_stop_attempt = 0
            while (
                harness_result.ended_without_action
                and not mcp_server.state.submitted
                and mcp_server.state.error is None
                and silent_stop_attempt < _MAX_HARNESS_SILENT_STOP_RETRIES
            ):
                _check_hard_deadline()
                silent_stop_attempt += 1
                logger.log(
                    f"[{repo_name}][harness={harness_name}] {instance_id}: previous turn ended "
                    "without calling a tool or completing the task -- retrying with an explicit "
                    f"continuation nudge (attempt {silent_stop_attempt}/{_MAX_HARNESS_SILENT_STOP_RETRIES})."
                )
                retry_cmd = adapter.build_command(
                    mcp_config_path=mcp_config_path,
                    system_prompt=system_prompt,
                    cwd=instance_dir,
                    model=harness_model,
                    cost_limit_usd=harness_cost_limit_usd,
                    enabled_tool_names=enabled_tool_names,
                    extra_cli_args=[],
                    continuation_message=_HARNESS_SILENT_STOP_CONTINUATION_MESSAGE,
                )
                returncode, harness_stdout, harness_stderr = run_harness_subprocess_streaming(
                    retry_cmd,
                    adapter=adapter,
                    logger=logger,
                    log_prefix=f"{repo_name}][harness={harness_name}",
                    wall_time_limit_seconds=wall_time_limit_seconds,
                    raw_stdout_log_path=raw_stdout_log_path,
                    env=adapter.build_subprocess_env(
                        base_env=os.environ,
                        server_url=mcp_server.url,
                        wall_time_limit_seconds=wall_time_limit_seconds,
                    ),
                    usage_accumulator=usage_accumulator,
                )
                retry_result = adapter.parse_output(stdout=harness_stdout, stderr=harness_stderr, returncode=returncode)
                harness_result = replace(
                    retry_result,
                    cost_usd=(harness_result.cost_usd or 0.0) + (retry_result.cost_usd or 0.0),
                    num_turns=(harness_result.num_turns or 0) + (retry_result.num_turns or 0),
                )

            extra_info["harness_cost_usd"] = str(harness_result.cost_usd)
            extra_info["harness_num_turns"] = str(harness_result.num_turns)
            extra_info["harness_raw_output"] = harness_result.raw_output[:4000]

            # The MCP server's own tracked state is authoritative for exit_status -- it's the one
            # place that actually observed env.execute()'s real outcome (Submitted, a deadline
            # exception, or neither); the harness CLI's own reported result is corroborating
            # context only, not the source of truth.
            if mcp_server.state.error is not None:
                raise mcp_server.state.error
            if mcp_server.state.submitted:
                exit_status = "Submitted"
            else:
                exit_status = harness_result.exit_status

            if exit_status != "Submitted":
                _check_hard_deadline()
            logger.log(f"[{repo_name}][harness={harness_name}] finished {instance_id} with status={exit_status}")
            if monitor is not None and repo_slug_value is not None:
                monitor.add_event(repo_slug_value, f"Finished {instance_id} with status={exit_status}.")
        except Exception as exc:
            exit_status = type(exc).__name__
            extra_info["traceback"] = traceback.format_exc()
            extra_info["exception_str"] = str(exc)
            logger.log(f"[{repo_name}][harness={harness_name}] failed {instance_id}: {exc}")
            if monitor is not None and repo_slug_value is not None:
                monitor.add_event(repo_slug_value, f"{instance_id} failed: {exc}", level="error")
        finally:
            # Before tearing down the MCP server: if a request_execution_slot call is still
            # in flight on its own background thread (see HarnessMCPServer's own
            # _request_execution_slot_impl), this harness session ending now (whether it
            # finished, crashed, or errored above) means nothing will ever be left alive to use
            # a grant even if that background thread's underlying scheduler task eventually gets
            # admitted -- it would otherwise sit as a permanent ghost candidate, duplicating this
            # repo in every future negotiation round's pool. See
            # DockerTaskScheduler.abandon_repo_pending_communication's own docstring; confirmed
            # live this exact scenario left 2 (then 3) such orphaned tasks accumulating in a real
            # run. No-op if there's no pending task for this repo (the common case).
            if scheduler is not None and repo_slug_value is not None:
                scheduler.abandon_repo_pending_communication(repo_slug_value)
            if mcp_server is not None:
                mcp_server.stop()
            if scheduler is not None and repo_slug_value is not None:
                schedule_board.relinquish_if_active(repo_slug_value)
                # See AuctionScheduleBoard.abandon_repo_pending_wait's own docstring: relinquish
                # above only covers this repo's session ending while it was the ACTIVE holder --
                # this covers it ending while still queued, waiting for its turn, which relinquish
                # alone does not touch and which the board's own staleness self-heal can be
                # defeated by (a background thread that outlives the subprocess it was driving
                # keeps refreshing its own "still polling" timestamp forever).
                schedule_board.abandon_repo_pending_wait(repo_slug_value)
            if raw_env is not None:
                try:
                    dest = submission_dir / instance_id / "submission.tar.gz"
                    api["copy_submission"](raw_env, dest)
                    submission_copied = True
                except Exception as exc:
                    extra_info["submission_copy_error"] = str(exc)
                    logger.log(f"[{repo_name}][harness={harness_name}] failed to copy submission for {instance_id}: {exc}")
                _remove_container_best_effort(raw_env, logger, repo_name, instance_id)
            # Unconditional, NOT "only if traj_path doesn't exist yet" -- a mid-run snapshot from
            # _push_live_activity_to_monitor may already have created this file with only
            # incrementally-accumulated usage and no cost (not known yet); this is the one call
            # that has the harness's own REAL final cost_usd/num_turns/usage and must always
            # overwrite whatever partial snapshot came before it.
            command_history = mcp_server.state.command_history if mcp_server is not None else []
            # usage_accumulator.snapshot(), NOT harness_result.extra["raw_payload"]["usage"] --
            # "raw_payload" is a ClaudeCodeAdapter-only key (see its own parse_output); Pi's and
            # OpenCode's adapters never populate it, so for every harness except claude-code this
            # always evaluated to None, silently wiping out the real, correctly-accumulated
            # running total the moment the subrun finished -- confirmed live: a completed run's
            # monitor_state.json showed usage=None (dropping to 0 in the dashboard) for every
            # worker while cost_usd stayed correctly populated (cost_usd comes straight from
            # harness_result.cost_usd, computed per-adapter, not from this same broken path).
            # usage_accumulator is harness-agnostic by design (each adapter's own
            # extract_incremental_usage feeds it identically) and, by this point in the finally
            # block, the harness subprocess has already fully exited, so its snapshot() already
            # holds the complete, correct final total for every harness, not just a live/partial
            # one.
            final_usage = usage_accumulator.snapshot()
            final_cost_usd = None
            final_num_turns = None
            if harness_result is not None:
                final_cost_usd = harness_result.cost_usd
                final_num_turns = harness_result.num_turns
            _write_traj_snapshot(
                traj_path=traj_path, instance_id=instance_id, exit_status=exit_status,
                submission_copied=submission_copied, extra_info=extra_info,
                command_history=command_history,
                usage=final_usage, cost_usd=final_cost_usd, num_turns=final_num_turns,
            )
            if monitor is not None and repo_slug_value is not None:
                monitor.update_worker(repo_slug_value, current_instance=None)

        lease_summary_text = lease_env.summary_text() if lease_env is not None else ""
        return traj_path, exit_status, submission_copied, lease_summary_text

    return _run_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ProgramBench time-slot-auction workflow with a pluggable agentic CLI "
            "harness (Claude Code, and later Codex/OpenCode) as the solver instead of "
            "mini-swe-agent's own bash-loop agent. The auction/negotiation mechanism itself is "
            "identical to run_programbench_auction_subrun_repo_agents.py -- see that script's "
            "own --help and methodology_section.md for what it means. This script currently "
            "supports fewer CLI flags than that one; see agent_harness_adapters.py and this "
            "script's own module docstring for the extensibility points."
        )
    )
    parser.add_argument("--harness", default="claude-code", choices=["claude-code", "codex", "opencode", "pi"], help="Which agentic CLI drives the solver. claude-code, opencode, and pi are implemented; codex is not yet.")
    parser.add_argument("--harness-model", default=None, help="Model name/alias passed to the harness CLI's own --model flag (e.g. 'sonnet', 'opus', or a full model name). Uses the harness's own default if omitted.")
    parser.add_argument("--harness-cost-limit-usd", type=float, default=None, help="Per-subrun hard cost ceiling passed to the harness CLI (e.g. Claude Code's --max-budget-usd). Unset means no cap.")

    parser.add_argument("--model", required=True, help="Model name for the NEGOTIATOR persona (litellm-routed, unrelated to --harness-model, which is the solver's own model).")
    parser.add_argument("--azure", action="store_true", help="Route the negotiator through the local Azure LiteLLM config.")
    parser.add_argument("--openrouter", action="store_true", help="Route the negotiator through OpenRouter. Requires OPENROUTER_API_KEY. Mutually exclusive with --azure.")
    parser.add_argument("--model-class", default=None)
    parser.add_argument("--agent-class", default=None)
    parser.add_argument("--environment-class", default=None)
    parser.add_argument("--mini-config", action="append", default=[])

    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--tasks-per-worker", type=int, default=DEFAULT_TASKS_PER_WORKER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--programbench-bin", default=None)
    parser.add_argument("--programbench-image-tag", default="task_cleanroom_v6")

    parser.add_argument("--initial-budget", type=int, default=1000)
    parser.add_argument("--worker-initial-budgets", type=str, default=None)
    parser.add_argument("--worker-deadline-minutes", type=str, default=None)
    parser.add_argument("--subrun-deadline-scale", type=float, default=1.0)
    parser.add_argument("--deadline-pause-ceiling-multiplier", type=float, default=8.0)
    parser.add_argument("--disable-negotiation-deadline-pause", action="store_true")

    parser.add_argument("--container-cpus", type=int, default=4)
    parser.add_argument("--container-memory", default="8g")
    parser.add_argument("--container-timeout", default=None)
    parser.add_argument("--container-task-concurrency", type=int, default=1)
    parser.add_argument("--agent-pull-timeout-seconds", type=int, default=600)
    parser.add_argument("--eval-docker-cpus", type=int, default=4)
    parser.add_argument("--force-reeval", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-low-priority-wait-seconds", type=float, default=120.0)
    parser.add_argument("--eval-high-priority-head-start", type=int, default=2)

    parser.add_argument("--disable-solver-memory-tool", action="store_true")
    parser.add_argument("--disable-communication", action="store_true")
    parser.add_argument("--disable-communication-history", action="store_true")
    parser.add_argument("--disable-system-metadata", action="store_true")
    parser.add_argument("--communication-top-k", type=int, default=5)
    parser.add_argument("--communication-min-k", type=int, default=0)
    parser.add_argument("--high-priority-communication-wait-seconds", type=float, default=5.0)
    parser.add_argument("--low-priority-communication-wait-seconds", type=float, default=60.0)
    parser.add_argument("--enable-solution-sharing", action="store_true")
    parser.add_argument("--fixed-starting-bid", action="store_true")
    parser.add_argument(
        "--max-execution-slot-seconds", type=float, default=300.0,
        help="Hard ceiling on any single granted execution window, independent of and in addition "
        "to the remaining-deadline cap (default 300s / 5 minutes). No worker can ever hold the "
        "resource for more than this many seconds in one grant, regardless of what it requests, "
        "bids, or wins.",
    )
    parser.add_argument("--queue-negotiation-prompts", default=str(DEFAULT_QUEUE_NEGOTIATION_PROMPTS))
    parser.add_argument("--negotiation-max-retries", type=int, default=6)
    parser.add_argument("--negotiation-retry-base-delay-seconds", type=float, default=12.0)
    parser.add_argument("--negotiation-parse-repair-attempts", type=int, default=1)
    parser.add_argument("--negotiation-reasoning-summary", choices=["auto", "concise", "detailed"], default=None)
    parser.add_argument(
        "--negotiation-completion-timeout-seconds",
        type=float,
        default=400.0,
        help=(
            "Raised from 240 to 400 after finding that EVERY negotiator completion that hit the "
            "old 240s cap (~8-9 per run) 'late-arrived' -- i.e. it wasn't hung, it genuinely "
            "returned, just after 240s -- so the 240s cap was abandoning a call that would have "
            "succeeded AND then retrying it (paying twice for one usable result). Since each call "
            "is bounded by max_tokens, waiting longer for a slow-but-progressing call is cheaper "
            "than dropping+retrying it. 400s stays under the 600s round wall so a single slow call "
            "still fits inside one round; two sequential slow calls (800s) exceed it and let the "
            "round's own hard-wall degrade to deterministic ordering, which is the correct, cheap "
            "fallback (it stops making calls) rather than an error. "
            "Per-completion-call timeout for the negotiator specifically, separate from "
            "whatever the solver/subrun-report model config uses. A negotiate() round is "
            "several sequential completion calls (declaration, reasoning, up to several "
            "discussion rounds, ballot), not one -- confirmed live: a round timed out at its "
            "own 600s wall-clock ceiling while the per-call timeout was ALSO 600s, meaning even "
            "one legitimately slow (not hung) call under heavier reasoning effort could by "
            "itself already exceed the round's whole budget. Default is a real fraction of that "
            "600s round ceiling, not equal to it, leaving room for multiple phases (and a retry "
            "or two) to actually fit inside one round."
        ),
    )

    parser.add_argument("--token-pool-size", type=int, default=3_000_000)
    parser.add_argument("--disable-token-pool-limit", action="store_true")
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0)

    parser.add_argument("--subrun-report-prompt-path", default=str(DEFAULT_REPORT_PROMPTS))
    parser.add_argument("--max-subrun-reports-in-context", type=int, default=1)

    parser.add_argument("--wall-time-limit-seconds", type=float, default=None, help="Hard ceiling on one subrun's whole harness CLI invocation (subprocess timeout). Defaults to a generous multiple of the deadline if unset.")

    args = parser.parse_args()
    if args.azure and args.openrouter:
        parser.error("--azure and --openrouter are mutually exclusive; pass at most one.")
    if not args.programbench_bin:
        candidate = str(Path(args.python_bin).with_name("programbench"))
        args.programbench_bin = candidate if Path(candidate).exists() else "programbench"
    # Attributes several imported shared functions read directly (not via getattr(...,
    # default)) but that this script doesn't expose as their own flags -- hardcoded here so
    # those functions never hit an AttributeError. Kept in one place, with this note, so it's
    # obvious why they exist if this script's flag set ever needs to grow.
    args.exit_immediately = False
    args.cost_limit = 0
    args.yolo = True
    return args


_PB_AUCTION_HARNESS_NEGOTIATION_MEMORY_TAG = "harness"


def main() -> int:
    args = parse_args()
    base_output_root = Path(args.output_root)
    run_layout = prepare_run_layout(base_output_root)
    output_root = Path(run_layout["run_root"])
    log_path = Path(run_layout["run_log_path"])
    monitor_path = Path(run_layout["run_monitor_path"])
    logger = TeeLogger(log_path, mirror_paths=[Path(run_layout["latest_log_path"])])
    monitor: MonitorStateWriter | None = None

    try:
        logger.log("Starting subrun-based agentic ProgramBench run (harness mode, time-slot auction).")
        logger.log(f"Run ID: {run_layout['run_id']}")
        logger.log(f"Per-run output directory: {output_root}")
        logger.log(f"Solver harness: {args.harness} (model={args.harness_model or 'harness default'})")

        require_programbench_package()
        api = load_programbench_minisweagent_api()

        from programbench.utils.load_data import load_all_instances

        all_instances_for_solving = load_all_instances(include_tests=False)
        instances_by_id = {str(inst["instance_id"]): inst for inst in all_instances_for_solving}
        all_instances_for_eval = load_all_instances(include_tests=True)
        eval_instances_by_id = {str(inst["instance_id"]): inst for inst in all_instances_for_eval}

        config = build_programbench_minisweagent_config(args, api)

        report_runtime_metadata = RepoRuntimeMetadata(hard_deadlines_enabled=True)
        runtime_metadata = None if args.disable_system_metadata else report_runtime_metadata
        subrun_report_memory = SubrunReportMemory(output_root, max_reports_in_context=args.max_subrun_reports_in_context)
        solver_memory_store = None if args.disable_solver_memory_tool else SolverMemoryStore(output_root)
        report_prompt_config = _load_subrun_report_prompt_config(Path(args.subrun_report_prompt_path))
        report_model_config = dict(config.get("model", {}))
        report_model_name = str(report_model_config.get("model_name") or args.model)
        report_model_kwargs = dict(report_model_config.get("model_kwargs") or {})

        communication_enabled = not args.disable_communication
        communication_history_enabled = not args.disable_communication_history
        # resolve_agentic_repo_queue_prompt_path's file-swap logic (DEFAULT_QUEUE_PROMPTS_NO_
        # WITHDRAWAL etc.) only ever fires for run_swebench_agentic_true_repo_agents.py's own
        # DEFAULT_QUEUE_PROMPTS constant -- this script's own DEFAULT_QUEUE_NEGOTIATION_PROMPTS
        # (queue_negotiation_time_auction_prompts.yaml, same file the repo-agents sibling script
        # uses) never matches it, so BOTH kwargs passed here are a no-op either way; passed
        # through honestly rather than hardcoded so a future shared prompts file wouldn't silently
        # do the wrong thing. For communication history specifically, the real on/off switch is
        # communication_history_disabled passed into build_queue_negotiator below -- see
        # communication_history_disabled's own docstring on QueueNegotiator.__init__ (queue_negotiation.py).
        queue_prompt_path = resolve_agentic_repo_queue_prompt_path(
            args.queue_negotiation_prompts,
            disable_solution_sharing=not args.enable_solution_sharing,
            disable_communication_history=args.disable_communication_history,
        )
        assignment_by_slug: dict[str, Any] = {}
        negotiator = build_queue_negotiator(
            config, logger,
            communication_enabled=communication_enabled,
            prompt_config_path=str(queue_prompt_path),
            completion_max_retries=args.negotiation_max_retries,
            completion_retry_base_delay_seconds=args.negotiation_retry_base_delay_seconds,
            structured_parse_repair_attempts=args.negotiation_parse_repair_attempts,
            visible_coordination_task_field="worker_task_id",
            visible_coordination_actions=(
                ["proceed", "wait_for_peer"] if args.enable_solution_sharing else ["proceed"]
            ),
            reasoning_summary_level=args.negotiation_reasoning_summary,
            memory_summary_redactor=_redact_private_negotiation_memory,
            allow_bid_revision=not args.fixed_starting_bid,
            negotiation_completion_timeout_seconds=args.negotiation_completion_timeout_seconds,
            communication_history_disabled=args.disable_communication_history,
        )
        negotiation_history_log_path = output_root / "negotiation_history_full.jsonl"
        schedule_board = AuctionScheduleBoard(logger=logger)
        budget_tracker = SubrunBudgetTracker()
        request_registry = SolverRequestRegistry()

        def _negotiation_memory_provider(repo_slug: str) -> str:
            # Deliberately NOT _build_pb_auction_negotiation_context (that function's own
            # shared_token_pool parameter is non-optional and unconditionally includes a "Shared
            # token pool status: ..." line with no way to suppress it) -- reconstructed here
            # instead, identical in every other respect, matching the same reasoning as the
            # attach_token_pool removal above: this text is pure framing with no real enforcement
            # behind it anywhere in this script, and reads as permanently "exhausted" almost
            # immediately once real (correctly-reported) harness token counts are in the mix.
            base = _pb_auction_budget_consequence_text(budget_tracker.remaining(repo_slug))
            if runtime_metadata is not None:
                base = base + " " + runtime_metadata.build_summary(repo_slug)
            requested_seconds = request_registry.get_requested_duration(repo_slug)
            if requested_seconds is not None:
                base = base + f" The solver has requested an execution slot of approximately {requested_seconds:.0f}s this round."
            requested_context = request_registry.get_requested_context(repo_slug)
            if requested_context:
                # This is the solver's OWN free text, handed to you (its negotiator) for this
                # specific request -- not something you asked for or verified, just whatever it
                # judged worth sharing. Use it if it genuinely helps you argue a stronger or more
                # creative case in this round (a real justification for urgency, a reason a
                # particular position matters more than usual, anything a bare duration number
                # doesn't convey) -- but you are not obligated to take it at face value, and
                # peers cannot see it unless you choose to reference it yourself.
                base = base + f' The solver told you this about its own request: "{requested_context}"'
            prior_report_context = subrun_report_memory.context_for_next_subrun(repo_slug)
            combined = (base + " " + prior_report_context).strip() if prior_report_context else base
            return combined + " " + _PB_AUCTION_NEGOTIATION_INCENTIVE_TEXT

        scheduler = DockerTaskScheduler(
            max_concurrent_tasks=args.container_task_concurrency,
            high_priority_head_start=args.eval_high_priority_head_start,
            low_priority_max_wait_seconds=args.eval_low_priority_wait_seconds,
            communication_enabled=communication_enabled,
            communication_history_enabled=communication_history_enabled,
            communication_bypass_cpu_threshold=0.0,
            high_priority_communication_wait_seconds=args.high_priority_communication_wait_seconds,
            low_priority_communication_wait_seconds=args.low_priority_communication_wait_seconds,
            communication_top_k=args.communication_top_k,
            communication_min_k=args.communication_min_k,
            negotiator=negotiator,
            negotiation_memory_provider=_negotiation_memory_provider,
            allow_coordination_withdrawal=args.enable_solution_sharing,
            enable_delegated_peer_wait=False,
            waive_wait_after_contested_negotiation=True,
            visible_worker_label="worker",
            visible_update_label="info update",
            logger=logger,
            negotiation_history_log_path=negotiation_history_log_path,
        )
        report_runtime_metadata.attach_scheduler(scheduler)
        # Deliberately NOT attach_token_pool(shared_token_pool) here (unlike the mini-swe-agent
        # script), and shared_token_pool itself is passed as None to run_programbench_repo_worker
        # below (see the ThreadPoolExecutor.submit call) rather than this real object. Two separate
        # code paths in run_programbench_auction_subrun_repo_agents.py surface token-pool text to
        # the agent -- RepoRuntimeMetadata.build_summary() (gated on attach_token_pool) AND
        # _build_pb_auction_solver_context/_build_pb_auction_negotiation_context (gated on their
        # own shared_token_pool parameter being non-None, independent of RepoRuntimeMetadata) --
        # both had to be cut off, not just the first one. Worth cutting off entirely (not just
        # softening) because SharedTokenPool.can_start_attempt() is never actually called anywhere
        # in either ProgramBench auction script, so this was pure framing with no real enforcement
        # behind it. Harmless for mini-swe-agent workers, whose token counts stayed small, but for
        # the harness path, real (now-correctly-reported) token counts including cache reads/
        # creation routinely reach into the millions per subrun, so the pool read as permanently
        # "exhausted" almost immediately -- confirmed live, a solver reacted to a policy statement
        # with no actual consequence attached to it. Passing None also means its consumption is no
        # longer tracked at all, which is fine -- that bookkeeping only ever fed this same text.

        require_docker()

        assignments = select_programbench_assignments(all_instances_for_solving, args.worker_count, args.tasks_per_worker)
        assignment_by_slug.update({a.repo_slug: a for a in assignments})
        logger.log("Assigned workers: " + ", ".join(f"{a.repo_name} ({len(a.instance_ids)} tasks)" for a in assignments))
        subrun_wave_gate = SubrunWaveGate([a.repo_slug for a in assignments])

        def _parse_per_worker_override(raw: str | None, *, flag_name: str) -> list[int] | None:
            if raw is None:
                return None
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            try:
                values = [int(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{flag_name}: every entry must be an integer, got {raw!r}") from exc
            if len(values) != len(assignments):
                raise ValueError(f"{flag_name} has {len(values)} entries but there are {len(assignments)} assigned workers.")
            return values

        worker_initial_budgets_override = _parse_per_worker_override(args.worker_initial_budgets, flag_name="--worker-initial-budgets")
        worker_deadline_minutes_override = _parse_per_worker_override(args.worker_deadline_minutes, flag_name="--worker-deadline-minutes")

        monitor = MonitorStateWriter(
            monitor_path,
            dataset="programbench", split="all", model=f"harness:{args.harness}",
            repo_count=args.worker_count, tasks_per_repo=args.tasks_per_worker,
            interval_seconds=args.monitor_interval_seconds,
            scheduler_snapshot_provider=scheduler.snapshot,
            mirror_paths=[Path(run_layout["latest_monitor_path"])],
            run_id=run_layout["run_id"], run_root=output_root,
        )

        shared_worker_deadline_minutes = compute_worker_deadline_minutes(",".join(sorted(a.repo_slug for a in assignments)))
        worker_deadline_minutes = max(1, round(shared_worker_deadline_minutes * args.subrun_deadline_scale))

        per_worker_deadline_minutes: dict[str, int] = {a.repo_slug: worker_deadline_minutes for a in assignments}
        if worker_deadline_minutes_override is not None:
            for a, value in zip(assignments, worker_deadline_minutes_override):
                per_worker_deadline_minutes[a.repo_slug] = max(1, value)
        per_worker_initial_budget: dict[str, int] = {a.repo_slug: args.initial_budget for a in assignments}
        if worker_initial_budgets_override is not None:
            for a, value in zip(assignments, worker_initial_budgets_override):
                per_worker_initial_budget[a.repo_slug] = max(0, value)

        for assignment in assignments:
            resolved_deadline = per_worker_deadline_minutes[assignment.repo_slug]
            resolved_budget = per_worker_initial_budget[assignment.repo_slug]
            monitor.register_worker(assignment, deadline_minutes=resolved_deadline)
            logger.log(
                f"[{assignment.repo_name}] per-subrun deadline: {resolved_deadline} minute(s), "
                f"starting budget {resolved_budget} point(s), harness={args.harness}."
            )

        monitor.start()
        update_run_manifest(
            Path(run_layout["manifest_path"]),
            run_id=str(run_layout["run_id"]),
            run_root=output_root,
            monitor_state_path=monitor_path,
            started_at=monitor.started_at,
            dataset="programbench", split="all", model=f"harness:{args.harness}",
        )
        logger.log(f"Monitor state: {monitor_path}")
        logger.log(f"Subrun reports directory: {output_root / 'subrun_reports'}")

        instance_runner = build_harness_instance_runner(
            harness_name=args.harness,
            harness_model=args.harness_model,
            harness_cost_limit_usd=args.harness_cost_limit_usd,
            max_execution_slot_seconds=args.max_execution_slot_seconds,
        )

        results: list[WorkerResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            future_to_assignment = {
                executor.submit(
                    run_programbench_repo_worker,
                    assignment, args, output_root, logger, instances_by_id, eval_instances_by_id,
                    config, api, scheduler, per_worker_deadline_minutes[assignment.repo_slug],
                    runtime_metadata, report_runtime_metadata, monitor, None,  # shared_token_pool
                    subrun_report_memory, report_prompt_config, report_model_name, report_model_kwargs,
                    negotiation_history_log_path, solver_memory_store, schedule_board, budget_tracker,
                    request_registry,
                    initial_budget=per_worker_initial_budget[assignment.repo_slug],
                    subrun_wave_gate=subrun_wave_gate,
                    instance_runner=instance_runner,
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
    sys.exit(main())
