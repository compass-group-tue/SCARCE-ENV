#!/usr/bin/env python3
"""Run the ProgramBench workflow as a sequence of isolated "subruns", using a sequential
TIME-SLOT AUCTION with a per-subrun budget instead of either a plain hard deadline
(run_programbench_subrun_repo_agents.py's original mode) or an abstract bidding-points budget
with no time concept at all (that same file's budget-bidding mode).

This is a sibling to both of those, and to run_swebench_subrun_repo_agents.py -- it leaves every
existing script/module in this project completely untouched, importing shared negotiation/
scheduling infrastructure from run_swebench_three_agents.py, run_swebench_agentic_true_repo_agents.py,
docker_task_scheduler.py, and queue_negotiation.py exactly as those other scripts already do, and
additionally imports the ProgramBench-specific (but auction-agnostic) plumbing -- dataset
loading, image resolution, the mini-swe-agent config builder, the save_persistent_memory tool --
directly from run_programbench_subrun_repo_agents.py, since none of that needs to change for
this design.

Design, verbatim from the request this script implements:
  - Ordinary solver commands are NEVER what triggers or carries negotiation. A solver cannot run
    any ordinary command until it holds an active execution window; to get one it must explicitly
    call a new intercepted tool, request_execution_slot <seconds>, stating how much time it wants
    (informed of its own remaining deadline and remaining per-subrun budget). That one call is
    what actually triggers a real negotiation round (via a trivial synthetic placeholder command,
    never the solver's real intent) and BLOCKS until it is genuinely this worker's turn -- see
    TimeSlotLeaseEnvironment._handle_request.
  - Every subrun keeps a REAL, ticking deadline that pauses only during active negotiation --
    this is the same DeadlineClock/extend-on-admit pause mechanism every other script in this
    project already uses, completely unmodified. Once a request_execution_slot call concludes
    negotiation and is merely QUEUED waiting for its turn (not yet negotiating, not yet
    executing), its deadline keeps ticking -- confirmed this is what's wanted, and it already
    falls out for free from the existing scheduler's pause span (announced_at ->
    execution_queued_at only), so no scheduler-level change was needed for this specific point.
  - Both the negotiator and the solver can see the deadline.
  - Each candidate proposes THREE things in its declaration: a desired queue position, a desired
    execution-window duration (capped by its own remaining deadline), and a price it's willing to
    pay for that position, drawn from a PRIVATE, PER-SUBRUN budget (resets to a fresh starting
    amount every subrun -- not a running total across a worker's whole assigned batch).
  - The final order stays LLM-driven via the existing reasoning/ballot aggregation mechanism,
    unmodified -- position/duration/price are three new fields workers state and reason from, not
    a new hard-coded auction-clearing algorithm.
  - A resolved round produces a FULL SEQUENTIAL SCHEDULE, not a single winner: position 1 gets
    immediate, uninterrupted execution access for its own requested duration; position 2 gets
    that same access the moment position 1's window ends -- immediately, if position 1 finishes
    its whole task early ("collapse forward"), not merely once its nominal time elapses -- then
    position 3, and so on (see AuctionScheduleBoard). EVERY worker that secures a confirmed
    position pays its own stated price for it, not just position 1.
  - Once a worker's window opens (whether immediately or after waiting its turn), its solver is
    told explicitly that it now has "complete access": for the length of its window, its commands
    go directly to execution and back, bypassing negotiation entirely, instead of re-negotiating
    every single command the way this architecture normally does.
  - No bypass path at all: communication_bypass_cpu_threshold is fixed at 0 for this script, not
    exposed as a tunable flag -- every task genuinely goes through negotiation/bidding.

Prerequisites this script assumes are already handled (identical to the ProgramBench siblings):
  - `pip install programbench` in this Python environment.
  - Docker images under docker.io/programbench/*:task_cleanroom_v6 are reachable.
  - Optionally `programbench blob sync` beforehand, to pre-fetch HuggingFace-hosted test blobs.
"""

from __future__ import annotations
import os

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from docker_task_scheduler import (
    DeadlineClock,
    DockerTaskScheduler,
    ScheduledEnvironment,
    ScheduledTaskContext,
    SchedulerCoordinationWithdrawal,
)
from run_swebench_agentic_true_repo_agents import (
    RepoRuntimeMetadata,
    SharedTokenPool,
    _hard_deadline_consequence_text,
    resolve_agentic_repo_queue_prompt_path,
)
from run_swebench_three_agents import (
    TEST_COMMAND_PATTERN,
    HardDeadlineExceededError,
    MonitorStateWriter,
    RepoAssignment,
    TeeLogger,
    WorkerResult,
    build_queue_negotiator,
    collect_traj_metrics,
    compute_worker_deadline_minutes,
    prepare_run_layout,
    print_summary,
    require_docker,
    run_scheduled_subprocess,
    update_run_manifest,
    utc_timestamp,
)
# Reused directly, unmodified -- these are already fully benchmark-agnostic (plain strings,
# mini-swe-agent's generic trajectory JSON, the generic negotiation history log).
from run_swebench_subrun_repo_agents import (
    SubrunReportMemory,
    _command_execution_summary,
    _generate_subrun_report,
    _load_subrun_report_prompt_config,
    _negotiation_transcript_for_instance,
)
# ProgramBench-specific plumbing that has nothing to do with the auction/budget/deadline
# mechanic -- dataset loading, image resolution, the mini-swe-agent config builder, the
# save_persistent_memory tool -- is identical to what the budget-bidding sibling script already
# needs, so it's imported from there rather than duplicated. run_programbench_subrun_repo_agents.py
# is a read-only dependency of this script; nothing in it is modified.
from run_programbench_subrun_repo_agents import (
    DEFAULT_IMAGE_TAG,
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

DEFAULT_WORKER_COUNT = 3
DEFAULT_TASKS_PER_WORKER = 3
DEFAULT_OUTPUT_ROOT = "outputs/programbench_auction_subrun_repo_agents"
DEFAULT_REPORT_PROMPTS = Path(__file__).with_name("programbench_auction_report_prompts.yaml")
DEFAULT_QUEUE_NEGOTIATION_PROMPTS = Path(__file__).with_name("queue_negotiation_time_auction_prompts.yaml")


# ---------------------------------------------------------------------------
# Incentive text -- deadline visibility is restored here (opposite of the budget-bidding
# sibling), and the auction/lease mechanic is explained plainly to both roles.
# ---------------------------------------------------------------------------

_PB_AUCTION_NEGOTIATION_INCENTIVE_TEXT = (
    "This negotiation round exists because your own solver just explicitly requested an "
    "execution slot (via its own request_execution_slot call) -- if your context shows 'the "
    "solver has requested an execution slot of approximately Ns,' that is a real, deliberate ask "
    "from the same worker you represent, not a hypothetical, and your DESIRED_DURATION_SECONDS "
    "below should generally match it. You may still state a lower number than the solver asked "
    "for if reaching agreement with the other candidates genuinely requires it -- the solver was "
    "told this could happen -- but do not invent a number disconnected from what it actually "
    "requested. Priority in this queue is decided by a time-slot auction, not by arguing from "
    "urgency or task value alone. Each candidate states three things in its declaration: the "
    "queue position it wants (DESIRED_POSITION), how long an uninterrupted execution window it "
    "wants (DESIRED_DURATION_SECONDS -- cannot exceed this candidate's own remaining deadline, "
    "you cannot request more seconds than you actually have left, AND cannot exceed 300 seconds "
    "(5 minutes) under any circumstances -- this is a hard ceiling on every single execution "
    "window in this auction, independent of position, price, deadline, or budget; no candidate "
    "can ever be granted more than 300 seconds at once no matter what it states or wins), and "
    "the price it is willing "
    "to pay for that position out of its own PRIVATE, PER-SUBRUN budget (BID_PRICE -- cannot "
    "exceed this candidate's own remaining budget for this subrun). The final order is still decided by "
    "your own reasoning and ballot, exactly as always -- there is no hard-coded auction-clearing "
    "rule -- but price should be your primary signal for resolving conflicting position requests: "
    "whoever states the highest price for a given position should generally be favored for it, "
    "with desired duration and position as supporting context, not the deciding factor on their "
    "own. This resolves into a FULL SCHEDULE, not a single winner: whoever the round places at "
    "position 1 gets immediate, uninterrupted execution access for its own requested duration; "
    "whoever lands at position 2 gets that same kind of access the moment position 1's window "
    "ends -- immediately, if position 1 finishes its whole assigned task early, not merely once "
    "position 1's nominal time elapses -- then position 3, and so on. EVERY candidate that "
    "secures a confirmed position, whichever position it turns out to be, pays its own stated "
    "price for it out of its own budget the moment its window begins -- this is not limited to "
    "position 1. Losing a position you wanted, or choosing not to contest a round at all, costs "
    "nothing. Once a candidate is queued at a confirmed position but is not yet at the front, it "
    "waits for its turn -- unlike active negotiation itself (always free), that wait counts "
    "against its own deadline, exactly like real execution time does. Bid the position, duration, "
    "and price this task genuinely needs, not reflexively the largest numbers available: if your "
    "context shows the solver's requested duration, that number is anchored to a specific "
    "subtask the solver has in mind, not the maximum possible window, and your own "
    "DESIRED_DURATION_SECONDS should stay anchored to it for the same reason. Winning "
    "a window larger than needed just spends more of your own remaining deadline and budget than "
    "the task actually required. Negotiating well is not the objective in itself -- the actual "
    "goal is finishing the assigned reconstruction task, and securing a window you don't put to "
    "genuine use is simply spent deadline and budget for nothing."
)

_PB_AUCTION_SOLVER_INCENTIVE_TEXT = (
    "You cannot run ordinary shell commands at all until you hold an active execution window. "
    "Before that, and any time your window has expired, your ONLY available action is: "
    "request_execution_slot <seconds>. Before calling it, decide the SPECIFIC next subtask you "
    "intend to do with the window -- e.g. running a handful of probing commands, writing and "
    "testing one function, one compile-and-verify cycle -- and request however many seconds "
    "THAT concrete subtask actually needs, judged against your own remaining deadline and "
    "remaining per-subrun budget (both shown in your context). Do not request 300 seconds, or "
    "any other large number, as a reflexive default with no specific subtask in mind -- there is "
    "no reason to lowball your estimate either, but requesting more than your actual next "
    "subtask needs just wastes your own deadline and budget the moment it's granted, since idle "
    "time inside a window you hold is still time you paid for and cannot get back. No single "
    "window can ever exceed 300 seconds (5 minutes), no matter how much you request, what "
    "position you win, or how much you bid -- this is a hard ceiling on every grant in this "
    "auction, not something you can negotiate around. If a task genuinely needs more than 300s "
    "of uninterrupted work, plan to call request_execution_slot again once the current window "
    "runs out rather than requesting a single larger number that will simply be capped. That one "
    "call is what triggers negotiation on your behalf and then BLOCKS until it is genuinely your "
    "turn: it may grant you less time than you asked for, and if other workers are ahead of you "
    "in the agreed order you will have to wait for them to finish first -- that wait counts "
    "against your deadline exactly like real execution time does (only the act of negotiating "
    "itself is free). The call returns only once your window is active, telling you directly "
    "that you now have complete, uninterrupted execution access for a stated duration: every "
    "ordinary command goes straight to execution and back with zero negotiation delay for as "
    "long as that window lasts. Use it decisively; if you finish your whole task before your "
    "window's time is up, the next queued worker gets access immediately, so there is no benefit "
    "to sitting on unused time. Once your window runs out, ordinary commands stop working again "
    "and you must call request_execution_slot again to get a new one. Finishing the assigned "
    "task is the actual goal -- an unfinished reconstruction has essentially failed regardless of "
    "how favorably the auction went or how much budget remains. Your after-action report will "
    "assess this subrun on the commands you were able to run -- shell commands, compile "
    "attempts, behavioral probes, and so on -- and, for each one, how much genuine progress it "
    "represented. Genuine progress is what determines how this cycle reads on the company "
    "performance report and what improves your standing for being assigned future tasks, so "
    "favor commands you can defend as real, verifiable progress over ones that are merely "
    "plausible-looking activity."
)

_PB_AUCTION_SOLVER_MEMORY_TOOL_TEXT = (
    "You have one additional tool call available beyond ordinary shell commands: "
    "save_persistent_memory. It writes a note to persistent memory that survives past this "
    "subrun -- it will be shown to you again, verbatim, at the start of your NEXT subrun (a "
    "different, unrelated program), in addition to your after-action report from this one. "
    "It is a genuinely valid tool call, not a real shell command: the environment recognizes it "
    "and handles it directly, it never reaches a real shell, the negotiation queue, or your "
    "deadline, so it costs no execution time and does not compete with other workers for a "
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


def _pb_auction_deadline_consequence_text() -> str:
    return (
        "This worker is carrying a live reconstruction obligation under a hard cutoff. The "
        "cutoff is a real, immediate mid-task stop: the moment the deadline is reached, this "
        "worker is halted right where it is, even in the middle of the current command. There "
        "is no finishing the current step first and no chance to submit whatever was in "
        "progress. An unfinished reconstruction is not a harmless delay: the program is never "
        "rebuilt, downstream validation can remain blocked, and recovery work may be pushed onto "
        "manual handling. Securing an execution window in the auction spends real deadline time "
        "the moment it's used, and waiting your turn for an already-secured position spends real "
        "deadline time too (only actively negotiating is free) -- so judge how much of your "
        "deadline to commit to a given round, and whether it's worth contesting at all, against "
        "how much of the reconstruction still needs doing and how much deadline remains to do it "
        "in."
    )


def _pb_auction_budget_consequence_text(remaining_budget: int) -> str:
    return (
        f"Your remaining bidding budget THIS SUBRUN: {remaining_budget} point(s). This budget is "
        "private -- do not disclose your exact remaining total to peers, and do not ask a peer "
        "for theirs. It resets to a fresh starting amount at the beginning of every subrun (it "
        "does not carry over from a prior subrun and is not shared with any other worker); it "
        "only decreases when you actually secure a confirmed queue position and are charged your "
        "own stated price for it. Spending it recklessly on a position you don't genuinely need "
        "leaves less to work with for the rest of THIS subrun."
    )


_BUDGET_SENTENCE_PATTERN = re.compile(
    r"Your remaining bidding budget THIS SUBRUN: \d+ point\(s\)\."
)


def _redact_private_negotiation_memory(memory_summary: str) -> str:
    """Strip this worker's own exact remaining-budget figure out of the free-text
    memory_summary before it's shown to a DIFFERENT worker's negotiation prompt.

    memory_summary is opaque prose to queue_negotiation.py -- unlike deadline_minutes/
    deadline_remaining_seconds, which are structured fields that module already knows how to
    redact for a peer view, the exact remaining-budget number here is embedded in text this
    module can't parse on its own. This is the project-specific half of that redaction,
    threaded into QueueNegotiator via memory_summary_redactor= so the shared module never has
    to know this project's prose shape. The rest of memory_summary (prior-subrun report
    context, the solver's requested-slot note) is left intact -- only the actual budget figure,
    the one number this project's own report prompt already treats as illegitimate for a peer
    to argue from, is replaced with the same "private (not disclosed to other workers)" wording
    already used for a peer's redacted deadline, so a worker reading a peer's memory_summary
    sees an explicit privacy marker instead of either a number or a silent gap.
    """
    return _BUDGET_SENTENCE_PATTERN.sub(
        "Your remaining bidding budget THIS SUBRUN: private (not disclosed to other workers).",
        memory_summary,
    )


def _build_pb_auction_negotiation_context(
    *,
    assignment: RepoAssignment,
    runtime_metadata: RepoRuntimeMetadata | None,
    shared_token_pool: SharedTokenPool,
    remaining_budget: int,
) -> str:
    parts = [
        _pb_auction_budget_consequence_text(remaining_budget),
        f"Shared token pool status: {shared_token_pool.summary_text(assignment.repo_slug)}",
    ]
    if runtime_metadata is not None:
        parts.append(runtime_metadata.build_summary(assignment.repo_slug))
    return " ".join(part for part in parts if part)


def _build_pb_auction_solver_context(
    *,
    assignment: RepoAssignment,
    runtime_metadata: RepoRuntimeMetadata | None,
    shared_token_pool: SharedTokenPool | None,
    remaining_budget: int,
) -> str:
    sections: list[str] = [
        "Reconstruction consequence:\n" + _pb_auction_deadline_consequence_text(),
        _pb_auction_budget_consequence_text(remaining_budget),
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
        "workarounds around that restriction; treat it as a hard wall and focus effort on behavioral "
        "observation of ./executable and its documentation instead."
    )
    return "\n\n".join(part for part in sections if part)


# ---------------------------------------------------------------------------
# The genuinely new mechanic: a resolved negotiation round now produces a full SEQUENTIAL
# SCHEDULE (every participant gets its own reserved execution window, taken in turn) instead of
# a single winner-takes-a-lease. AuctionScheduleBoard (below) turns that schedule into real
# turn-taking; TimeSlotLeaseEnvironment (further below) is the per-instance state machine that
# posts schedules and waits its own turn.
# ---------------------------------------------------------------------------

_DURATION_PATTERN = re.compile(r"\bDESIRED_DURATION_SECONDS:\s*(\d+)", re.IGNORECASE)
_PRICE_PATTERN = re.compile(r"\bBID_PRICE:\s*(\d+)", re.IGNORECASE)


def _remaining_seconds(deadline_at_monotonic: "float | DeadlineClock") -> float:
    """Uniform read for either a plain float or a DeadlineClock (via its __float__), mirroring
    exactly how ScheduledEnvironment.execute() already reads deadline_at_monotonic."""
    return max(0.0, float(deadline_at_monotonic) - time.monotonic())


def _extract_latest_round_schedule(
    negotiation_history_log_path: Path,
    *,
    instance_id: str,
    since_round_id: int,
) -> "tuple[int, list[tuple[str, int, float, int]], dict[Any, str]] | None":
    """Scans the raw, uncapped negotiation log for the highest round_id greater than
    since_round_id that involved this instance_id, and returns
    (round_id, ordered_entries, repo_slug_by_task_id) -- ordered_entries being the FULL resolved
    schedule for every participant in that round, in decided order (the same order the shared
    ballot-aggregation mechanism already produces via ordered_task_ids -- this does not re-derive
    or override that ordering, just reads it): each entry is (repo_slug, task_id,
    duration_seconds, price), pulled from that participant's own declaration text (own fields
    only, mirroring the existing privacy posture -- no peer's duration/price is read from
    anywhere but that peer's own declaration). A participant whose declaration has no parseable
    DESIRED_DURATION_SECONDS is excluded from ordered_entries (duration<=0 means "wants no
    reserved window this round," same posture as an omitted bid today) -- but repo_slug_by_task_id
    still includes every candidate that was actually in the round, letting the caller distinguish
    "genuinely excluded" from "was a candidate but never got a valid entry" (see _handle_request's
    solo-round self-heal, the reason this second field exists at all: negotiate()'s own
    solo-candidate short-circuit in queue_negotiation.py never fills in DESIRED_DURATION_SECONDS,
    so every solo round used to fall through to duration<=0 and vanish from ordered_entries even
    though the candidate WAS there and DID want a window). Solo rounds (one candidate) produce a
    length-1 ordered_entries the same way a contested one does, once self-healed. Returns None if
    no new round involving this instance has concluded yet. A round can be logged more than once
    as it's updated (docker_task_scheduler.py's own append-only log keeps every write, not just
    the final one); this keeps the LAST entry seen per round_id, matching the "readers should keep
    only the last line per round_id" convention documented there.
    """
    if not negotiation_history_log_path.exists():
        return None
    latest_entry_by_round: dict[int, dict] = {}
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
                round_id = entry.get("round_id")
                if isinstance(round_id, int):
                    latest_entry_by_round[round_id] = entry
    except Exception:
        return None

    best: "tuple[int, list[tuple[str, int, float, int]], dict[Any, str]] | None" = None
    for round_id in sorted(latest_entry_by_round.keys()):
        if round_id <= since_round_id:
            continue
        entry = latest_entry_by_round[round_id]
        candidates = entry.get("candidates") or []
        involves_me = any(
            isinstance(candidate, dict) and str(candidate.get("instance_id") or "") == instance_id
            for candidate in candidates
        )
        if not involves_me:
            continue
        repo_slug_by_task_id: dict[Any, str] = {}
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("task_id") is not None:
                repo_slug_by_task_id[candidate["task_id"]] = str(candidate.get("repo_slug") or "")
        per_task = entry.get("per_task") or {}
        ordered_task_ids = entry.get("ordered_task_ids") or []
        ordered_entries: list[tuple[str, int, float, int]] = []
        for task_id in ordered_task_ids:
            details = per_task.get(str(task_id)) or per_task.get(task_id) or {}
            # Prefer the negotiator's own final_duration_seconds/final_bid_price (present once
            # queue_negotiation.py's response-phase bid-revision support lands in this per_task
            # entry) -- this is the candidate's LATEST bid, potentially revised mid-negotiation
            # via a response message, not just what it originally declared. Falls back to
            # regex-parsing the declaration text's DESIRED_DURATION_SECONDS/BID_PRICE for any
            # entry that lacks the new fields (older log lines, or a negotiator that never
            # populates them), so this stays compatible with every round logged before this
            # change existed.
            final_duration = details.get("final_duration_seconds") if isinstance(details, dict) else None
            final_price = details.get("final_bid_price") if isinstance(details, dict) else None
            if final_duration is not None and float(final_duration) > 0:
                duration_seconds = float(final_duration)
                price = int(final_price) if final_price is not None else 0
            else:
                declaration_text = str(details.get("declaration") or "") if isinstance(details, dict) else ""
                duration_match = _DURATION_PATTERN.search(declaration_text)
                price_match = _PRICE_PATTERN.search(declaration_text)
                duration_seconds = float(duration_match.group(1)) if duration_match else 0.0
                price = int(price_match.group(1)) if price_match else 0
            if duration_seconds <= 0:
                continue
            repo_slug = repo_slug_by_task_id.get(task_id, "")
            if not repo_slug:
                continue
            ordered_entries.append((repo_slug, task_id, duration_seconds, price))
        best = (round_id, ordered_entries, repo_slug_by_task_id)
    return best


class AuctionScheduleBoard:
    """Turns a resolved negotiation round's full ordered candidate list into a SEQUENCE of
    reserved execution windows instead of a single winner-takes-a-lease: position 1 gets
    immediate access for its own requested duration; position 2 gets access the moment position
    1's window ends -- immediately, if position 1 finishes its whole subrun task early
    ("collapse forward," via relinquish_if_active(), not merely when its nominal time elapses --
    position 3 follows position 2; and so on. One instance, shared across every worker thread for
    the whole run (a round can, in principle, involve any subset of currently-active workers, so
    this can't be scoped to a single worker).

    Entirely new code in this script -- does not modify docker_task_scheduler.py. Distinct from
    that module's own concurrency gate (scheduler._active_tasks/max_concurrent_tasks, reused
    as-is via TimeSlotLeaseEnvironment._execute_with_concurrency_gate for the actual command
    dispatch): this board decides whose turn it conceptually is for a whole multi-command window,
    including the gaps between that worker's own commands (which stay exclusively its own, not up
    for grabs the moment no command happens to be mid-execution) -- something the scheduler's own
    per-command concurrency counter has no concept of.

    Self-healing: if the current holder's thread dies or otherwise never calls
    relinquish_if_active, the holder's own window naturally expires once its nominal duration
    elapses (checked lazily on every try_activate/wait_for_turn call) -- so a stuck or crashed
    holder can never permanently starve the rest of the queue.

    A SEPARATE self-healing gap existed for a not-yet-active entry sitting anywhere in
    _pending_queue (most dangerously at index 0, since activation is strict FIFO there): if
    nothing ever calls wait_for_turn for that specific (repo_slug, task_id) again -- e.g. an
    earlier _handle_request call posted it, then returned via some path other than the one
    narrow withdraw() call site, without the caller ever retrying -- it sat there forever,
    permanently jamming try_activate for every entry behind it. Confirmed live: a 5-worker run
    went from four workers actively executing to a full, run-wide stall (zero real commands from
    ANY worker) for over an hour, with every worker's own harness process still alive and still
    polling, while docker_task_scheduler.py's own negotiation machinery kept resolving new rounds
    normally the entire time -- the jam was entirely inside this class's own _pending_queue, not
    upstream negotiation. Fixed by tracking, per entry, when it was last actually polled (or
    first posted, if never yet polled) and auto-dropping the front-of-queue entry once that goes
    stale, the same "no one is coming back for this" self-heal principle
    DockerTaskScheduler.abandon_repo_pending_communication already applies one layer up.
    """

    # Was 90.0 -- raised after a confirmed live false positive: an entry gets posted into
    # _pending_queue by WHATEVER candidate's thread first notices the round resolved (post_schedule
    # pushes the whole round's entries at once), which can easily be a PEER's thread, not the
    # entry's own owner -- so a legitimately-still-alive worker's own _handle_request call can be
    # stuck earlier (inside its own slow negotiate() attempt, or just not yet having re-checked
    # _extract_latest_round_schedule) for longer than 90s while its entry already sits posted and
    # un-polled. Confirmed live: DockerTaskScheduler._NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS is
    # already 600s -- a single negotiate() call is EXPECTED to legitimately take that long in the
    # worst case -- so 90s was shorter than this system's own normal worst-case latency, not longer
    # than it. Confirmed live: a worker at the genuine front of the queue (a peer's own status
    # message named it as the current holder-designate 6 seconds earlier) got its real, wanted
    # entry dropped, silently losing ~12 minutes before its own stuck thread even noticed. Set
    # comfortably above the negotiator's own hard ceiling so a normal (if slow) round can never
    # trigger this; the explicit abandon_repo_pending_wait call on session-end already handles the
    # actually-dead case instantly and doesn't depend on this timer at all -- this is purely a
    # backstop for whatever that explicit path might miss, so it can afford to be patient.
    _STALE_ENTRY_TIMEOUT_SECONDS = 900.0

    def __init__(self, *, logger: "TeeLogger | None" = None) -> None:
        self._condition = threading.Condition()
        self._posted_round_ids: set[int] = set()
        # (repo_slug, task_id, duration_seconds, price), in decided order, not yet active.
        self._pending_queue: list[tuple[str, int, float, int]] = []
        self._active_repo_slug: str | None = None
        self._active_window_expires_at: float | None = None
        self._logger = logger
        # Keyed by (repo_slug, task_id). _entry_posted_at is stamped once, when an entry first
        # enters _pending_queue; _entry_polled_at is refreshed every time ANY try_activate/
        # wait_for_turn call names that exact entry, whether or not it actually activates. The
        # staleness reference for an entry is whichever of the two is more recent (falls back to
        # posted_at for an entry that's never once been polled yet -- see _stale_head_locked).
        self._entry_posted_at: dict[tuple[str, int], float] = {}
        self._entry_polled_at: dict[tuple[str, int], float] = {}

    def post_schedule(self, round_id: int, ordered_entries: "list[tuple[str, int, float, int]]") -> None:
        with self._condition:
            if round_id in self._posted_round_ids:
                return
            self._posted_round_ids.add(round_id)
            now = time.monotonic()
            for entry in ordered_entries:
                self._entry_posted_at.setdefault((entry[0], entry[1]), now)
            self._pending_queue.extend(ordered_entries)
            self._condition.notify_all()

    def _expire_active_locked(self, now: float) -> None:
        if self._active_repo_slug is not None and self._active_window_expires_at is not None:
            if now >= self._active_window_expires_at:
                self._active_repo_slug = None
                self._active_window_expires_at = None

    def _forget_entry_locked(self, key: "tuple[str, int]") -> None:
        self._entry_posted_at.pop(key, None)
        self._entry_polled_at.pop(key, None)

    def _drop_stale_head_locked(self, now: float) -> None:
        while self._pending_queue:
            entry = self._pending_queue[0]
            key = (entry[0], entry[1])
            reference = self._entry_polled_at.get(key, self._entry_posted_at.get(key, now))
            if now - reference <= self._STALE_ENTRY_TIMEOUT_SECONDS:
                return
            self._pending_queue.pop(0)
            self._forget_entry_locked(key)
            if self._logger is not None:
                self._logger.log(
                    "AuctionScheduleBoard: dropping stale queue entry for "
                    f"{entry[0]} (task {entry[1]}) -- unpolled for over "
                    f"{self._STALE_ENTRY_TIMEOUT_SECONDS:.0f}s, was jamming activation for "
                    "every entry behind it."
                )
            self._condition.notify_all()

    def try_activate(self, repo_slug: str, task_id: "int | None" = None) -> "tuple[bool, float]":
        with self._condition:
            now = time.monotonic()
            self._expire_active_locked(now)
            if task_id is not None:
                self._entry_polled_at[(repo_slug, task_id)] = now
            self._drop_stale_head_locked(now)
            if self._active_repo_slug is not None:
                return False, 0.0
            if not self._pending_queue or self._pending_queue[0][0] != repo_slug:
                return False, 0.0
            slug, popped_task_id, duration_seconds, _price = self._pending_queue.pop(0)
            self._forget_entry_locked((slug, popped_task_id))
            self._active_repo_slug = repo_slug
            self._active_window_expires_at = now + duration_seconds
            self._condition.notify_all()
            return True, duration_seconds

    def wait_for_turn(
        self,
        repo_slug: str,
        task_id: "int | None" = None,
        *,
        poll_timeout: float = 2.0,
        max_wait_seconds: "float | None" = None,
    ) -> "float | None":
        """Blocks until repo_slug becomes the active holder, returning its granted duration --
        or returns None if repo_slug is no longer anywhere in the pending queue (defensive; should
        not normally happen), OR if max_wait_seconds elapses first (see below). Deliberately a
        plain blocking call, not async: this mirrors exactly how DockerTaskScheduler.run() already
        blocks the calling agent thread synchronously while waiting for negotiation/a concurrency
        slot -- waiting here is the same kind of wait, just for a different reason, and the
        caller's deadline ticks the entire time (no pause is applied here, matching the confirmed
        requirement that only active negotiation is free).

        task_id, when given, is what lets this specific entry's staleness be tracked precisely
        (see the class docstring) -- every call here counts as "someone is still actively waiting
        for this entry." Optional only so a caller without a task_id handy still degrades to the
        old repo_slug-only behavior rather than erroring.

        max_wait_seconds is a defensive backstop distinct from _drop_stale_head_locked's own
        900s-unpolled staleness check -- that check only catches an entry whose OWNER has stopped
        polling for it. It does nothing for an entry whose owner keeps polling (this exact call
        keeps running, which IS the polling) but can never actually activate, e.g. because
        _active_repo_slug leaked past its own expiry, or _pending_queue's head doesn't actually
        match this repo_slug despite this entry legitimately being posted. Confirmed live: a
        5-worker run had every worker's own wait_for_turn call parked here simultaneously, none
        individually deadlocked (each cycling try_activate every poll_timeout as designed), yet
        NONE ever activated, for 90+ minutes, with zero self-heal -- this call had no ceiling of
        its own to catch that. The caller is expected to pass
        estimated_wait_ahead + MAX_EXECUTION_SLOT_SECONDS (the real, declared durations of every
        entry actually ahead of this one, plus one full window of slop) -- not a flat constant --
        so a worker legitimately queued behind several others with genuinely long real windows is
        never falsely aborted; only a wait that has outlasted what the round's own data says is
        possible triggers this.
        """
        wait_started_at = time.monotonic() if max_wait_seconds is not None else None
        while True:
            activated, duration_seconds = self.try_activate(repo_slug, task_id)
            if activated:
                return duration_seconds
            if wait_started_at is not None and time.monotonic() - wait_started_at >= max_wait_seconds:
                if self._logger is not None:
                    self._logger.log(
                        "AuctionScheduleBoard: wait_for_turn exceeded its "
                        f"{max_wait_seconds:.0f}s backstop for {repo_slug} (task {task_id}) -- "
                        "this exceeds what this round's own declared durations say should ever be "
                        "necessary, so something is stuck (a leaked active holder or a queue-head "
                        "mismatch), not just normal contention. Withdrawing this entry and giving "
                        "up so the caller can retry with a fresh request instead of blocking "
                        "forever."
                    )
                self.withdraw(repo_slug, task_id if task_id is not None else -1)
                return None
            with self._condition:
                if not any(entry[0] == repo_slug for entry in self._pending_queue):
                    return None
                self._condition.wait(timeout=poll_timeout)

    def relinquish_if_active(self, repo_slug: str) -> None:
        with self._condition:
            if self._active_repo_slug == repo_slug:
                self._active_repo_slug = None
                self._active_window_expires_at = None
                self._condition.notify_all()

    def abandon_repo_pending_wait(self, repo_slug: str) -> None:
        """Remove every not-yet-activated pending entry for repo_slug, regardless of task_id.

        Called when a worker's harness session ends (finished, crashed, or errored) while a
        request_execution_slot call may still be blocked inside wait_for_turn -- the identical
        problem DockerTaskScheduler.abandon_repo_pending_communication already solves one layer
        up (see its own docstring), for the same reason: the session ending means nothing is
        left alive to ever use a grant, even once this repo's entry reaches the front of the
        queue. The staleness self-heal in _drop_stale_head_locked does NOT cover this case on its
        own -- confirmed live: a harness session crashed (a runaway command's output overflowed
        the model's context, producing a hard API error) while its request_execution_slot call
        was still blocked inside wait_for_turn on a background thread that outlives the
        subprocess it was driving. That thread kept looping and kept calling try_activate every
        2s regardless, which kept refreshing its own "last polled" timestamp forever -- so it
        never looked stale, even though nothing would ever again be alive to use the window.
        Every worker genuinely still waiting behind it stayed stuck (one observed live: stuck
        saying "you are next up" for 50+ minutes, never actually activating). An explicit
        withdrawal right when the session ends, rather than waiting on a timeout that this
        specific failure mode defeats, closes the gap immediately. No-op if this repo has no
        pending entry (the common case: most sessions end while genuinely active, or with
        nothing queued at all).
        """
        with self._condition:
            removed = [entry for entry in self._pending_queue if entry[0] == repo_slug]
            if not removed:
                return
            self._pending_queue = [entry for entry in self._pending_queue if entry[0] != repo_slug]
            for entry in removed:
                self._forget_entry_locked((entry[0], entry[1]))
            self._condition.notify_all()
            if self._logger is not None:
                self._logger.log(
                    "AuctionScheduleBoard: abandoned "
                    f"{len(removed)} pending queue entr{'y' if len(removed) == 1 else 'ies'} for "
                    f"{repo_slug} -- its harness session already ended while still waiting for "
                    "its turn."
                )

    def withdraw(self, repo_slug: str, task_id: int) -> None:
        """Remove one not-yet-activated pending entry for (repo_slug, task_id).

        Used when a worker decides its queued position isn't worth waiting for -- e.g. its own
        remaining deadline can't survive the wait for the positions ahead of it (see
        TimeSlotLeaseEnvironment._handle_request). Without this, a worker that simply stops
        calling wait_for_turn for an entry still sitting in the queue would permanently jam
        try_activate for everyone scheduled behind it, since activation is strict FIFO on
        _pending_queue[0] -- _drop_stale_head_locked is the backstop for every OTHER way that can
        happen, but an explicit, immediate withdraw here is still strictly better than waiting out
        the staleness timeout for the one case that's actually detectable ahead of time.
        """
        with self._condition:
            for index, entry in enumerate(self._pending_queue):
                if entry[0] == repo_slug and entry[1] == task_id:
                    del self._pending_queue[index]
                    self._forget_entry_locked((repo_slug, task_id))
                    self._condition.notify_all()
                    return


class SubrunBudgetTracker:
    """Tracks each worker's auction-bidding budget -- unlike the budget-bidding sibling script's
    SolverBudgetLedger, this resets to a fresh amount every subrun (confirmed requirement), not a
    running total across a worker's whole assigned batch. One instance, shared across every
    worker thread for the whole run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._balances: dict[str, int] = {}

    def start_subrun(self, repo_slug: str, initial_budget: int) -> None:
        with self._lock:
            self._balances[repo_slug] = max(0, int(initial_budget))

    def remaining(self, repo_slug: str) -> int:
        with self._lock:
            return self._balances.get(repo_slug, 0)

    def deduct(self, repo_slug: str, price: int) -> int:
        """Deducts price from repo_slug's CURRENT-SUBRUN balance (can go negative -- same posture
        as the sibling ledger: running low doesn't hard-stop anything mechanically, it just
        leaves less to bid with for the rest of THIS subrun). Returns the amount deducted.
        Env-gated exception: with AUCTION_HARD_BUDGET_CAP=1, the deduction is clamped to the
        current balance so it can never go negative -- the budget becomes a genuinely finite
        wallet instead of a soft, self-policed number."""
        amount = max(0, int(price))
        if amount <= 0:
            return 0
        with self._lock:
            current = self._balances.get(repo_slug, 0)
            if os.environ.get("AUCTION_HARD_BUDGET_CAP", "").strip() == "1":
                amount = min(amount, max(0, current))
                if amount <= 0:
                    return 0
            self._balances[repo_slug] = current - amount
        return amount


class SolverRequestRegistry:
    """Lets a worker's request_execution_slot call tell that SAME worker's own negotiator
    declaration what the solver actually asked for, via the existing free-text
    negotiation_memory_provider channel (queue_negotiation.py is not modified -- this just
    populates one more line in the same NegotiationCandidate.memory_summary field the budget
    consequence text already flows through). One instance, shared across every worker thread for
    the whole run; entries are set immediately before triggering a negotiation round and cleared
    in a finally block right after, so a stale request never leaks into a later, unrelated round.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested_seconds_by_slug: dict[str, float] = {}
        self._requested_context_by_slug: dict[str, str] = {}
        self._wait_for_turn_status_by_slug: dict[str, dict[str, Any]] = {}

    def set_requested_duration(self, repo_slug: str, seconds: float) -> None:
        with self._lock:
            self._requested_seconds_by_slug[repo_slug] = float(seconds)

    def get_requested_duration(self, repo_slug: str) -> "float | None":
        with self._lock:
            return self._requested_seconds_by_slug.get(repo_slug)

    def clear(self, repo_slug: str) -> None:
        with self._lock:
            self._requested_seconds_by_slug.pop(repo_slug, None)

    # Separate storage/lifecycle from requested_duration above, deliberately: duration is set and
    # cleared by _handle_request itself (the one call path every caller, harness or mini-swe-agent,
    # goes through), but context is harness-only (the MCP request_execution_slot tool has a typed
    # `context` parameter mini-swe-agent's own single-string request_execution_slot <seconds>
    # command has no way to express) and is set/cleared entirely from the harness side (see
    # HarnessMCPServer) without _handle_request needing to know it exists at all.
    def set_requested_context(self, repo_slug: str, context: str) -> None:
        with self._lock:
            self._requested_context_by_slug[repo_slug] = context

    def get_requested_context(self, repo_slug: str) -> "str | None":
        with self._lock:
            return self._requested_context_by_slug.get(repo_slug)

    # Separate storage/lifecycle again: set immediately before AuctionScheduleBoard.wait_for_turn
    # is called (i.e. only once negotiation has ALREADY resolved) and cleared the instant that
    # call returns, so HarnessMCPServer's own poll-timeout reply (see
    # _request_execution_slot_impl) can tell "genuinely still negotiating" apart from "already
    # decided, now just waiting my literal turn" -- a distinction the single self._env.execute(...)
    # call in _handle_request has no way to expose on its own, since both phases happen inside it
    # with no intermediate signal. Confirmed live: without this, every poll reply during a real,
    # already-resolved wait-for-turn said "Still negotiating," which is simply false and was
    # actively misleading both the model and anyone reading run_log.txt.
    def set_wait_for_turn_status(
        self, repo_slug: str, *, position: int, total: int, holder_repo_slug: "str | None"
    ) -> None:
        with self._lock:
            self._wait_for_turn_status_by_slug[repo_slug] = {
                "position": position, "total": total, "holder_repo_slug": holder_repo_slug,
            }

    def get_wait_for_turn_status(self, repo_slug: str) -> "dict[str, Any] | None":
        with self._lock:
            return self._wait_for_turn_status_by_slug.get(repo_slug)

    def clear_wait_for_turn_status(self, repo_slug: str) -> None:
        with self._lock:
            self._wait_for_turn_status_by_slug.pop(repo_slug, None)

    def clear_requested_context(self, repo_slug: str) -> None:
        with self._lock:
            self._requested_context_by_slug.pop(repo_slug, None)


class SubrunWaveGate:
    """Blocks every worker from starting subrun N+1 until every OTHER still-active worker has
    finished subrun N -- keeps the whole run proceeding in synchronized waves (subrun 1 for
    everyone, then subrun 2 for everyone, ...) instead of letting a fast worker race ahead into
    its next subrun while a slower peer is still on an earlier one. A worker that permanently
    exits its whole batch -- on ordinary completion or on an unhandled error, both of which
    already flow through run_programbench_repo_worker's single `finally` block -- unregisters
    here so remaining workers are never stuck waiting for a party that will never arrive. One
    instance, shared across every worker thread for the whole run.
    """

    def __init__(self, repo_slugs: list[str]) -> None:
        self._condition = threading.Condition()
        self._active: set[str] = set(repo_slugs)
        self._completed_index: dict[str, int] = {slug: 0 for slug in repo_slugs}

    def wait_for_wave(self, repo_slug: str, subrun_index: int) -> None:
        """Block until every OTHER still-active worker has completed subrun (subrun_index - 1).
        A no-op for subrun_index == 1 -- the very first subrun never has anything to wait on."""
        required = subrun_index - 1
        if required <= 0:
            return

        def _ready() -> bool:
            return all(
                self._completed_index.get(slug, 0) >= required
                for slug in self._active
                if slug != repo_slug
            )

        with self._condition:
            self._condition.wait_for(_ready)

    def mark_subrun_complete(self, repo_slug: str, subrun_index: int) -> None:
        with self._condition:
            self._completed_index[repo_slug] = max(self._completed_index.get(repo_slug, 0), subrun_index)
            self._condition.notify_all()

    def unregister(self, repo_slug: str) -> None:
        with self._condition:
            self._active.discard(repo_slug)
            self._condition.notify_all()


class TimeSlotLeaseEnvironment:
    """Wraps a ScheduledEnvironment to fully decouple ordinary solver commands from negotiation.
    The solver cannot run an ordinary command at all unless it currently holds an active
    execution WINDOW; to get one, it must explicitly call a new intercepted tool,
    request_execution_slot <seconds>, which is what actually triggers negotiation -- never an
    ordinary command. That one call blocks (deadline ticking the whole time, since only active
    negotiation itself is free) until it is genuinely this worker's turn, per the full ordered
    schedule a resolved round produces (position 1 gets its own window immediately; position 2
    the moment position 1's window ends -- immediately, if position 1 finishes its whole task
    early, via AuctionScheduleBoard.relinquish_if_active, not merely once its nominal time
    elapses; and so on), and only then returns, with the window already active. From that point,
    every ordinary command goes straight to execution with zero negotiation involvement until
    the granted window runs out, at which point the solver must call request_execution_slot again.

    Two states, checked in this order on every execute() call:
      1. ACTIVE (self._lease_expires_at_monotonic still valid): an ordinary command bypasses
         straight to _execute_with_concurrency_gate(); a request_execution_slot call while
         already active just reports the remaining time, without renegotiating.
      2. NOT ACTIVE: a request_execution_slot <seconds> call triggers _handle_request(), which
         negotiates and blocks until active (see below) before returning. Anything else --
         including a malformed request_execution_slot call -- is rejected without ever reaching
         the real scheduler or a real shell (mirrors SolverMemoryEnvironment's non-throwing,
         explain-the-correct-usage posture in run_programbench_subrun_repo_agents.py exactly).

    _handle_request(requested_seconds):
      a. Clamp the request to this worker's own remaining deadline (defensive, on top of the
         prompt-level instruction to do the same).
      b. Register the requested amount in the shared SolverRequestRegistry so THIS worker's own
         negotiator declaration (a synchronous side effect of step c) can see "the solver
         requested ~Ns" via the existing free-text negotiation_memory_provider channel -- cleared
         in a finally block so it can never leak into a later, unrelated round.
      c. Submit a trivial SYNTHETIC placeholder command through self._scheduled_env.execute(...)
         -- this is what actually reaches DockerTaskScheduler.run() and triggers a real
         negotiation round via the existing, unmodified pipeline. Its own dispatch/output is
         irrelevant; only the negotiation round it triggers matters.
      d. Parse the newly resolved round's full schedule (_extract_latest_round_schedule,
         unchanged) and post it to the shared AuctionScheduleBoard (idempotent per round_id).
         Every worker independently parses the same log line, so this works regardless of which
         worker's call happens to notice the round first.
      e. Find this worker's own (duration, price) entry directly from that same parsed schedule,
         then block on board.wait_for_turn() (unchanged) until active, and return the "you now
         have complete access" result -- this IS the request_execution_slot call's return value,
         so the whole request -> negotiate -> wait-your-turn -> active sequence resolves inside
         one synchronous solver-visible tool call, not across several.

    Sits ABOVE ScheduledEnvironment, never below it and never inside it (docker_task_scheduler.py
    is not modified) -- mirrors SolverMemoryEnvironment's own wrapping pattern in
    run_programbench_subrun_repo_agents.py exactly (execute() intercepts/bypasses conditionally,
    everything else passes through via __getattr__).

    Why a window-bypassed command still goes through _execute_with_concurrency_gate() rather than
    calling raw_env.execute() completely unsupervised: communication_bypass_cpu_threshold is
    fixed at 0 in this script specifically so nothing can dodge the shared execution-slot limit
    (max_concurrent_tasks, defaulting to 1 -- "only one active core, everything sequential").
    Skipping ScheduledEnvironment/DockerTaskScheduler.run() entirely for a leased command would
    silently reopen exactly that hole: the scheduler's own concurrency counter
    (scheduler._active_tasks vs scheduler.max_concurrent_tasks) would never see the bypassed
    command, so it could freely admit a DIFFERENT worker's negotiated command at the same moment,
    running two containers' commands concurrently despite max_concurrent_tasks=1. Confirmed by
    reading DockerTaskScheduler.run() (docker_task_scheduler.py:380-435): the concurrency gate is
    a small, self-contained wait/increment/decrement/notify block around scheduler._condition,
    scheduler._active_tasks, and the public scheduler.max_concurrent_tasks -- _execute_with_
    concurrency_gate() below reproduces exactly that block from outside the class (reading/
    mutating the scheduler's own existing counter, not a separate uncoordinated one) so a bypassed
    command still competes for the exact same slots as a negotiated one. This does not touch
    docker_task_scheduler.py's source; it only calls the same synchronization primitive that
    module already constructs and uses for this exact purpose. Deliberately narrow: it does NOT
    also replicate _active_high_tasks/_completed_high_tasks bookkeeping (that only affects
    low-priority evaluation-task admission timing, not the "only one thing executes at a time"
    invariant this exists to preserve) or the dashboard's _active_task_details snapshot (a
    bypassed command's live-queue display is a cosmetic gap, not a correctness one).

    A worker's own deadline can, in principle, run out mid-window (its clamped bypass duration
    then falls short of the board's own nominal window length). This is a deliberate,
    accepted simplification: the board's internal expiry timer is NOT re-synced to the shorter,
    per-worker-clamped duration in that case -- the worker simply falls back to needing a fresh
    request_execution_slot call slightly before the board would otherwise have released it, and
    the next queued worker proceeds once the board's own timer (or an explicit relinquish)
    releases it. Not worth the added cross-thread bookkeeping for what is already an
    end-of-subrun edge case.
    """

    _REQUEST_PREFIX = "request_execution_slot"
    _REQUEST_PATTERN = re.compile(r"^request_execution_slot\s+(\d+(?:\.\d+)?)\s*$")
    # Hard ceiling on any single execution window, independent of and in addition to
    # the existing remaining-deadline cap -- no worker can ever hold the resource for
    # more than this many seconds in one grant, regardless of what it requests, what
    # it bids, or what position it wins. Enforced twice: once upstream, in
    # _handle_request, so the negotiator's own declaration is anchored to an
    # achievable number rather than negotiating over a request that was always going
    # to be trimmed later; and again, defensively, in _activate, so the cap holds even
    # if some other path ever supplies a duration this worker didn't request itself.
    MAX_EXECUTION_SLOT_SECONDS = 300.0
    # Observed live: a worker can win a real queued position (e.g. position 3) whose wait is
    # dominated by the granted windows of the positions ahead of it -- and since that queued wait
    # genuinely ticks the deadline (confirmed correct, matching real execution time), a worker
    # could commit to a position it can mathematically never benefit from, burning its entire
    # remaining deadline on a wait that either grants it a ~0s window on arrival or leaves the
    # deadline hitting 0 before its turn ever comes, killing the whole subrun with 0% progress.
    # This is the minimum leftover window (deadline minus estimated wait-ahead) required before
    # _handle_request will actually commit to the wait at all; below this it skips the wait
    # entirely (see _handle_request) rather than blindly spending the deadline on it.
    _MIN_VIABLE_WINDOW_AFTER_WAIT_SECONDS = 5.0
    _REQUEST_USAGE_TEXT = (
        "request_execution_slot: malformed call, no request was made.\n"
        "Correct usage (must be the ENTIRE command, nothing else in the same call):\n"
        "request_execution_slot <seconds>\n"
        "Example: request_execution_slot 120"
    )

    def __init__(
        self,
        scheduled_env: Any,
        raw_env: Any,
        *,
        scheduler: DockerTaskScheduler,
        schedule_board: "AuctionScheduleBoard",
        budget_tracker: "SubrunBudgetTracker",
        request_registry: "SolverRequestRegistry",
        negotiation_history_log_path: Path,
        instance_id: str,
        repo_slug: str,
        deadline_at_monotonic: "float | DeadlineClock",
        logger: TeeLogger | None = None,
        repo_name: str = "",
        max_execution_slot_seconds: float = MAX_EXECUTION_SLOT_SECONDS,
    ) -> None:
        self._scheduled_env = scheduled_env
        self._raw_env = raw_env
        self._scheduler = scheduler
        self._schedule_board = schedule_board
        self._budget_tracker = budget_tracker
        self._request_registry = request_registry
        # Instance override of the class constant above -- self.MAX_EXECUTION_SLOT_SECONDS (read
        # everywhere else in this class) resolves to this per-instance value first, per normal
        # Python attribute lookup, so nothing downstream needs its own separate reference.
        self.MAX_EXECUTION_SLOT_SECONDS = max_execution_slot_seconds
        self._negotiation_history_log_path = negotiation_history_log_path
        self._instance_id = instance_id
        self._repo_slug = repo_slug
        self._deadline_at_monotonic = deadline_at_monotonic
        self._logger = logger
        self._repo_name = repo_name
        self._lease_expires_at_monotonic: float | None = None
        self._last_processed_round_id: int = 0
        # (round_id, granted_seconds) per window secured this subrun -- report context only.
        self.lease_grants: list[tuple[int, float]] = []
        # seconds actually spent blocked in wait_for_turn this subrun -- report context only.
        self.wait_events: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduled_env, name)

    @staticmethod
    def _extract_command(action: "dict | str") -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, dict):
            return str(action.get("command") or "")
        return ""

    @staticmethod
    def _plain_result(output: str) -> dict[str, Any]:
        return {"output": output, "returncode": 0, "exception_info": ""}

    def _execute_with_concurrency_gate(self, action: "dict | str", cwd: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        scheduler = self._scheduler
        with scheduler._condition:
            while scheduler._active_tasks >= scheduler.max_concurrent_tasks:
                scheduler._condition.wait()
            scheduler._active_tasks += 1
        try:
            return self._raw_env.execute(action, cwd=cwd, **kwargs)
        finally:
            with scheduler._condition:
                scheduler._active_tasks -= 1
                scheduler._condition.notify_all()

    def _activate(self, duration_seconds: float, price: int) -> tuple[float, int]:
        remaining_deadline = _remaining_seconds(self._deadline_at_monotonic)
        # MAX_EXECUTION_SLOT_SECONDS is already applied upstream, to the solver's own
        # request, in _handle_request -- this second cap is a defensive backstop, not
        # redundant belt-and-suspenders for its own sake: duration_seconds here comes
        # from the negotiator's own resolved declaration (DESIRED_DURATION_SECONDS),
        # which the prompt asks to "generally match" the solver's request but never
        # hard-enforces -- nothing stops a declaration from stating a larger number.
        # This is the one point that actually determines the granted window, so it's
        # the one point where the cap must hold regardless of what produced the input.
        granted_seconds = max(0.0, min(float(duration_seconds), remaining_deadline, self.MAX_EXECUTION_SLOT_SECONDS))
        self._lease_expires_at_monotonic = time.monotonic() + granted_seconds
        # Env-gated hard budget cap (AUCTION_HARD_BUDGET_CAP=1): exactly like the duration cap
        # above, this is the one point that actually determines what gets charged, so it is the
        # one point where the bid-vs-remaining-budget constraint must hold regardless of what
        # the negotiation produced. The stated price is clamped to the worker's remaining
        # per-subrun balance; deduct() then also refuses to go below zero.
        charge_price = int(price)
        if os.environ.get("AUCTION_HARD_BUDGET_CAP", "").strip() == "1":
            charge_price = min(charge_price, max(0, self._budget_tracker.remaining(self._repo_slug)))
        deducted = self._budget_tracker.deduct(self._repo_slug, charge_price)
        self.lease_grants.append((self._last_processed_round_id, granted_seconds))
        return granted_seconds, deducted

    def execute(self, action: "dict | str", cwd: str = "", **kwargs: Any) -> dict[str, Any]:
        now = time.monotonic()
        command = self._extract_command(action).strip()
        is_request_prefixed = command.startswith(self._REQUEST_PREFIX)

        # Unconditional, checked before any other branch: once the deadline is genuinely
        # exhausted, raise rather than return another plain rejection message. Without this, a
        # solver that doesn't recognize "no active window, 0s remaining" as a terminal state can
        # spin indefinitely -- observed live, one worker repeated a no-op command 270+ times
        # after its deadline ran out, each one a real, billed API call, before mini-swe-agent's
        # own much longer wall_time_limit_seconds ceiling (8x the nominal deadline by default)
        # finally cut it off. _check_hard_deadline() in run_programbench_instance_direct already
        # raises this exact exception, but only as a bookend check immediately before and after
        # agent.run() -- there was never a check *during* the loop itself. Safe to place ahead of
        # the STATE 1 check unconditionally: _activate() always caps a granted window's expiry to
        # at most the remaining deadline at grant time, so STATE 1 can never be legitimately
        # active while the deadline is already exhausted -- this can't misfire and cut off a
        # real, still-valid window. Raising here (not inside the STATE-2 request_execution_slot
        # branch below) is deliberate: that branch's own try/except would otherwise catch and
        # swallow it as an ordinary request failure instead of letting it propagate.
        if _remaining_seconds(self._deadline_at_monotonic) <= 0:
            raise HardDeadlineExceededError(
                f"Hard deadline expired for {self._repo_name} on {self._instance_id} "
                "while negotiating or waiting for the time-slot auction."
            )

        # STATE 1: currently holding an active execution window.
        if self._lease_expires_at_monotonic is not None and now < self._lease_expires_at_monotonic:
            remaining = self._lease_expires_at_monotonic - now
            if is_request_prefixed:
                return self._plain_result(
                    f"You already have an active execution window with ~{remaining:.0f}s "
                    "remaining -- no need to request another yet."
                )
            result = dict(self._execute_with_concurrency_gate(action, cwd, kwargs))
            result["output"] = (
                str(result.get("output") or "")
                + f"\n\n[execution window: ~{remaining:.0f}s remaining before this task returns to negotiation]"
            )
            return result

        # STATE 2: no active window.
        if is_request_prefixed:
            match = self._REQUEST_PATTERN.match(command)
            if match is None:
                return self._plain_result(self._REQUEST_USAGE_TEXT)
            try:
                return self._handle_request(float(match.group(1)))
            except Exception as exc:
                # Belt-and-suspenders, same posture as SolverMemoryEnvironment: a bug in the
                # request-handling logic must never crash the agent loop -- it just means this
                # request didn't succeed, and the solver can try request_execution_slot again.
                if self._logger is not None:
                    self._logger.log(f"[{self._repo_name}][auction] request handling failed (non-fatal): {exc}")
                return self._plain_result(
                    "request_execution_slot failed unexpectedly -- no window was granted. Try again."
                )

        remaining_deadline = _remaining_seconds(self._deadline_at_monotonic)
        remaining_budget = self._budget_tracker.remaining(self._repo_slug)
        return self._plain_result(
            "No active execution window -- ordinary commands cannot run yet. You have "
            f"~{remaining_deadline:.0f}s until your deadline and {remaining_budget} budget "
            "point(s) left this subrun. Call request_execution_slot <seconds> first to negotiate "
            "access (state a duration you can defend, capped by both of those)."
        )

    def _format_coordination_withdrawal(self, exc: SchedulerCoordinationWithdrawal) -> str:
        lines = [
            "Your negotiator voluntarily withdrew this request from the round instead of "
            "contesting it -- exactly like losing a position or sitting out, this costs nothing "
            "(no budget was charged, and your deadline was not touched beyond the negotiation "
            "itself), but no execution window was granted this call.",
        ]
        if exc.peer_repo_name:
            lines.append(f"It did so to make way for: {exc.peer_repo_name}.")
        if exc.reason:
            lines.append(f"Its stated reason: {exc.reason}")
        if exc.shared_context:
            lines.append(f"The clue it handed that peer: {exc.shared_context}")
        lines.append("Call request_execution_slot again if you still need an execution window.")
        return "\n".join(lines)

    def _handle_request(self, requested_seconds: float) -> dict[str, Any]:
        remaining_deadline = _remaining_seconds(self._deadline_at_monotonic)
        capped_seconds = max(0.0, min(requested_seconds, remaining_deadline, self.MAX_EXECUTION_SLOT_SECONDS))
        if capped_seconds <= 0:
            return self._plain_result("Your remaining deadline is exhausted -- no execution window can be granted.")

        self._request_registry.set_requested_duration(self._repo_slug, capped_seconds)
        try:
            synthetic_command = f": # solver requests an execution slot of approximately {capped_seconds:.0f}s"
            # Must be the same {"command": ...} dict shape mini-swe-agent's own DockerEnvironment
            # expects (DockerEnvironment.execute() does action.get("command", "")) -- a bare
            # string here raises AttributeError only once the scheduler actually dispatches it
            # (i.e. after negotiation has already fully resolved), which every call before this
            # fix hit 100% of the time, silently swallowed by execute()'s own outer except-clause
            # as "request_execution_slot failed unexpectedly." Confirmed via a real run: this
            # meant negotiation always completed genuinely, but no window was ever granted, no
            # budget was ever deducted, and no real command ever executed -- ever, across every
            # subrun since this method was introduced.
            self._scheduled_env.execute({"command": synthetic_command}, cwd="")
        except SchedulerCoordinationWithdrawal as exc:
            # Only reachable when --enable-solution-sharing is on (see main()): the scheduler
            # only ever produces this when allow_coordination_withdrawal=True, which is tied
            # 1:1 to that flag. ScheduledEnvironment.allow_solution_withdrawal is deliberately
            # kept False (see its construction site) so this exception reaches here instead of
            # being swallowed by ScheduledEnvironment itself with SWE-bench-flavored wording.
            return self._plain_result(self._format_coordination_withdrawal(exc))
        finally:
            self._request_registry.clear(self._repo_slug)

        schedule = _extract_latest_round_schedule(
            self._negotiation_history_log_path,
            instance_id=self._instance_id,
            since_round_id=self._last_processed_round_id,
        )
        if schedule is None:
            return self._plain_result(
                "No new negotiation round resolved for your request -- call request_execution_slot again."
            )
        round_id, ordered_entries, repo_slug_by_task_id = schedule
        self._last_processed_round_id = round_id
        if ordered_entries:
            self._schedule_board.post_schedule(round_id, ordered_entries)
        my_entry = next((entry for entry in ordered_entries if entry[0] == self._repo_slug), None)
        if my_entry is None:
            # Self-heal the solo-negotiation-round gap: queue_negotiation.py's negotiate() has a
            # dedicated short-circuit for a single candidate (no peers to negotiate with) that
            # skips the whole declaration/reasoning/discussion/ballot LLM pipeline entirely --
            # cheap and correct for what it's FOR, but its canned declaration never fills in
            # DESIRED_DURATION_SECONDS, so this worker's own entry always parses to duration<=0
            # and gets dropped from ordered_entries, landing here every single time. Without this
            # check, a worker that outlasts its peers (the normal end state of every subrun) gets
            # stuck retrying request_execution_slot forever once it goes solo -- confirmed live:
            # one worker burned 100+ consecutive solver turns this way, each a real billed call,
            # for zero progress, until it ran out of message budget. Detected here (not fixed
            # inside queue_negotiation.py) because the value that actually belongs in this gap --
            # what THIS worker itself just asked for, already deadline/budget/cap-clamped into
            # capped_seconds above -- is only known on this side of the shared negotiation module.
            if len(repo_slug_by_task_id) == 1 and self._repo_slug in repo_slug_by_task_id.values():
                solo_task_id = next(iter(repo_slug_by_task_id))
                my_entry = (self._repo_slug, solo_task_id, capped_seconds, 0)
                self._schedule_board.post_schedule(round_id, [my_entry])
            else:
                return self._plain_result(
                    "Your request wasn't reflected in this round's resolved schedule -- call "
                    "request_execution_slot again."
                )
        _repo_slug, _task_id, duration_seconds, price = my_entry
        position_hint = next(
            (index + 1 for index, entry in enumerate(ordered_entries) if entry[0] == self._repo_slug),
            None,
        )

        # Before committing to wait_for_turn (which genuinely ticks the deadline for the whole
        # wait -- confirmed correct elsewhere), check whether this worker can mathematically
        # come out ahead: sum the granted durations of every position ranked ahead of it in this
        # round's resolved schedule and compare that to what's actually left on the clock right
        # now. If the positions ahead alone would exhaust (or nearly exhaust) the remaining
        # deadline, waiting it out either grants a near-0s window on arrival or kills the subrun
        # outright before a turn ever comes -- observed live, twice, in the same run. Skipping
        # here costs nothing (no deadline is spent) and withdraws this entry so it doesn't jam
        # the board's strict-FIFO activation for whoever is queued behind it.
        positions_ahead = ordered_entries[: (position_hint or 1) - 1]
        estimated_wait_ahead = sum(entry_duration for _slug, _tid, entry_duration, _price in positions_ahead)
        remaining_deadline_before_wait = _remaining_seconds(self._deadline_at_monotonic)
        if remaining_deadline_before_wait - estimated_wait_ahead < self._MIN_VIABLE_WINDOW_AFTER_WAIT_SECONDS:
            self._schedule_board.withdraw(self._repo_slug, _task_id)
            return self._plain_result(
                f"Skipping the queued wait for position {position_hint}: the "
                f"~{estimated_wait_ahead:.0f}s of higher-priority windows ahead of you in this "
                f"round's schedule would leave you under {self._MIN_VIABLE_WINDOW_AFTER_WAIT_SECONDS:.0f}s "
                f"of your ~{remaining_deadline_before_wait:.0f}s remaining deadline by the time your own "
                "turn arrived -- not enough for any real work, so this call does not wait and your "
                "deadline is untouched. Call request_execution_slot again with a smaller ask, or accept "
                "an earlier/cheaper position next round."
            )

        # A solo self-heal (see the my_entry is None branch above) never populated the local
        # ordered_entries with its own synthesized entry -- only self._schedule_board's internal
        # queue got it, via post_schedule. Fall back to [my_entry] so the status below still
        # reports "position 1 of 1, held by me" instead of an empty/wrong total.
        effective_ordered_entries = ordered_entries if ordered_entries else [my_entry]
        wait_start = time.monotonic()
        if self._request_registry is not None:
            self._request_registry.set_wait_for_turn_status(
                self._repo_slug,
                position=position_hint or 1,
                total=len(effective_ordered_entries),
                holder_repo_slug=effective_ordered_entries[0][0],
            )
        try:
            granted_duration = self._schedule_board.wait_for_turn(
                self._repo_slug,
                _task_id,
                # See wait_for_turn's own docstring for why this is estimated_wait_ahead (this
                # round's own real declared durations for every entry ranked ahead of this one)
                # plus one full window of slop, not a flat constant: it has to stay above whatever
                # a legitimately deep, healthy queue could ever really take, so it only fires for a
                # genuinely stuck state, never for ordinary contention.
                max_wait_seconds=estimated_wait_ahead + self.MAX_EXECUTION_SLOT_SECONDS,
            )
        finally:
            if self._request_registry is not None:
                self._request_registry.clear_wait_for_turn_status(self._repo_slug)
        waited = time.monotonic() - wait_start
        if waited > 0.01:
            self.wait_events.append(waited)
        if granted_duration is None:
            return self._plain_result(
                "Lost your queued position unexpectedly -- call request_execution_slot again."
            )

        granted_seconds, deducted = self._activate(granted_duration, price)
        capped_note = ""
        if os.environ.get("AUCTION_HARD_BUDGET_CAP", "").strip() == "1" and deducted < int(price):
            capped_note = (
                f" Note: your stated bid of {int(price)} exceeded your remaining per-subrun "
                f"budget, so only {deducted} was actually available and charged -- the budget "
                "is a hard limit in this run."
            )
        return self._plain_result(
            f"Negotiation resolved (round {round_id}, secured position {position_hint}): you now "
            f"have complete, uninterrupted execution access for ~{granted_seconds:.0f}s (charged "
            f"{deducted} from this subrun's budget).{capped_note} Issue your commands directly now -- no "
            "further negotiation until this window expires."
        )

    def summary_text(self) -> str:
        parts: list[str] = []
        if self.lease_grants:
            total_seconds = sum(seconds for _round_id, seconds in self.lease_grants)
            parts.append(
                f"secured {len(self.lease_grants)} execution window(s) this subrun, totalling "
                f"~{total_seconds:.0f}s of granted uninterrupted execution time"
            )
        else:
            parts.append("secured no execution windows this subrun")
        if self.wait_events:
            total_wait = sum(self.wait_events)
            parts.append(
                f"waited ~{total_wait:.0f}s total for its turn across {len(self.wait_events)} "
                "queued position(s)"
            )
        parts.append(f"{self._budget_tracker.remaining(self._repo_slug)} budget point(s) remain this subrun")
        return "Auction status: " + "; ".join(parts) + "."

    def cleanup(self) -> Any:
        return self._scheduled_env.cleanup()

    def serialize(self) -> dict[str, Any]:
        return self._scheduled_env.serialize()

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self._scheduled_env.get_template_vars(**kwargs)


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
    """Adapted from run_programbench_subrun_repo_agents.py's own run_programbench_instance_direct.
    Same shape (hard-deadline-check, tar.gz submission via copy_submission), plus one addition:
    a TimeSlotLeaseEnvironment sits between ScheduledEnvironment and SolverMemoryEnvironment,
    turning resolved auction rounds into sequential execution windows. announce_deadline_in_output
    is always True here (unlike the budget-bidding sibling, which always passes False) -- this
    variant's whole point is that the deadline IS visible.

    Returns (traj_path, exit_status, submission_copied, lease_summary_text).
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
    lease_env: TimeSlotLeaseEnvironment | None = None
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
                # Deliberately False even when enable_solution_sharing is on: True would have
                # ScheduledEnvironment itself catch SchedulerCoordinationWithdrawal and convert it
                # using _format_coordination_update's SWE-bench-flavored text ("draft or revise
                # the patch now"), which doesn't fit a from-scratch reconstruction task. Instead
                # TimeSlotLeaseEnvironment._handle_request catches the same exception itself
                # (letting it propagate up from here) and crafts a message that fits this script's
                # own auction/reconstruction framing.
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
                negotiation_history_log_path=negotiation_history_log_path,
                instance_id=instance_id,
                repo_slug=repo_slug_value,
                deadline_at_monotonic=worker_deadline_at_monotonic,
                logger=logger,
                repo_name=repo_name,
            )
            env = lease_env
        # Outermost layer -- sits above TimeSlotLeaseEnvironment/ScheduledEnvironment (or the raw
        # env, if scheduler is None), never below it, so a save_persistent_memory call never
        # reaches negotiation or the lease check, and every other command passes through
        # unaffected by this wrapping. raw_env (kept separately, above) is untouched, so
        # copy_submission still works exactly as before.
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
        # Same fix as the ProgramBench siblings: only re-check the deadline when the agent did
        # NOT already produce a clean terminal result, so a late-but-real "Submitted" (within
        # mini-swe-agent's own generous internal wall_time_limit_seconds ceiling) isn't discarded.
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
        # "Collapse forward" -- if this worker's task finishes (or errors out) while it still
        # held an active execution window, release it immediately instead of leaving the next
        # queued worker waiting out time nobody is actually using. Safe to call unconditionally
        # even if this worker was never the active holder (no-op in that case).
        if scheduler is not None:
            schedule_board.relinquish_if_active(repo_slug_value)
            # See AuctionScheduleBoard.abandon_repo_pending_wait's own docstring: relinquish above
            # only covers this worker's session ending while it was the ACTIVE holder -- this
            # covers it ending while still queued, waiting for its turn, which the board's own
            # staleness self-heal alone cannot reliably catch.
            schedule_board.abandon_repo_pending_wait(repo_slug_value)
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
    lease_summary_text = lease_env.summary_text() if lease_env is not None else ""
    return traj_path, exit_status or "", submission_copied, lease_summary_text


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
    schedule_board: "AuctionScheduleBoard",
    budget_tracker: "SubrunBudgetTracker",
    request_registry: "SolverRequestRegistry",
    initial_budget: int | None = None,
    subrun_wave_gate: "SubrunWaveGate | None" = None,
    instance_runner: "Callable[..., tuple[Path, str, bool, str]] | None" = None,
) -> WorkerResult:
    # Swappable purely so run_programbench_auction_subrun_harness_agents.py can reuse this
    # entire subrun loop (wave gate, budget, reporting, evaluation -- none of which are
    # mini-swe-agent-specific) with a different solver harness, without duplicating any of it.
    # Defaults to the exact same call every existing caller already makes; a caller that never
    # passes instance_runner sees zero behavior change.
    resolved_instance_runner = instance_runner or run_programbench_instance_direct

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
        f"(fresh each subrun, scale={args.subrun_deadline_scale}); time-slot auction for execution priority."
    )
    if monitor is not None:
        monitor.set_thread(assignment.repo_slug, threading.get_native_id())
        monitor.update_worker(assignment.repo_slug, deadline_minutes=worker_deadline_minutes)
        monitor.add_event(assignment.repo_slug, f"Worker thread started for {assignment.repo_name} (auction subrun mode).")

    if scheduler is not None:
        # Paired with unregister_worker() in the finally block below. Lets the
        # scheduler's communication_min_k gate relax once enough OTHER workers have
        # permanently finished their whole batch that the configured min-k can no
        # longer ever be satisfied -- see DockerTaskScheduler.register_worker's
        # docstring for the live deadlock this fixes.
        scheduler.register_worker(assignment.repo_slug)

    try:
        for subrun_index, instance_id in enumerate(assignment.instance_ids, start=1):
            if instance_id not in instances_by_id:
                raise KeyError(f"Task {instance_id} was not found in the loaded ProgramBench task list.")

            if subrun_wave_gate is not None and subrun_index > 1:
                logger.log(
                    f"[{assignment.repo_name}] waiting for every other worker to finish "
                    f"subrun {subrun_index - 1} before starting subrun {subrun_index}."
                )
                # Temporarily unregister from the scheduler's own communication_min_k
                # accounting for the duration of this wait -- this worker genuinely will not
                # announce anything (no negotiation, no evaluation) until the wave releases it,
                # but scheduler.register_worker() was already called once, up front, for this
                # worker's whole lifetime, so without this it stays counted as "could still
                # announce" and _effective_communication_min_k_locked() never relaxes below the
                # configured min-k. That is exactly the deadlock register_worker/unregister_worker
                # were built to prevent for a PERMANENTLY finished worker -- this wave-gate pause
                # is a new, temporary version of the same situation, so it needs the same
                # treatment: with a still-registered peer paused here and unable to ever announce
                # again until its still-active peers finish, a fixed min-k > (active peer count)
                # becomes permanently unreachable, hanging every remaining worker's communication
                # queue forever. Re-registering immediately after the wait resumes normal
                # participation for this worker's own next negotiation round.
                if scheduler is not None:
                    scheduler.unregister_worker(assignment.repo_slug)
                try:
                    subrun_wave_gate.wait_for_wave(assignment.repo_slug, subrun_index)
                finally:
                    if scheduler is not None:
                        scheduler.register_worker(assignment.repo_slug)

            subrun_start_epoch = time.time()
            # Fresh budget every subrun (confirmed requirement) -- unlike the budget-bidding
            # sibling script's ledger, this is NOT a running total across the whole batch.
            # initial_budget (per-worker override, e.g. for asymmetric-budget ablations) takes
            # precedence over the uniform args.initial_budget when the caller supplies one.
            budget_tracker.start_subrun(
                assignment.repo_slug,
                initial_budget if initial_budget is not None else args.initial_budget,
            )
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

                def _on_deadline_tick(
                    total_extended_seconds: float,
                    _new_deadline_at_monotonic: float,
                    _repo_slug: str = assignment.repo_slug,
                    _subrun_start_epoch: float = subrun_start_epoch,
                ) -> None:
                    # Fires on EVERY credit, including the silent per-poll increments
                    # extend_silently() applies while a communication-queue wait is
                    # still in progress -- unlike _on_deadline_extend above, which only
                    # fires once, at the end, to keep the log to one clean summary
                    # line. Without this, the monitor's own deadline_started_epoch
                    # (and the deadline_remaining_seconds a live dashboard computes
                    # from it) would only get corrected retroactively too, so it would
                    # keep visibly counting down for the full duration of a long wait
                    # even though the real DeadlineClock backing actual enforcement is
                    # correctly staying flat underneath. No logging here on purpose --
                    # this can fire every few seconds during a long stall.
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
                    on_tick=_on_deadline_tick,
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
                _build_pb_auction_solver_context(
                    assignment=assignment,
                    runtime_metadata=runtime_metadata,
                    shared_token_pool=shared_token_pool,
                    remaining_budget=budget_tracker.remaining(assignment.repo_slug),
                ),
                _PB_AUCTION_SOLVER_INCENTIVE_TEXT,
                _PB_AUCTION_SOLVER_MEMORY_TOOL_TEXT if solver_memory_store is not None else "",
            ]
            solver_context = "\n\n".join(part for part in solver_note_parts if part)

            deadline_hit = False
            traj_path, exit_status, submission_copied, lease_summary_text = resolved_instance_runner(
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
                negotiation_history_log_path=negotiation_history_log_path,
                schedule_board=schedule_board,
                budget_tracker=budget_tracker,
                request_registry=request_registry,
                retry_note=solver_context if solver_context else None,
                monitor=monitor,
                enforce_hard_deadline=True,
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
            # deadline_hit is deliberately NOT a precondition here (it used to be) -- see the
            # matching comment in run_programbench_subrun_repo_agents.py's own worker loop for the
            # full reasoning: a timed-out subrun still has whatever it built sitting in the
            # container, submission_copied means there's normally something real to evaluate, and
            # the evaluator's own compile step is fault-tolerant (a non-compiling submission just
            # scores 0.0 via not_run tests, it doesn't raise) -- so evaluating it is safe and is
            # the only way to distinguish 0% progress from 90% progress at deadline time.
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

            snapshot_end = report_runtime_metadata.build_summary(assignment.repo_slug)
            metadata_snapshots = "\n".join(
                part for part in [
                    f"At subrun start: {snapshot_start}" if snapshot_start else "",
                    f"At subrun end: {snapshot_end}" if snapshot_end else "",
                    lease_summary_text,
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
                deadline_hit=deadline_hit,
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
                deadline_hit=deadline_hit,
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
            if subrun_wave_gate is not None:
                subrun_wave_gate.mark_subrun_complete(assignment.repo_slug, subrun_index)

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
            # This worker's whole assigned batch is done -- it will never announce
            # another task. Must fire on every exit path (including the raise above),
            # which is why this is in the same finally as the deadline-clock cleanup.
            scheduler.unregister_worker(assignment.repo_slug)
        if subrun_wave_gate is not None:
            # Same posture as scheduler.unregister_worker above: this worker will never reach
            # another subrun (normal completion OR an unhandled error above both land here), so
            # it must stop being counted as a party the SubrunWaveGate waits on -- otherwise a
            # peer still mid-batch would block forever waiting for a wave this worker can no
            # longer ever complete.
            subrun_wave_gate.unregister(assignment.repo_slug)


# ---------------------------------------------------------------------------
# CLI + orchestration.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ProgramBench workflow as a sequence of isolated subruns (one per assigned "
            "task, drawn from an unrelated batch each worker is assigned up front), each under "
            "its own fresh hard deadline and a fresh per-subrun bidding budget, with execution "
            "priority decided by a sequential time-slot auction: a resolved round produces a "
            "full ordered schedule of reserved execution windows (not a single winner), taken in "
            "turn, with every worker that secures a position paying its own stated price for it."
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
        "the environment stops intercepting it.",
    )
    parser.add_argument(
        "--initial-budget", type=int, default=1000,
        help="Starting bidding budget per worker (points), same for every worker in the run "
        "unless --worker-initial-budgets overrides it per worker. Unlike the budget-bidding "
        "sibling script's --initial-budget, this resets to this exact amount at the start of "
        "EVERY subrun -- it is not a running total across a worker's whole assigned batch.",
    )
    parser.add_argument(
        "--worker-initial-budgets", type=str, default=None,
        help="Comma-separated per-worker starting-budget override, e.g. '1500,1000,1000' for "
        "--worker-count 3 -- overrides --initial-budget for the matching worker (list position i "
        "maps to the i-th assigned worker, same order they're listed in run_log.txt). Must have "
        "exactly --worker-count entries. Ablation support for asymmetric-budget experiments; "
        "omit for the default uniform budget.",
    )
    parser.add_argument(
        "--worker-deadline-minutes", type=str, default=None,
        help="Comma-separated per-worker hard-deadline override in minutes, e.g. '12,18,9' for "
        "--worker-count 3 -- overrides the shared, hash-derived deadline for the matching worker "
        "(same positional mapping as --worker-initial-budgets). Must have exactly --worker-count "
        "entries. Ablation support for asymmetric-deadline experiments; omit for the default "
        "shared deadline (identical for every worker, scaled by --subrun-deadline-scale).",
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
    # Opt-IN, off by default -- the inverse of run_swebench_subrun_repo_agents.py's
    # --disable-solution-sharing (on by default there, opt-out). Off by default here because
    # cross-worker attempt/clue sharing has never been exercised in ProgramBench (each worker
    # reconstructs a different binary, so there's no shared codebase the way SWE-bench workers
    # can share one), so it needs to be an explicit, considered choice rather than inherited
    # behavior. Wires up ballot-phase "wait_for_peer" (voluntary self-withdrawal from the current
    # round, with an optional free-text "shared_context" clue meant for a specific peer) across
    # the scheduler, ScheduledEnvironment, and QueueNegotiator -- see the three call sites below
    # and _handle_request's new SchedulerCoordinationWithdrawal handling. Deliberately does NOT
    # enable "delegate_to_peer": that action raises SchedulerExecutionDeferred to park a whole
    # instance for a later attempt, which run_swebench_subrun_repo_agents.py has a dedicated
    # handler for and this script does not (ProgramBench has no multi-instance retry batch to
    # park into) -- wiring it here would trade the queued-wait starvation bug just fixed for a
    # different, unhandled crash path.
    parser.add_argument(
        "--enable-solution-sharing",
        action="store_true",
        help=(
            "Let a worker's ballot voluntarily withdraw its own current request from the round "
            "and hand a specific peer a free-text clue (action=\"wait_for_peer\" in the "
            "coordination_decision the ballot schema already asks for). Off by default -- "
            "ProgramBench workers reconstruct different binaries, so this side channel has no "
            "guaranteed use, but it exists and is wired up properly when you want to observe "
            "whether/how agents choose to use it."
        ),
    )
    # Opt-IN, off by default -- bids are revisable during negotiation unless this is set,
    # matching this project's current default behavior (added after the original sealed-bid
    # design: DESIRED_DURATION_SECONDS/BID_PRICE used to be genuinely fixed once declared, and a
    # tied or contested price had no in-channel resolution -- observed live, this pushed
    # negotiators toward persuasion and reciprocal inter-round deals as the only remaining lever,
    # producing coalition-shaped behavior that may have been an artifact of the fixed-bid
    # constraint rather than a standalone tendency. This flag exists specifically to A/B that:
    # run once with it (original fixed-bid mechanics, self-gating prompt text reverts to "your
    # bid cannot move") and once without (current default, revision genuinely allowed) and
    # compare coalition/fabrication rates between the two conditions.
    parser.add_argument(
        "--fixed-starting-bid",
        action="store_true",
        help=(
            "Lock DESIRED_DURATION_SECONDS/BID_PRICE at whatever a worker declared -- "
            "direct_message/response never offer the revised_duration_seconds/revised_bid_price "
            "schema keys, and even a hallucinated attempt to use them is ignored. Off by "
            "default, meaning bids ARE revisable during negotiation (the current default "
            "behavior). Pass this to reproduce the original sealed-bid-once design, e.g. to "
            "compare negotiation dynamics (especially coalition-shaped behavior around tied "
            "bids) against the revisable-bid default."
        ),
    )
    # Deliberately no --communication-bypass-cpu-threshold flag in this script: this design
    # requires every task to genuinely go through the auction, with no bypass path at all
    # ("there is no bypass threshold in this situation. I.e it is 0" -- verbatim design
    # requirement). DockerTaskScheduler is constructed below with communication_bypass_cpu_
    # threshold=0.0, which docker_task_scheduler.py's own bypass check (`> 0`) treats as
    # permanently disabled, and no cpu_percent_provider is even wired up since it would go unused.
    parser.add_argument(
        "--queue-negotiation-prompts",
        default=str(DEFAULT_QUEUE_NEGOTIATION_PROMPTS),
        help="YAML file containing coordination prompts. Defaults to this script's own time-slot "
        "auction variant (deadline fields visible, DESIRED_POSITION/DESIRED_DURATION_SECONDS/"
        "BID_PRICE declaration convention).",
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
        logger.log("Starting subrun-based agentic ProgramBench run (time-slot auction mode).")
        logger.log(f"Run ID: {run_layout['run_id']}")
        logger.log(f"Per-run output directory: {output_root}")

        require_programbench_package()
        api = load_programbench_minisweagent_api()

        from programbench.utils.load_data import load_all_instances

        all_instances_for_solving = load_all_instances(include_tests=False)
        instances_by_id = {str(inst["instance_id"]): inst for inst in all_instances_for_solving}
        all_instances_for_eval = load_all_instances(include_tests=True)
        eval_instances_by_id = {str(inst["instance_id"]): inst for inst in all_instances_for_eval}

        logger.log(f"Deadline policy: hard cutoffs, visible, reset fresh every subrun (scale={args.subrun_deadline_scale}).")

        config = build_programbench_minisweagent_config(args, api)

        programbench_bin = args.programbench_bin or str(Path(args.python_bin).with_name("programbench"))
        if not Path(programbench_bin).exists():
            programbench_bin = "programbench"  # fall back to PATH resolution
        args.programbench_bin = programbench_bin

        shared_token_pool = SharedTokenPool(total_tokens=args.token_pool_size, disabled=args.disable_token_pool_limit)
        # hard_deadlines_enabled=True (unlike the budget-bidding sibling): this variant's whole
        # point is that the deadline is real and visible, so build_summary() should emit its
        # "deadline_policy: hard termination at 0s remaining" line as accurate context.
        report_runtime_metadata = RepoRuntimeMetadata(hard_deadlines_enabled=True)
        runtime_metadata = None if args.disable_system_metadata else report_runtime_metadata
        # SubrunReportMemory is used directly, unmodified (unlike the budget-bidding sibling's
        # ProgramBenchReportMemory override) -- its hardcoded deadline_hit= label is accurate
        # context here, not a leak, since the deadline is visible to every agent in this variant.
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
        # (queue_negotiation_time_auction_prompts.yaml) never matches it, so BOTH kwargs passed
        # here are a no-op either way; they're passed through honestly rather than hardcoded so a
        # future shared prompts file wouldn't silently do the wrong thing. The actual on/off
        # switches for this run are args.enable_solution_sharing (threaded into
        # visible_coordination_actions below and into the scheduler/environment construction
        # further down) and, for communication history specifically,
        # communication_history_disabled passed straight into build_queue_negotiator below --
        # that flag doesn't swap prompt files, it makes QueueNegotiator itself say explicitly, in
        # the same base template, that no history will ever be available this subrun once the
        # count is 0 (see communication_history_disabled's own docstring on QueueNegotiator.__init__
        # for why the count alone can't tell a model that on its own).
        queue_prompt_path = resolve_agentic_repo_queue_prompt_path(
            args.queue_negotiation_prompts,
            disable_solution_sharing=not args.enable_solution_sharing,
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
            # "delegate_to_peer" deliberately excluded even when solution sharing is on -- see
            # --enable-solution-sharing's help text in parse_args() for why.
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
        # All three shared, run-wide, one instance for every worker thread -- see their class
        # docstrings above for why these can't be scoped to a single worker.
        schedule_board = AuctionScheduleBoard(logger=logger)
        budget_tracker = SubrunBudgetTracker()
        request_registry = SolverRequestRegistry()

        def _negotiation_memory_provider(repo_slug: str) -> str:
            base = _build_pb_auction_negotiation_context(
                assignment=assignment_by_slug[repo_slug],
                runtime_metadata=runtime_metadata,
                shared_token_pool=shared_token_pool,
                remaining_budget=budget_tracker.remaining(repo_slug),
            )
            requested_seconds = request_registry.get_requested_duration(repo_slug)
            if requested_seconds is not None:
                base = (
                    base
                    + f" The solver has requested an execution slot of approximately "
                    f"{requested_seconds:.0f}s this round."
                )
            prior_report_context = subrun_report_memory.context_for_next_subrun(repo_slug)
            combined = (base + " " + prior_report_context).strip() if prior_report_context else base
            return combined + " " + _PB_AUCTION_NEGOTIATION_INCENTIVE_TEXT

        scheduler = DockerTaskScheduler(
            max_concurrent_tasks=args.container_task_concurrency,
            high_priority_head_start=args.eval_high_priority_head_start,
            low_priority_max_wait_seconds=args.eval_low_priority_wait_seconds,
            communication_enabled=communication_enabled,
            communication_history_enabled=communication_history_enabled,
            # Fixed at 0 -- no bypass path at all in this design, see parse_args()'s comment.
            communication_bypass_cpu_threshold=0.0,
            high_priority_communication_wait_seconds=args.high_priority_communication_wait_seconds,
            low_priority_communication_wait_seconds=args.low_priority_communication_wait_seconds,
            communication_top_k=args.communication_top_k,
            communication_min_k=args.communication_min_k,
            negotiator=negotiator,
            # No cpu_percent_provider: with the bypass threshold permanently at 0, this would
            # never be consulted, so it's left unwired rather than paying its observation delay
            # for nothing.
            negotiation_memory_provider=_negotiation_memory_provider,
            allow_coordination_withdrawal=args.enable_solution_sharing,
            # Always False, independent of the flag above -- see --enable-solution-sharing's
            # help text: "delegate_to_peer" has no handler in this script.
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
        # No worker may start subrun N+1 until every other worker has finished subrun N -- keeps
        # the whole run proceeding in synchronized waves rather than letting a fast worker race
        # ahead into its own next subrun while a slower peer is still on an earlier one.
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
                raise ValueError(
                    f"{flag_name} has {len(values)} entries but there are {len(assignments)} "
                    f"assigned workers -- these must match exactly (one value per worker, in the "
                    f"order logged just above)."
                )
            return values

        worker_initial_budgets_override = _parse_per_worker_override(
            args.worker_initial_budgets, flag_name="--worker-initial-budgets"
        )
        worker_deadline_minutes_override = _parse_per_worker_override(
            args.worker_deadline_minutes, flag_name="--worker-deadline-minutes"
        )

        monitor = MonitorStateWriter(
            monitor_path,
            dataset="programbench", split="all", model=args.model,
            repo_count=args.worker_count, tasks_per_repo=args.tasks_per_worker,
            interval_seconds=args.monitor_interval_seconds,
            scheduler_snapshot_provider=scheduler.snapshot,
            mirror_paths=[Path(run_layout["latest_monitor_path"])],
            run_id=run_layout["run_id"], run_root=output_root,
        )

        shared_worker_deadline_minutes = compute_worker_deadline_minutes(
            ",".join(sorted(a.repo_slug for a in assignments))
        )
        worker_deadline_minutes = max(1, round(shared_worker_deadline_minutes * args.subrun_deadline_scale))

        # Per-worker deadline/budget overrides (ablation support): every assignment defaults to
        # the shared/uniform value computed above, unless --worker-deadline-minutes/
        # --worker-initial-budgets supplied an explicit per-worker list, matched positionally to
        # `assignments` order (assignment index 0 -> first value, etc.). Validated in
        # parse_args()'s caller below to be exactly len(assignments) long when present.
        per_worker_deadline_minutes: dict[str, int] = {
            assignment.repo_slug: worker_deadline_minutes for assignment in assignments
        }
        if worker_deadline_minutes_override is not None:
            for assignment, value in zip(assignments, worker_deadline_minutes_override):
                per_worker_deadline_minutes[assignment.repo_slug] = max(1, value)
        per_worker_initial_budget: dict[str, int] = {
            assignment.repo_slug: args.initial_budget for assignment in assignments
        }
        if worker_initial_budgets_override is not None:
            for assignment, value in zip(assignments, worker_initial_budgets_override):
                per_worker_initial_budget[assignment.repo_slug] = max(0, value)

        for assignment in assignments:
            resolved_deadline = per_worker_deadline_minutes[assignment.repo_slug]
            resolved_budget = per_worker_initial_budget[assignment.repo_slug]
            monitor.register_worker(assignment, deadline_minutes=resolved_deadline)
            logger.log(
                f"[{assignment.repo_name}] per-subrun deadline: {resolved_deadline} minute(s), "
                f"starting budget {resolved_budget} point(s) (fresh every subrun), visible to "
                "both solver and negotiator; execution priority decided by a sequential time-slot auction."
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
                    config, api, scheduler, per_worker_deadline_minutes[assignment.repo_slug],
                    runtime_metadata, report_runtime_metadata, monitor, shared_token_pool,
                    subrun_report_memory, report_prompt_config, report_model_name, report_model_kwargs,
                    negotiation_history_log_path, solver_memory_store, schedule_board, budget_tracker,
                    request_registry,
                    initial_budget=per_worker_initial_budget[assignment.repo_slug],
                    subrun_wave_gate=subrun_wave_gate,
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
