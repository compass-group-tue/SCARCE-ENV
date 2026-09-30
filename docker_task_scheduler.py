from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import os
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import Any, Callable, Optional, Protocol, TypeVar

from queue_negotiation import NegotiationCandidate, QueueNegotiator


class LoggerLike(Protocol):
    def log(self, message: str = "") -> None: ...


T = TypeVar("T")
CpuProvider = Callable[[], Optional[float]]
_DATACLASS_SLOTS_KW: dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **_DATACLASS_SLOTS_KW)
class ScheduledTaskContext:
    priority: str
    task_kind: str
    repo_name: str
    repo_slug: str
    instance_id: str | None = None
    description: str = ""
    deadline_minutes: int | None = None
    deadline_at_monotonic: float | None = None


class DeadlineClock:
    """Mutable, extendable wall-clock deadline.

    Used ONLY by run_swebench_subrun_repo_agents.py so that a repo's hard per-subrun
    deadline pauses while its current command is sitting in the scheduler's
    negotiation/communication phase, and only ticks down during real execution/
    thinking time. Every other caller in this codebase continues to pass a plain
    float wherever a deadline is expected, and every consumer normalizes with
    float(...) -- for a float that is a no-op, so this class has zero effect on
    existing callers.

    Thread-safety: guarded by its own private lock, independent of
    DockerTaskScheduler._condition, so it is safe to call .extend() while holding
    that condition (see the extend-on-admit hook in DockerTaskScheduler.run). The
    on_extend callback, if supplied, is invoked AFTER releasing this lock and must
    not call back into the scheduler (no .run/.register_deadline_clock/.snapshot
    calls), to avoid a lock-ordering cycle with DockerTaskScheduler._condition.
    """

    def __init__(
        self,
        deadline_at_monotonic: float,
        *,
        repo_slug: str,
        wall_time_ceiling_seconds: float,
        on_extend: Callable[[float, float, float], None] | None = None,
        on_tick: Callable[[float, float], None] | None = None,
    ) -> None:
        self._base_deadline_at_monotonic = float(deadline_at_monotonic)
        self.repo_slug = repo_slug
        self._wall_time_ceiling_seconds = max(1.0, float(wall_time_ceiling_seconds))
        self._extended_seconds = 0.0
        self._lock = Lock()
        self._on_extend = on_extend
        # Separate from on_extend on purpose: on_extend fires once per LOGGABLE
        # extension event (the original extend(), or announce_total_extension()'s
        # end-of-wait summary) -- deliberately NOT on every extend_silently() call,
        # to avoid one log line per communication-queue poll. But anything mirroring
        # this clock's value elsewhere (e.g. a live monitor/dashboard field) needs to
        # know about every real credit, silent ones included, or it stays stuck
        # showing a naive, still-decaying countdown throughout a long wait even though
        # the clock itself is correctly staying flat internally. on_tick exists purely
        # for that: called with (total_extended, new_deadline) on every actual credit,
        # with no logging side effect of its own.
        self._on_tick = on_tick

    @property
    def deadline_at_monotonic(self) -> float:
        with self._lock:
            return self._base_deadline_at_monotonic + self._extended_seconds

    def __float__(self) -> float:
        return self.deadline_at_monotonic

    def remaining_seconds(self, now: float | None = None) -> float:
        current = monotonic() if now is None else now
        return max(0.0, self.deadline_at_monotonic - current)

    def wall_time_ceiling_seconds(self) -> float:
        """Generous, but finite, ceiling for mini-swe-agent's own internal
        wall_time_limit_seconds, which is computed once at instance start and
        cannot itself be made pause-aware. Fixed at construction; does not grow
        with extensions."""
        return self._wall_time_ceiling_seconds

    def extend(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._extended_seconds += seconds
            total_extended = self._extended_seconds
            new_deadline = self._base_deadline_at_monotonic + total_extended
        if self._on_extend is not None:
            self._on_extend(seconds, total_extended, new_deadline)
        if self._on_tick is not None:
            self._on_tick(total_extended, new_deadline)

    def extend_silently(self, seconds: float) -> None:
        """Same effect as extend() but never fires on_extend -- on_tick still fires.

        Used to credit small increments *while a wait is still in progress* (once per
        communication-queue poll, not just once at the end) so remaining_seconds(),
        read live by anyone watching, never actually decays during the wait -- it used
        to only be credited retroactively, in one lump sum, after the whole wait ended,
        which is correct for the final total but means a long wait (a min-k stall, a
        slow peer) visibly ticks down in real time for the whole time it's in progress.
        Pair with announce_total_extension() once the wait concludes to still log the
        entire span as a single summary line, matching prior behavior exactly. on_tick
        (if supplied) fires here too -- it exists precisely to keep something like a
        live monitor field in sync on every credit, not just the final logged one.
        """
        if seconds <= 0:
            return
        with self._lock:
            self._extended_seconds += seconds
            total_extended = self._extended_seconds
            new_deadline = self._base_deadline_at_monotonic + total_extended
        if self._on_tick is not None:
            self._on_tick(total_extended, new_deadline)

    def announce_total_extension(self, total_seconds: float) -> None:
        """Fire on_extend for `total_seconds` WITHOUT re-extending the clock.

        Call once, after a sequence of extend_silently() calls whose increments already
        sum to total_seconds, to produce the usual single "deadline paused for Ns"
        summary line instead of one log line per poll interval.
        """
        if total_seconds <= 0 or self._on_extend is None:
            return
        with self._lock:
            total_extended = self._extended_seconds
            new_deadline = self._base_deadline_at_monotonic + total_extended
        self._on_extend(total_seconds, total_extended, new_deadline)


@dataclass(**_DATACLASS_SLOTS_KW)
class _QueuedTask:
    task_id: int
    context: ScheduledTaskContext
    announced_at: float
    execution_queued_at: float | None
    enqueued_at: float
    completed_high_count_at_enqueue: int
    communication_bypassed: bool = False
    communication_bypass_reason: str = ""


@dataclass(**_DATACLASS_SLOTS_KW)
class _ActiveTask:
    task_id: int
    context: ScheduledTaskContext
    started_at: float


@dataclass(**_DATACLASS_SLOTS_KW)
class _CompletedTask:
    task_id: int
    context: ScheduledTaskContext
    queued_at: float
    started_at: float
    finished_at: float


@dataclass(**_DATACLASS_SLOTS_KW)
class _ExecutionRecord:
    task_id: int
    context: ScheduledTaskContext
    queued_at: float
    started_at: float
    finished_at: float


class TaskAbandoned(Exception):
    """Raised from run() when abandon_repo_pending_communication() marks this task's own repo as
    abandoned while it was still waiting in communication (not yet admitted into the execution
    queue).

    Exists for exactly one scenario: a harness-driven worker's request_execution_slot call runs
    on its own background thread (see HarnessMCPServer._request_execution_slot_impl), separate
    from the harness subprocess itself. If that subprocess exits (finishes, crashes, gets killed)
    while its background thread is still blocked here waiting for admission, nothing previously
    stopped that thread or removed its announced task from self._queue -- it became a permanent
    ghost candidate, sitting in every future negotiation round's pool forever, since nothing was
    ever left alive to consume a grant even if one eventually arrived. Confirmed live: a run
    where a harness process died mid-poll left two (then three) such orphaned tasks
    accumulating, each showing up as an extra, duplicate "candidate" for the same repo in later
    rounds' snapshots.

    The caller-side fix is orchestrator code calling abandon_repo_pending_communication(repo_slug)
    once a worker's harness session has ended, from the same finally block that already tears
    down that worker's MCP server -- see run_programbench_auction_subrun_harness_agents.py. This
    exception is what makes that call actually unstick the orphaned background thread instead of
    only preventing the ghost from being included in FUTURE announcements: run()'s own busy-wait
    loop notices the abandonment on its next wakeup, cleans up self._queue itself, and raises this
    so the (now-orphaned, but still-alive) caller thread returns instead of blocking forever."""

    pass


@dataclass(frozen=True, **_DATACLASS_SLOTS_KW)
class SchedulerCoordinationWithdrawal(Exception):
    task_id: int
    repo_name: str
    repo_slug: str
    instance_id: str | None
    peer_task_id: int | None
    peer_repo_name: str | None
    peer_repo_slug: str | None
    reason: str
    shared_context: str
    # True for a genuine delegate-to-peer relationship (a persistent _DelegatedWait exists or is
    # being created for this repo): the current instance should be parked via
    # SchedulerExecutionDeferred rather than nudged with a synthetic tool message. False for the
    # softer, one-off wait_for_peer nudge, which keeps grinding on the same instance.
    defer_execution: bool = False


@dataclass(frozen=True, **_DATACLASS_SLOTS_KW)
class SchedulerExecutionDeferred(Exception):
    repo_name: str
    repo_slug: str
    instance_id: str | None
    peer_task_id: int | None
    peer_repo_name: str | None
    peer_repo_slug: str | None
    reason: str
    shared_context: str


@dataclass(frozen=True, **_DATACLASS_SLOTS_KW)
class _WithdrawalRecord:
    task_id: int
    context: ScheduledTaskContext
    withdrawn_at: float
    peer_task_id: int | None
    peer_repo_name: str | None
    peer_repo_slug: str | None
    reason: str
    shared_context: str


@dataclass(**_DATACLASS_SLOTS_KW)
class _DelegatedWait:
    waiter_repo_name: str
    waiter_repo_slug: str
    peer_task_id: int | None
    peer_repo_name: str | None
    peer_repo_slug: str | None
    reason: str
    expected_handoff: str
    created_at: float


@dataclass(**_DATACLASS_SLOTS_KW)
class _DeliveredDelegation:
    waiter_repo_name: str
    waiter_repo_slug: str
    peer_task_id: int | None
    peer_repo_name: str | None
    peer_repo_slug: str | None
    handoff_text: str
    delivered_at: float


class DockerTaskScheduler:
    """Serialize container-side work with negotiation before queue admission."""

    _NEGOTIATION_HISTORY_MAXLEN = 100
    # Ceiling on how long run()'s communication-queue wait ever blocks between wakeups,
    # even when _communication_wait_timeout_locked() says there's no bound (e.g. min-k
    # not yet satisfied). Two things need this: (1) live, incremental deadline-clock
    # crediting (see run()) needs periodic wakeups to have anything to do, and (2) an
    # unbounded wait with nothing left to notify it -- e.g. every other worker has
    # permanently finished -- would otherwise block forever with zero chance to
    # re-evaluate. Small enough that the live deadline view stays visually fresh;
    # large enough that the extra lock/condition-check overhead is negligible.
    _COMMUNICATION_POLL_SECONDS = 5.0
    # Defensive backstop, not a normal-path timer: if a task has been sitting un-negotiated
    # this long, something has gone wrong with whatever was supposed to re-arm
    # _communication_dirty (a new announce, a full-batch drain, ...) -- the richest real
    # negotiation rounds observed in practice run well under this, so a legitimate round in
    # progress should never trip it. Forcing dirty=True here just makes the scheduler
    # re-evaluate; it is always safe (worst case, the re-check finds nothing has actually
    # changed and no-ops), so there is no failure mode where firing this makes things worse.
    _COMMUNICATION_STALL_WATCHDOG_SECONDS = 1800.0
    # Hard wall-clock ceiling on a SINGLE negotiate() call, not a normal-path timer either: a
    # genuinely healthy round (even a slow one, with real per-phase LLM retries) should resolve
    # well under this. Added after a live run showed a 3-genuine-candidate round (confirmed via
    # negotiation log inspection -- NOT a min-k-relaxation bug, which was the first, incorrect
    # diagnosis) spin for 35+ minutes straight on repeated malformed-JSON responses across every
    # phase, for every candidate, with nothing to force it to resolve. See
    # _ensure_negotiation_state_locked's own handling: on timeout, this round falls back to the
    # SAME deterministic admission order used for a real negotiation exception (existing code,
    # unchanged), so a badly-behaving negotiator degrades to "everyone gets admitted in priority/
    # deadline order" instead of blocking the pool indefinitely. The abandoned call itself keeps
    # running in the background (same posture as QueueNegotiator's own
    # _call_completion_with_hard_timeout) -- its eventual result is simply never read.
    _NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS = float(os.environ.get("NEGOTIATION_ROUND_TIMEOUT_SECONDS", "600"))
    # Backstop on top of the backstop above: if the background thread abandoned by the timeout
    # above never actually finishes (a genuine hang, not just a slow-but-alive call), the
    # existing code deliberately leaves _negotiation_in_progress stuck True forever -- see its
    # own comment for why (avoiding two negotiate() calls racing on the negotiator's own shared
    # state). Confirmed live: exactly this happened on a real run -- one round's negotiate() call
    # hit the wall-clock timeout and was abandoned, and no further round ever resolved again for
    # the rest of that run, even though _wait_for_turn's own separate backstop kept individual
    # workers from hanging forever on the resulting dead queue. Traced every self.-attribute
    # QueueNegotiator.negotiate() mutates outside __init__: only
    # _cost_of_delay_multiplier_by_task_id (a whole-dict reassignment every call, and the
    # abandoned call's own result is already discarded by the caller regardless of what it reads)
    # and _compacted_history_cache (a per-repo_slug optimization cache that already degrades
    # gracefully to full-text on any failure). Neither is correctness-critical to race against,
    # so it's an acceptable, bounded risk to force-clear the flag and let a fresh attempt start
    # once the abandoned call has ALSO blown well past its own wall-clock budget again -- set to
    # double that budget so this never fires for a call that's merely slow, only one that's
    # provably still stuck long after it should have finished one way or another.
    _NEGOTIATION_IN_PROGRESS_STUCK_BACKSTOP_SECONDS = _NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS * 2
    # Modest, unbiased default granted to every candidate when negotiation fails for ANY reason
    # (timeout above, or a real exception) and falls back to deterministic ordering. Added after
    # a live run showed this fallback compounding rather than resolving the underlying problem:
    # without this, EVERY candidate's final_duration/final_bid_price stayed unset, so
    # _extract_latest_round_schedule (in run_programbench_auction_subrun_repo_agents.py) found
    # nothing parseable for any of them and told every worker "not reflected in this round's
    # resolved schedule -- call request_execution_slot again," forcing a BRAND NEW full
    # negotiation round from scratch -- which, if the underlying negotiator instability was
    # persistent rather than a one-off blip, had a good chance of ALSO timing out, repeating
    # indefinitely instead of ever converging. Deliberately small and the same for everyone (not
    # an attempt to guess what real bidding would have produced) -- this is what "negotiation
    # genuinely could not resolve, degrade to make forward progress possible" should look like:
    # a real, small, safely-capped-downstream window, not a repeated dead end.
    _FALLBACK_GRANT_DURATION_SECONDS = 60.0
    _FALLBACK_BID_PRICE = 0

    def __init__(
        self,
        *,
        max_concurrent_tasks: int = 1,
        high_priority_head_start: int = 2,
        low_priority_max_wait_seconds: float = 120.0,
        communication_enabled: bool = True,
        communication_history_enabled: bool = True,
        communication_bypass_cpu_threshold: float = 80.0,
        high_priority_communication_wait_seconds: float = 30.0,
        low_priority_communication_wait_seconds: float = 60.0,
        communication_top_k: int = 5,
        communication_min_k: int = 0,
        negotiator: QueueNegotiator | None = None,
        cpu_percent_provider: CpuProvider | None = None,
        negotiation_memory_provider: Callable[[str], str] | None = None,
        allow_coordination_withdrawal: bool = False,
        enable_delegated_peer_wait: bool = False,
        waive_wait_after_contested_negotiation: bool = False,
        visible_worker_label: str = "peer",
        visible_update_label: str = "handoff",
        logger: LoggerLike | None = None,
        negotiation_history_log_path: Path | str | None = None,
    ) -> None:
        if max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be >= 1")
        if high_priority_head_start < 0:
            raise ValueError("high_priority_head_start must be >= 0")
        if communication_top_k < 1:
            raise ValueError("communication_top_k must be >= 1")
        if communication_min_k < 0:
            raise ValueError("communication_min_k must be >= 0")
        if communication_min_k > 0 and communication_min_k > communication_top_k:
            raise ValueError("communication_min_k cannot exceed communication_top_k")
        self.max_concurrent_tasks = max_concurrent_tasks
        self.high_priority_head_start = high_priority_head_start
        self.low_priority_max_wait_seconds = low_priority_max_wait_seconds
        self.communication_enabled = communication_enabled
        self.communication_history_enabled = communication_history_enabled
        self.communication_bypass_cpu_threshold = communication_bypass_cpu_threshold
        self.high_priority_communication_wait_seconds = high_priority_communication_wait_seconds
        self.low_priority_communication_wait_seconds = low_priority_communication_wait_seconds
        self.communication_top_k = communication_top_k
        self.communication_min_k = communication_min_k
        self._negotiator = negotiator
        self._cpu_percent_provider = cpu_percent_provider
        self._negotiation_memory_provider = negotiation_memory_provider
        self.allow_coordination_withdrawal = allow_coordination_withdrawal
        self.enable_delegated_peer_wait = enable_delegated_peer_wait
        self.waive_wait_after_contested_negotiation = waive_wait_after_contested_negotiation
        self.visible_worker_label = str(visible_worker_label or "peer")
        self.visible_update_label = str(visible_update_label or "handoff")
        self._logger = logger
        self._condition = Condition()
        self._queue: deque[_QueuedTask] = deque()
        self._next_task_id = 1
        self._active_tasks = 0
        self._active_high_tasks = 0
        self._completed_high_tasks = 0
        self._active_task_details: dict[int, _ActiveTask] = {}
        self._recent_completed: deque[_CompletedTask] = deque(maxlen=24)
        self._communication_dirty = True
        self._negotiation_in_progress = False
        # Set alongside _negotiation_in_progress = True (below), cleared alongside it back to
        # None -- lets _ensure_negotiation_state_locked's own early-return tell "a healthy
        # negotiate() call is genuinely still running" apart from "this has been stuck long
        # enough that the call must have actually hung." See
        # _NEGOTIATION_IN_PROGRESS_STUCK_BACKSTOP_SECONDS's own comment for the incident this
        # closes.
        self._negotiation_in_progress_since: float | None = None
        self._watchdog_warned_task_ids: set[int] = set()
        self._negotiated_task_ids: list[int] = []
        self._negotiation_message_by_task_id: dict[int, str] = {}
        self._negotiation_rationale_by_task_id: dict[int, str] = {}
        self._negotiation_rank_by_task_id: dict[int, int] = {}
        self._declaration_by_task_id: dict[int, str] = {}
        self._declaration_reasoning_summary_by_task_id: dict[int, str] = {}
        self._reasoning_by_task_id: dict[int, str] = {}
        self._reasoning_reasoning_summary_by_task_id: dict[int, str] = {}
        self._discussion_by_task_id: dict[int, str] = {}
        self._ballot_by_task_id: dict[int, str] = {}
        self._ballot_reasoning_summary_by_task_id: dict[int, str] = {}
        self._coordination_decision_by_task_id: dict[int, dict[str, Any]] = {}
        # Previously only ever created as a side effect of the unconditional reset at the top of
        # _ensure_negotiation_state_locked (now removed -- these are merged via .update() instead
        # of reset-then-reassigned, see that method's own comment) -- declared here explicitly so
        # they exist before that method's first call regardless of which branch it takes.
        self._final_duration_by_task_id: dict[int, float] = {}
        self._final_price_by_task_id: dict[int, int] = {}
        self._latest_negotiation_transcript: list[dict[str, Any]] = []
        self._negotiation_history: list[dict[str, Any]] = []
        # Every round is also appended here, in full, with no cap -- unlike
        # self._negotiation_history (bounded to _NEGOTIATION_HISTORY_MAXLEN so that the
        # frequently-rewritten monitor_state.json doesn't grow unbounded and slow down the
        # periodic monitor write). A round can be appended more than once as it's updated
        # (declaration/reasoning/discussion/ballot progress, later execution-outcome refreshes);
        # readers should keep only the last line per round_id. Append-only writes are O(1) per
        # call, so this doesn't reintroduce the rewrite-cost problem the cap was built to avoid.
        self._negotiation_history_log_fh = None
        if negotiation_history_log_path is not None:
            log_path = Path(negotiation_history_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._negotiation_history_log_fh = log_path.open("a", encoding="utf-8")
        self._negotiation_round_counter = 0
        self._execution_record_by_task_id: dict[int, _ExecutionRecord] = {}
        self._withdrawal_record_by_task_id: dict[int, _WithdrawalRecord] = {}
        self._delegated_wait_by_repo_slug: dict[str, _DelegatedWait] = {}
        self._delivered_delegation_by_repo_slug: dict[str, _DeliveredDelegation] = {}
        self._deadline_clocks_by_repo_slug: dict[str, DeadlineClock] = {}
        # Empty unless a caller opts in via register_worker(). While empty,
        # _communication_min_k_ready_locked's behavior is identical to before this
        # existed -- purely additive, zero risk to any caller that doesn't use it.
        self._registered_worker_repo_slugs: set[str] = set()
        # Populated only by abandon_repo_pending_communication() -- see TaskAbandoned's own
        # docstring for what this is for. Checked once per wakeup inside run()'s own communication
        # busy-wait loop, then immediately discarded from here (the task itself gets removed from
        # self._queue at the same time), so this set only ever holds task_ids genuinely still
        # blocked in that loop at this exact moment, never a growing history.
        self._abandoned_task_ids: set[int] = set()

    def register_deadline_clock(self, repo_slug: str, clock: "DeadlineClock | None") -> None:
        """Attach (or clear, if clock is None) the pause-aware deadline clock for a
        repo's current subrun. Only run_swebench_subrun_repo_agents.py calls this;
        the registry stays empty for every other caller, so the extend-on-admit
        hook in run() below is a guaranteed no-op for them."""
        with self._condition:
            if clock is None:
                self._deadline_clocks_by_repo_slug.pop(repo_slug, None)
            else:
                self._deadline_clocks_by_repo_slug[repo_slug] = clock

    def register_worker(self, repo_slug: str) -> None:
        """Mark repo_slug as capable of announcing a future task -- called once, before
        a worker's subrun loop starts. Together with unregister_worker(), this lets
        _communication_min_k_ready_locked cap the required candidate count at however
        many workers can still possibly show up, instead of demanding a fixed
        communication_min_k that can become permanently unreachable once enough
        workers finish their whole assigned batch and exit for good (observed live: a
        lone remaining worker waited forever for two peers who had already finished
        and would never announce again -- min-k=3 with only 1 worker left can never be
        satisfied). A caller that never calls this at all sees no behavior change --
        the effective-min-k logic only activates once at least one worker is
        registered.
        """
        with self._condition:
            self._registered_worker_repo_slugs.add(repo_slug)

    def unregister_worker(self, repo_slug: str) -> None:
        """Mark repo_slug as permanently done -- it will never announce again. Call
        once a worker's whole subrun loop (every assigned task) has finished, in a
        finally block so it fires on every exit path. Wakes any blocked communication
        wait so a stalled peer immediately re-evaluates against the smaller pool
        instead of waiting for its next scheduled poll.

        Also marks the pool dirty (see _communication_dirty's own field docstring for what
        this flag gates): removing a registered worker can change what
        _effective_communication_min_k_locked() returns (it caps communication_min_k at how
        many workers could still possibly announce), which can flip a still-waiting
        candidate's own admission from "blocked, not enough peers" to "ready, admit now" --
        e.g. a lone remaining worker whose only competitors just finished should immediately
        resolve as a solo pool. Without this, nothing re-arms a fresh check: the departing
        worker's own last task is typically already resolved by this point, so it doesn't
        independently set dirty, and a peer blocked in the communication-wait loop (run(),
        lines ~546-646) keeps polling against the OLD (larger) effective min-k until either a
        genuinely new task happens to announce (setting dirty for an unrelated reason) or the
        _COMMUNICATION_STALL_WATCHDOG_SECONDS anomaly-detector eventually notices and forces a
        re-check. Confirmed live: a run's last-remaining worker sat fully idle -- zero real
        negotiation activity, deadline correctly paused the whole time so no budget was lost,
        but zero actual progress either -- for the watchdog's entire ~1800s threshold before
        self-healing, even though the moment its last peer unregistered it should have gone
        solo and been admitted within a poll cycle. Setting dirty here closes that gap at the
        source instead of relying on the watchdog to eventually paper over it.
        """
        with self._condition:
            self._registered_worker_repo_slugs.discard(repo_slug)
            self._communication_dirty = True
            self._condition.notify_all()

    def abandon_repo_pending_communication(self, repo_slug: str) -> None:
        """Mark any of repo_slug's own tasks still waiting in communication (announced, but not
        yet admitted into the execution queue) as abandoned. See TaskAbandoned's own docstring
        for the full scenario this exists for: a task whose owning run() call is blocked on a
        background thread whose harness subprocess has already exited, which would otherwise sit
        as a permanent ghost candidate in every future negotiation round.

        Call this once a worker's harness session has genuinely ended (the harness subprocess
        has exited, one way or another), in the same finally block that already tears down that
        worker's MCP server -- before unregister_worker(), not instead of it: this cleans up an
        already-announced, in-flight task; unregister_worker() is about future min-k
        computation, an orthogonal concern.

        Safe to call even if repo_slug has no pending communication task at all (the common,
        non-broken case) -- this is then a complete no-op, matching every other caller of this
        scheduler that never uses this method at all.
        """
        with self._condition:
            for task in self._queue:
                if task.context.repo_slug == repo_slug and task.execution_queued_at is None:
                    self._abandoned_task_ids.add(task.task_id)
            self._condition.notify_all()

    def run(self, func: Callable[[], T], *, context: ScheduledTaskContext) -> T:
        with self._condition:
            preexisting_withdrawal = None
            if context.task_kind != "evaluation":
                preexisting_withdrawal = self._delegation_withdrawal_for_repo_locked(context.repo_slug)
            if preexisting_withdrawal is not None:
                raise preexisting_withdrawal
        task = self._announce(context)
        with self._condition:
            # Tracks how much of this wait has already been silently credited to the
            # deadline clock, live, as the loop below polls -- see the comment further
            # down for why this exists. Initialized to announced_at so the bypass case
            # (loop never runs at all) computes a correct full-span credit unchanged
            # from before this existed.
            last_credited_at = task.announced_at
            if self.communication_enabled and task.execution_queued_at is None:
                clock = self._deadline_clocks_by_repo_slug.get(context.repo_slug)
                while True:
                    if task.task_id in self._abandoned_task_ids:
                        # See TaskAbandoned's own docstring: this task's owning repo was marked
                        # abandoned (its harness subprocess has already exited) by
                        # abandon_repo_pending_communication(), called from a DIFFERENT thread
                        # than the one blocked here. Clean up and stop waiting -- nothing is
                        # left alive to ever use a grant even if this task were eventually
                        # admitted.
                        self._abandoned_task_ids.discard(task.task_id)
                        if task in self._queue:
                            self._queue.remove(task)
                        self._communication_dirty = True
                        self._condition.notify_all()
                        self._log(
                            "scheduler abandoned "
                            f"{context.priority} {context.task_kind} for {context.repo_name}"
                            f"{self._format_instance_suffix(context.instance_id)} -- its owning "
                            "harness session already ended while this request was still waiting "
                            "in communication."
                        )
                        raise TaskAbandoned(
                            f"{context.repo_name}'s harness session ended while this request "
                            "was still waiting in communication."
                        )
                    withdrawal = self._coordination_withdrawal_for_task_locked(task)
                    if withdrawal is not None:
                        self._queue.remove(task)
                        self._withdrawal_record_by_task_id[task.task_id] = _WithdrawalRecord(
                            task_id=task.task_id,
                            context=task.context,
                            withdrawn_at=monotonic(),
                            peer_task_id=withdrawal.peer_task_id,
                            peer_repo_name=withdrawal.peer_repo_name,
                            peer_repo_slug=withdrawal.peer_repo_slug,
                            reason=withdrawal.reason,
                            shared_context=withdrawal.shared_context,
                        )
                        self._communication_dirty = True
                        self._refresh_execution_outcomes_locked(task.task_id)
                        self._condition.notify_all()
                        self._log(
                            "scheduler withdrew "
                            f"{context.priority} {context.task_kind} for {context.repo_name}"
                            f"{self._format_instance_suffix(context.instance_id)} from communication "
                            f"so it can re-plan from a {self._visible_worker_phrase()} clue"
                            + (f" shared by {withdrawal.peer_repo_name}" if withdrawal.peer_repo_name else "")
                            + "."
                        )
                        raise withdrawal
                    if self._communication_can_admit_locked(task):
                        break
                    # Bounded even when _communication_wait_timeout_locked returns None
                    # (unbounded -- e.g. min-k not yet satisfied): without this cap, a
                    # wait with nothing left to notify it (observed live -- a lone
                    # remaining worker waiting on peers who had already finished for
                    # good) blocks forever with zero wakeups, so neither this
                    # incremental credit nor a fresh admission re-check ever runs.
                    # Polling this often is cheap (a lock acquire and a few
                    # comparisons) relative to how rarely it actually needs to matter.
                    raw_timeout = self._communication_wait_timeout_locked(task)
                    poll_timeout = (
                        self._COMMUNICATION_POLL_SECONDS
                        if raw_timeout is None
                        else min(raw_timeout, self._COMMUNICATION_POLL_SECONDS)
                    )
                    self._condition.wait(poll_timeout)
                    # Credit elapsed wait time to the deadline clock incrementally, on
                    # every wakeup, so remaining_seconds() read live by anything
                    # watching never actually decays while waiting on communication --
                    # it used to only be credited retroactively, in one lump sum, once
                    # the whole wait ended, which was correct for the eventual total
                    # but meant a long wait was visibly ticking down the entire time it
                    # was in progress. extend_silently() doesn't fire the "deadline
                    # paused" log line -- that still happens once, below, for the full
                    # span, exactly as before.
                    if clock is not None:
                        now_tick = monotonic()
                        elapsed = max(0.0, now_tick - last_credited_at)
                        if elapsed > 0:
                            clock.extend_silently(elapsed)
                            last_credited_at = now_tick
                    if (
                        not self._communication_dirty
                        and not self._negotiation_in_progress
                        and monotonic() - task.announced_at > self._COMMUNICATION_STALL_WATCHDOG_SECONDS
                        and task.task_id not in self._watchdog_warned_task_ids
                    ):
                        self._watchdog_warned_task_ids.add(task.task_id)
                        self._communication_dirty = True
                        self._log(
                            "scheduler watchdog: "
                            f"{context.priority} {context.task_kind} for {context.repo_name}"
                            f"{self._format_instance_suffix(context.instance_id)} has waited "
                            f"{monotonic() - task.announced_at:.0f}s un-negotiated with no negotiation "
                            "in progress -- forcing a fresh negotiation re-check. This should never "
                            "fire during a genuine (even slow) negotiation; if it does, whatever was "
                            "supposed to re-arm negotiation for this pool did not."
                        )
                        self._condition.notify_all()
            task.execution_queued_at = task.execution_queued_at or monotonic()
            # Fold this task's negotiation-phase wait into its repo's deadline clock,
            # if one is registered (only true under run_swebench_subrun_repo_agents.py
            # and this project's auction variant). For a bypassed task,
            # execution_queued_at was already set at announce time, so this naturally
            # computes to ~0 -- no special-casing needed.
            negotiation_duration = max(0.0, task.execution_queued_at - task.announced_at)
            if negotiation_duration > 0:
                clock = self._deadline_clocks_by_repo_slug.get(context.repo_slug)
                if clock is not None:
                    # Credit whatever tail hasn't been silently applied yet (usually a
                    # fraction of one poll interval), then announce the FULL span as a
                    # single summary line -- matching the exact log output this
                    # produced before live incremental crediting existed, even though
                    # most of the span was already applied above.
                    remaining_to_credit = max(0.0, task.execution_queued_at - last_credited_at)
                    if remaining_to_credit > 0:
                        clock.extend_silently(remaining_to_credit)
                    clock.announce_total_extension(negotiation_duration)
            task.enqueued_at = task.execution_queued_at
            task.completed_high_count_at_enqueue = self._completed_high_tasks
            still_waiting_ids = {item.task_id for item in self._queue if item.execution_queued_at is None}
            remaining_batch_members = still_waiting_ids & set(self._negotiated_task_ids)
            if not remaining_batch_members:
                # Only force a fresh negotiation once this whole decided batch has
                # drained into the execution queue; otherwise the rest of the group
                # keeps admitting against the order it already agreed on.
                self._communication_dirty = True
            self._condition.notify_all()
        self._log(
            "scheduler admitted "
            f"{context.priority} {context.task_kind} for {context.repo_name}"
            f"{self._format_instance_suffix(context.instance_id)} into the task queue"
            f"{self._format_description_suffix(context.description)}"
        )
        start_wait = monotonic()
        with self._condition:
            while True:
                next_task = self._next_eligible_task_locked()
                if next_task is not None and next_task.task_id == task.task_id and self._active_tasks < self.max_concurrent_tasks:
                    self._queue.remove(task)
                    self._active_tasks += 1
                    if self._is_high(task):
                        self._active_high_tasks += 1
                    started_at = monotonic()
                    self._active_task_details[task.task_id] = _ActiveTask(
                        task_id=task.task_id,
                        context=task.context,
                        started_at=started_at,
                    )
                    break
                wait_timeout = self._next_wait_timeout_locked(task)
                self._condition.wait(wait_timeout)

        started_at = self._active_task_details[task.task_id].started_at
        waited_for = started_at - start_wait
        self._log(
            "scheduler dispatching "
            f"{context.priority} {context.task_kind} for {context.repo_name}"
            f"{self._format_instance_suffix(context.instance_id)} after waiting {waited_for:.1f}s"
            f"{self._format_description_suffix(context.description)}"
        )
        try:
            return func()
        finally:
            with self._condition:
                self._active_tasks -= 1
                if self._is_high(task):
                    self._active_high_tasks -= 1
                    self._completed_high_tasks += 1
                active = self._active_task_details.pop(task.task_id, None)
                if active is not None:
                    finished_at = monotonic()
                    execution_record = _ExecutionRecord(
                        task_id=task.task_id,
                        context=task.context,
                        queued_at=task.enqueued_at,
                        started_at=active.started_at,
                        finished_at=finished_at,
                    )
                    self._execution_record_by_task_id[task.task_id] = execution_record
                    self._recent_completed.append(
                        _CompletedTask(
                            task_id=task.task_id,
                            context=task.context,
                            queued_at=task.enqueued_at,
                            started_at=active.started_at,
                            finished_at=finished_at,
                        )
                    )
                    self._refresh_execution_outcomes_locked(task.task_id)
                self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            now = monotonic()
            self._ensure_negotiation_state_locked(now)
            communication = [
                {
                    "task_id": task.task_id,
                    "priority": task.context.priority,
                    "task_kind": task.context.task_kind,
                    "repo_name": task.context.repo_name,
                    "repo_slug": task.context.repo_slug,
                    "instance_id": task.context.instance_id,
                    "description": task.context.description,
                    "deadline_minutes": task.context.deadline_minutes,
                    "deadline_remaining_seconds": round(self._deadline_remaining_seconds(task, now), 2),
                    "announced_for_seconds": round(max(0.0, now - task.announced_at), 2),
                    "communication_wait_seconds": self._communication_wait_seconds_for_priority(task.context.priority),
                    "ready_for_execution_queue": self._communication_ready_locked(task, now),
                    **self._communication_explanation_locked(task, now),
                }
                for task in self._queue
                if task.execution_queued_at is None
            ]
            queue = [
                {
                    "task_id": task.task_id,
                    "priority": task.context.priority,
                    "task_kind": task.context.task_kind,
                    "repo_name": task.context.repo_name,
                    "repo_slug": task.context.repo_slug,
                    "instance_id": task.context.instance_id,
                    "description": task.context.description,
                    "deadline_minutes": task.context.deadline_minutes,
                    "deadline_remaining_seconds": round(self._deadline_remaining_seconds(task, now), 2),
                    "queued_for_seconds": round(max(0.0, now - task.enqueued_at), 2),
                    "high_priority_completed_since_enqueue": self._completed_high_tasks - task.completed_high_count_at_enqueue,
                    "eligible": self._is_high(task) or self._low_priority_task_is_ready_locked(task),
                    **self._execution_explanation_locked(task, now),
                }
                for task in self._queue
                if task.execution_queued_at is not None
            ]
            active = [
                {
                    "task_id": item.task_id,
                    "priority": item.context.priority,
                    "task_kind": item.context.task_kind,
                    "repo_name": item.context.repo_name,
                    "repo_slug": item.context.repo_slug,
                    "instance_id": item.context.instance_id,
                    "description": item.context.description,
                    "deadline_minutes": item.context.deadline_minutes,
                    "deadline_remaining_seconds": round(self._deadline_remaining_seconds_from_context(item.context, now), 2),
                    "running_for_seconds": round(max(0.0, now - item.started_at), 2),
                }
                for item in self._active_task_details.values()
            ]
            recent_completed = [
                {
                    "task_id": item.task_id,
                    "priority": item.context.priority,
                    "task_kind": item.context.task_kind,
                    "repo_name": item.context.repo_name,
                    "repo_slug": item.context.repo_slug,
                    "instance_id": item.context.instance_id,
                    "description": item.context.description,
                    "deadline_minutes": item.context.deadline_minutes,
                    "deadline_remaining_seconds": round(self._deadline_remaining_seconds_from_context(item.context, now), 2),
                    "queued_for_seconds": round(max(0.0, item.started_at - item.queued_at), 2),
                    "running_for_seconds": round(max(0.0, item.finished_at - item.started_at), 2),
                }
                for item in reversed(self._recent_completed)
            ]
            return {
                "summary": {
                    "communication_enabled": self.communication_enabled,
                    "communication_history_enabled": self.communication_history_enabled,
                    "communication_bypass_cpu_threshold": self.communication_bypass_cpu_threshold,
                    "communication_tasks": len(communication),
                    "queued_tasks": len(queue),
                    "active_tasks": len(active),
                    "max_concurrent_tasks": self.max_concurrent_tasks,
                    "high_priority_head_start": self.high_priority_head_start,
                    "low_priority_max_wait_seconds": self.low_priority_max_wait_seconds,
                    "high_priority_communication_wait_seconds": self.high_priority_communication_wait_seconds,
                    "low_priority_communication_wait_seconds": self.low_priority_communication_wait_seconds,
                    "communication_top_k": self.communication_top_k,
                    "communication_min_k": self.communication_min_k,
                    "completed_high_tasks": self._completed_high_tasks,
                },
                "communication": communication,
                "active": active,
                "queued": queue,
                "recent_completed": recent_completed,
                "active_negotiation": self._active_negotiation_snapshot_locked(now),
                # Deep copy, not list(self._negotiation_history) (a shallow copy that still shares
                # every entry's own dict/list objects with the live scheduler). A caller that
                # serializes this snapshot (e.g. json.dumps for monitor_state.json) AFTER releasing
                # self._condition could otherwise observe a torn, self-inconsistent view of a round
                # entry if _refresh_execution_outcomes_locked or _record_negotiation_history_locked
                # mutates that same entry's fields in place concurrently (both update several keys
                # of the same dict as separate statements, not by replacing the whole dict
                # atomically). Entries are capped at _NEGOTIATION_HISTORY_MAXLEN, so this stays cheap.
                "communication_history": copy.deepcopy(self._negotiation_history),
            }

    def runtime_status_snapshot(self, repo_slug: str | None = None) -> dict[str, Any]:
        with self._condition:
            now = monotonic()
            communication = [
                task
                for task in self._queue
                if task.execution_queued_at is None
            ]
            queued = [
                task
                for task in self._queue
                if task.execution_queued_at is not None
            ]
            active = list(self._active_task_details.values())
            focus_task = next(
                (task for task in communication if task.context.repo_slug == repo_slug),
                None,
            )
            if focus_task is None:
                focus_task = next(
                    (task for task in queued if task.context.repo_slug == repo_slug),
                    None,
                )
            focus_active = None
            if repo_slug is not None:
                focus_active = next(
                    (item for item in active if item.context.repo_slug == repo_slug),
                    None,
                )
            return {
                "summary": {
                    "communication_tasks": len(communication),
                    "queued_tasks": len(queued),
                    "active_tasks": len(active),
                    "max_concurrent_tasks": self.max_concurrent_tasks,
                    "completed_high_tasks": self._completed_high_tasks,
                    "communication_top_k": self.communication_top_k,
                    "communication_min_k": self.communication_min_k,
                },
                "focus": {
                    "repo_slug": repo_slug,
                    "in_communication": focus_task is not None and focus_task.execution_queued_at is None,
                    "in_execution_queue": focus_task is not None and focus_task.execution_queued_at is not None,
                    "is_active": focus_active is not None,
                    "task_id": (
                        focus_task.task_id
                        if focus_task is not None
                        else (focus_active.task_id if focus_active is not None else None)
                    ),
                    "priority": (
                        focus_task.context.priority
                        if focus_task is not None
                        else (focus_active.context.priority if focus_active is not None else None)
                    ),
                    "communication_rank": (
                        1 + communication.index(focus_task)
                        if focus_task is not None and focus_task.execution_queued_at is None
                        else None
                    ),
                    "execution_queue_rank": (
                        1 + queued.index(focus_task)
                        if focus_task is not None and focus_task.execution_queued_at is not None
                        else None
                    ),
                    "communication_wait_remaining_seconds": (
                        round(self._communication_remaining_locked(focus_task, now), 2)
                        if focus_task is not None and focus_task.execution_queued_at is None
                        else None
                    ),
                    "running_for_seconds": (
                        round(max(0.0, now - focus_active.started_at), 2)
                        if focus_active is not None
                        else None
                    ),
                },
                "active_projects": [
                    {
                        "task_id": item.task_id,
                        "repo_name": item.context.repo_name,
                        "repo_slug": item.context.repo_slug,
                        "task_kind": item.context.task_kind,
                        "running_for_seconds": round(max(0.0, now - item.started_at), 2),
                    }
                    for item in active[:4]
                ],
                "communication_projects": [
                    {
                        "task_id": task.task_id,
                        "repo_name": task.context.repo_name,
                        "repo_slug": task.context.repo_slug,
                        "task_kind": task.context.task_kind,
                        "announced_for_seconds": round(max(0.0, now - task.announced_at), 2),
                    }
                    for task in communication[:6]
                ],
                "execution_queue_projects": [
                    {
                        "task_id": task.task_id,
                        "repo_name": task.context.repo_name,
                        "repo_slug": task.context.repo_slug,
                        "task_kind": task.context.task_kind,
                        "queued_for_seconds": round(max(0.0, now - task.enqueued_at), 2),
                    }
                    for task in queued[:6]
                ],
            }

    def _announce(self, context: ScheduledTaskContext) -> _QueuedTask:
        with self._condition:
            now = monotonic()
            task = _QueuedTask(
                task_id=self._next_task_id,
                context=context,
                announced_at=now,
                execution_queued_at=None if self.communication_enabled else now,
                enqueued_at=now,
                completed_high_count_at_enqueue=self._completed_high_tasks,
            )
            if self.communication_enabled:
                current_cpu = self._current_docker_cpu_percent()
                if (
                    current_cpu is not None
                    and self.communication_bypass_cpu_threshold > 0
                    and current_cpu <= self.communication_bypass_cpu_threshold
                ):
                    task.communication_bypassed = True
                    task.communication_bypass_reason = (
                        f"Docker CPU {current_cpu:.1f}% is below the bypass threshold "
                        f"{self.communication_bypass_cpu_threshold:.1f}%."
                    )
                    task.execution_queued_at = now
            self._next_task_id += 1
            self._queue.append(task)
            self._communication_dirty = True
            self._condition.notify_all()
        self._log(
            "scheduler announced "
            f"{context.priority} {context.task_kind} for {context.repo_name}"
            f"{self._format_instance_suffix(context.instance_id)}"
            + (
                f"; communication min-k {self.communication_min_k}"
                if self.communication_min_k > 0
                else f"; communication wait {self._communication_wait_seconds_for_priority(context.priority):.0f}s"
            )
            +
            f"{self._format_description_suffix(context.description)}"
        )
        if task.communication_bypassed and task.communication_bypass_reason:
            self._log(
                "scheduler bypassed communication for "
                f"{context.priority} {context.task_kind} in {context.repo_name}"
                f"{self._format_instance_suffix(context.instance_id)}: {task.communication_bypass_reason}"
            )
        return task

    def _next_eligible_task_locked(self) -> _QueuedTask | None:
        for task in self._queue:
            if task.execution_queued_at is None:
                continue
            if self._is_high(task) or self._low_priority_task_is_ready_locked(task):
                return task
        return None

    def _next_wait_timeout_locked(self, task: _QueuedTask) -> float | None:
        if self._is_high(task):
            return None
        max_wait_at = task.enqueued_at + self.low_priority_max_wait_seconds
        remaining = max_wait_at - monotonic()
        return max(0.0, remaining)

    def _low_priority_task_is_ready_locked(self, task: _QueuedTask) -> bool:
        if self._is_high(task):
            return True
        if monotonic() - task.enqueued_at >= self.low_priority_max_wait_seconds:
            return True
        completed_since_enqueue = self._completed_high_tasks - task.completed_high_count_at_enqueue
        if completed_since_enqueue >= self.high_priority_head_start:
            return True
        if self._active_high_tasks == 0 and not any(self._is_high(other) for other in self._queue):
            return True
        return False

    def _communication_remaining_locked(self, task: _QueuedTask, now: float | None = None) -> float:
        if self.communication_min_k > 0:
            return 0.0
        current = monotonic() if now is None else now
        wait_seconds = self._communication_wait_seconds_for_priority(task.context.priority)
        return max(0.0, (task.announced_at + wait_seconds) - current)

    def _communication_ready_locked(self, task: _QueuedTask, now: float | None = None) -> bool:
        return self._communication_remaining_locked(task, now) <= 0

    def _communication_wait_seconds_for_priority(self, priority: str) -> float:
        return (
            self.high_priority_communication_wait_seconds
            if priority == "high"
            else self.low_priority_communication_wait_seconds
        )

    def _visible_worker_phrase(self) -> str:
        return self.visible_worker_label

    def _visible_worker_task_phrase(self) -> str:
        return f"{self.visible_worker_label} task"

    def _visible_update_phrase(self) -> str:
        return self.visible_update_label

    def _visible_source_label(self) -> str:
        return f"{self.visible_worker_label.capitalize()} source"

    def _communication_tasks_locked(self) -> list[_QueuedTask]:
        if not self.communication_enabled:
            return []
        return [task for task in self._queue if task.execution_queued_at is None]

    def _communication_candidate_pool_locked(self) -> list[_QueuedTask]:
        tasks = sorted(self._communication_tasks_locked(), key=lambda task: (task.announced_at, task.task_id))
        return tasks[: self.communication_top_k]

    def _effective_communication_min_k_locked(self) -> int:
        """communication_min_k, capped at how many workers could ever still announce.

        Only takes effect once at least one worker has called register_worker() --
        otherwise identical to self.communication_min_k, unchanged. When active, this
        is what prevents a fixed min-k from becoming permanently unreachable: if only
        1 worker out of an original 3 is still registered (the other 2 finished their
        whole batch and will never announce again), a min-k of 3 relaxes to 1 rather
        than waiting forever for peers that no longer exist.
        """
        if not self._registered_worker_repo_slugs:
            return self.communication_min_k
        return min(self.communication_min_k, max(1, len(self._registered_worker_repo_slugs)))

    def _communication_min_k_ready_locked(self, pool: list[_QueuedTask] | None = None) -> bool:
        effective_min_k = self._effective_communication_min_k_locked()
        if effective_min_k <= 0:
            return True
        candidate_pool = pool if pool is not None else self._communication_candidate_pool_locked()
        return len(candidate_pool) >= effective_min_k

    def _ensure_negotiation_state_locked(self, now: float) -> None:
        if not self._communication_dirty:
            return
        pool = self._communication_candidate_pool_locked()
        # A solo pool (exactly one real candidate) bypasses _negotiation_in_progress entirely --
        # see the "len(candidates) == 1" branch further down for the actual resolution, which
        # never touches self._negotiator at all. Computed here, before the gate below, is what
        # makes that bypass real: without it, a lone candidate could still be blocked forever by
        # an EARLIER, unrelated round's negotiate() call that's still running (or genuinely hung
        # with no client-side timeout -- confirmed live: a round's own tasks had already drained
        # into execution via the hard-wall-clock circuit breaker's fallback grant, yet
        # _negotiation_in_progress stayed stuck True because the abandoned call itself kept
        # running in the background, indefinitely blocking every subsequent negotiate() attempt,
        # including a trivially-resolvable solo one that doesn't even need a real LLM call).
        # Safe regardless of what that other call is doing: this whole method already runs only
        # while holding self._condition, the solo branch below never calls self._negotiator (the
        # one piece of state genuinely shared with a concurrently-running negotiate() call), and
        # task_ids are never reused, so there's no possible overlap between a solo candidate here
        # and whatever an unrelated, still-running negotiate() call is doing with its own,
        # different candidates.
        is_solo_pool = len(pool) == 1
        if self._negotiation_in_progress and not is_solo_pool:
            stuck_for = (
                now - self._negotiation_in_progress_since
                if self._negotiation_in_progress_since is not None
                else 0.0
            )
            if stuck_for < self._NEGOTIATION_IN_PROGRESS_STUCK_BACKSTOP_SECONDS:
                # Another thread already owns resolving this dirty state -- it briefly released
                # the lock to make the actual LLM call (see below) rather than freeze every other
                # scheduler operation for the whole negotiation. Don't start a second, redundant
                # negotiation for the same pool; the owning thread clears (or re-sets) dirty when
                # it finishes and reacquires the lock.
                return
            # Past the backstop: the thread that set this flag hit its own
            # _NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS abandonment a full extra timeout
            # period ago and STILL hasn't cleared this, meaning it's not merely slow -- it's
            # genuinely hung. See _NEGOTIATION_IN_PROGRESS_STUCK_BACKSTOP_SECONDS's own comment
            # for why forcing this clear and proceeding below is an acceptable, bounded risk
            # rather than leaving every future round dead for the rest of this run.
            self._log(
                "queue negotiation: _negotiation_in_progress has been stuck for "
                f"{stuck_for:.0f}s -- well past its own {self._NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS:.0f}s "
                "wall-clock timeout, so the thread that set it is genuinely hung, not just slow. "
                "Forcing it clear and starting a fresh negotiation attempt for the current pool "
                "instead of leaving every future round dead for the rest of this run."
            )
            self._negotiation_in_progress = False
            self._negotiation_in_progress_since = None
        # These two early-return checks must run BEFORE any state gets reset below. They used to
        # run AFTER an unconditional wipe of every negotiation dict (including
        # _negotiated_task_ids), so a dirty re-check that bailed out here (e.g. a new, unrelated
        # task announcing -- _announce() always sets _communication_dirty=True -- while an
        # already-decided task from a prior round is still sitting in the pool waiting to drain
        # into the execution queue) destroyed that already-decided task's negotiated state for
        # nothing, with no new decision to replace it. Confirmed live consequence:
        # _communication_can_admit_locked's `already_decided = task.task_id in
        # self._negotiated_task_ids` check would then read False for a task a completed round had
        # already decided on, incorrectly re-subjecting it to the min-k gate and stalling it
        # indefinitely -- directly contradicting that check's own adjacent comment.
        if not pool:
            self._communication_dirty = False
            return
        if not self._communication_min_k_ready_locked(pool):
            self._communication_dirty = False
            return

        # _negotiated_task_ids and _latest_negotiation_transcript are reset fresh every time we
        # reach this point (about to run, or fall back for, a real round). Neither reset is
        # load-bearing anymore: _negotiated_task_ids is unconditionally reassigned a few lines
        # below regardless of which path is taken, and _latest_negotiation_transcript is no longer
        # read for recording at all (see resolved_transcript's own declaration below for that) --
        # it now exists solely for _active_negotiation_snapshot_locked's live-status display, which
        # only ever wants "the most recent attempt," so resetting it here is harmless but likewise
        # not load-bearing. Kept for clarity / to avoid carrying a stale value into that display
        # for longer than necessary.
        self._negotiated_task_ids = []
        self._negotiation_rank_by_task_id = {}
        self._latest_negotiation_transcript = []
        # Deliberately NOT resetting the per-task-id dicts below (_negotiation_message_by_task_id,
        # _negotiation_rationale_by_task_id, _declaration_by_task_id and its reasoning-summary
        # sibling, _reasoning_by_task_id and its sibling, _discussion_by_task_id, _ballot_by_task_id
        # and its sibling, _coordination_decision_by_task_id, _final_duration_by_task_id,
        # _final_price_by_task_id) to {} here -- every consumer of these reads them via
        # .get(task_id, ...) scoped to a specific round's own candidate_ids (confirmed: none of
        # them are ever read as a raw whole-dict copy), and task_id values are never reused across
        # rounds, so leaving old entries in place carries no cross-round contamination risk. This
        # used to reset-then-blindly-reassign each one from this round's own outcome, which is fine
        # when the outcome is genuinely richer than what came before, but silently regressed
        # already-recorded real content (a real declaration, a real bid price) back to empty/
        # default placeholders whenever a SECOND real negotiate() call for the same still-open pool
        # produced a poorer result than the first (e.g. the first succeeded richly, the second hit
        # the hard-timeout fallback) -- the exact same "blind replace destroys already-recorded
        # data" bug class as the transcript-merge fix elsewhere in this file, just on these
        # sibling fields. Below, every assignment into these dicts uses .update(...) instead of a
        # bare `= ...` reassignment so new values win per-task-id without erasing entries for
        # task_ids this particular call doesn't touch. (_negotiation_rank_by_task_id and
        # _negotiated_task_ids are NOT included in this change: rank/order carry no "richness" to
        # preserve -- a fresh assignment for this round's own candidates is correct every time --
        # and unlike the dicts above, rank_by_task_id IS read elsewhere as a raw whole-dict copy
        # scoped to "this round only" (_record_negotiation_history_locked,
        # _active_negotiation_snapshot_locked), so leaving stale cross-round entries in it would
        # itself be a new bug.)

        candidates = [
            NegotiationCandidate(
                task_id=task.task_id,
                repo_name=task.context.repo_name,
                repo_slug=task.context.repo_slug,
                priority=task.context.priority,
                task_kind=task.context.task_kind,
                instance_id=task.context.instance_id,
                description=task.context.description,
                announced_for_seconds=max(0.0, now - task.announced_at),
                communication_wait_seconds=self._communication_wait_seconds_for_priority(task.context.priority),
                deadline_minutes=task.context.deadline_minutes or 0,
                deadline_remaining_seconds=self._deadline_remaining_seconds(task, now),
                memory_summary=self._memory_summary_for_repo_slug(task.context.repo_slug),
            )
            for task in pool
        ]
        # Call-local, not a shared instance attribute: this is what actually gets recorded for
        # THIS round by _record_negotiation_history_locked below. Previously this was read from
        # self._latest_negotiation_transcript instead -- a single shared field written by whatever
        # negotiate() call most recently succeeded, for ANY round, with no tag linking it back to
        # the round that produced it. Confirmed live consequence: round A gets recorded, an
        # unrelated round B later succeeds and overwrites the shared field with B's own messages,
        # then round A gets re-checked (e.g. the stall watchdog) and a fresh, non-merging entry is
        # created for round A's own candidate pool but seeded with round B's leftover transcript --
        # e.g. a real run's round 4 entry ended up containing direct messages from a worker that
        # was only ever a round-3 candidate. Scoping this to a local variable set fresh in every
        # branch below (solo/timeout/error -> [], success -> this call's own outcome.transcript)
        # makes that cross-round leak structurally impossible: there's no shared value left to leak
        # from. self._latest_negotiation_transcript itself is kept (still written on success, see
        # the negotiator branch below) purely for _active_negotiation_snapshot_locked's separate
        # "what's happening right now" live dashboard read, which genuinely wants "whatever the
        # most recent attempt looked like" and self-corrects once the in-flight round resolves --
        # unlike the permanent, compounding record this local variable now protects.
        resolved_transcript: list[dict[str, Any]] = []
        if is_solo_pool:
            # No real negotiation is needed or wanted here -- resolve directly, without ever
            # touching self._negotiator, so a lone candidate can never be blocked by whatever an
            # unrelated round's negotiate() call is doing (see is_solo_pool's own computation
            # above for the full story). This deliberately mirrors QueueNegotiator.negotiate()'s
            # own len(candidates)==1 short-circuit (same ordered_task_ids shape, same 0.0/0
            # final_duration/final_price convention for "no real bid was negotiated") rather than
            # calling into it, since calling it at all would mean going through
            # _negotiation_in_progress and the background-thread machinery this whole branch
            # exists to skip.
            ordered_task_ids = [candidates[0].task_id]
            self._final_duration_by_task_id[candidates[0].task_id] = 0.0
            self._final_price_by_task_id[candidates[0].task_id] = 0
        elif self._negotiator is not None:
            # Everything above this point (building `candidates`) is cheap, in-memory, and
            # needs the lock. The actual negotiation is neither -- it's a real, possibly
            # multi-minute chain of LLM calls (multiple phases, multiple candidates, retries)
            # -- and this whole method is normally called while the caller already holds
            # self._condition (see run()'s polling loop). Holding that lock for the full
            # negotiation duration would freeze every other scheduler operation system-wide
            # for as long as it takes: no other worker could register, unregister, announce,
            # or even check whether IT can be admitted, since all of those need the same
            # lock. Worse, if a single completion ever genuinely hangs (not just times out --
            # a true hang somewhere below the hard-timeout wrapper), the entire multi-worker
            # run freezes with it, indistinguishable from every other kind of stall. Release
            # the lock for the call itself and reacquire it before touching any shared state
            # again. _negotiation_in_progress guards against a second thread starting a
            # redundant, concurrent negotiation for the same dirty pool while the lock is
            # released here.
            self._negotiation_in_progress = True
            self._negotiation_in_progress_since = now
            prior_history = (
                self._prior_communication_history_for_prompt_locked()
                if self.communication_history_enabled
                else []
            )
            # Runs the real call on its own thread (rather than just synchronously, released-lock,
            # like before this was added) so a hard wall-clock ceiling can be enforced on it: see
            # _NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS's own docstring for the live incident
            # this fixes. result_holder is a plain dict private to this call, not an attribute on
            # self or the negotiator -- if the wait below times out, nothing ever reads it again,
            # so a late write from the abandoned thread after we've moved on is inert, not a race.
            result_holder: dict[str, Any] = {}
            done_event = Event()

            def _run_negotiation() -> None:
                try:
                    result_holder["outcome"] = self._negotiator.negotiate(
                        candidates, prior_history=prior_history,
                    )
                except Exception as exc:  # noqa: BLE001
                    result_holder["error"] = exc
                finally:
                    # Deliberately NOT resetting _negotiation_in_progress in the timeout branch
                    # below -- only here, once this call has genuinely finished (successfully,
                    # with an error, or after being abandoned by a timed-out wait). This is what
                    # keeps a second, concurrent negotiate() call from ever starting against the
                    # same QueueNegotiator instance while this one might still be running --
                    # QueueNegotiator.negotiate() mutates its own instance state
                    # (_cost_of_delay_multiplier_by_task_id) at the top of every call, which is
                    # not safe to do from two threads at once.
                    with self._condition:
                        self._negotiation_in_progress = False
                        self._negotiation_in_progress_since = None
                        self._condition.notify_all()
                    done_event.set()

            negotiation_thread = Thread(target=_run_negotiation, daemon=True)
            self._condition.release()
            try:
                negotiation_thread.start()
                finished_in_time = done_event.wait(timeout=self._NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS)
            finally:
                self._condition.acquire()

            if not finished_in_time:
                self._log(
                    "queue negotiation: a round exceeded its "
                    f"{self._NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS:.0f}s hard wall-clock "
                    "budget -- abandoning this attempt and falling back to a deterministic "
                    "admission order for now. The abandoned call keeps running in the background; "
                    "if/when it eventually finishes, its result is discarded (nothing reads it "
                    "anymore). No new negotiation attempt will start for this negotiator until "
                    "the abandoned call actually finishes and clears _negotiation_in_progress "
                    "itself, whenever that is."
                )
                outcome = None
                negotiation_error: Exception | None = TimeoutError(
                    f"negotiate() exceeded the {self._NEGOTIATION_HARD_WALL_CLOCK_TIMEOUT_SECONDS:.0f}s hard wall-clock budget"
                )
            elif "error" in result_holder:
                outcome = None
                negotiation_error = result_holder["error"]
            else:
                outcome = result_holder["outcome"]
                negotiation_error = None

            if negotiation_error is not None:
                self._log(f"queue negotiation failed: {negotiation_error}")
                ordered_task_ids = []
                # See _FALLBACK_GRANT_DURATION_SECONDS's own docstring: without this, every
                # candidate's final_duration/final_bid_price stays unset, which downstream
                # (_extract_latest_round_schedule) is indistinguishable from "was a candidate but
                # never got a valid entry" -- the caller then has nothing to actually grant and
                # has to start an entirely new round from scratch instead of using this fallback.
                self._final_duration_by_task_id.update({
                    candidate.task_id: self._FALLBACK_GRANT_DURATION_SECONDS for candidate in candidates
                })
                self._final_price_by_task_id.update({
                    candidate.task_id: self._FALLBACK_BID_PRICE for candidate in candidates
                })
                # Every per-task-id dict below (_discussion_by_task_id etc.) is deliberately never
                # reset to {} between rounds -- see the comment above `candidates = [` for why --
                # so a candidate that survives from an earlier, SUCCESSFULLY negotiated round (its
                # own admission kept getting deferred) still has that earlier round's real
                # discussion/reasoning/declaration/ballot text sitting in these dicts. Before this
                # fix, this branch never touched any of them, so the recorded entry for THIS
                # (timed-out) round silently inherited and mislabeled that stale text as if it were
                # fresh -- confirmed live twice (round 4 and round 7 of two separate runs). The
                # fix is symmetric with the success branch just below: explicitly overwrite this
                # round's own entry with a placeholder so it's honest about having no fresh data,
                # without touching the earlier round's own already-recorded entry at all.
                _no_fresh_data_text = (
                    "Negotiation timed out this round -- no fresh data was produced for this "
                    "candidate."
                )
                self._negotiation_message_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._negotiation_rationale_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._declaration_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._declaration_reasoning_summary_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._reasoning_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._reasoning_reasoning_summary_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._discussion_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._ballot_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._ballot_reasoning_summary_by_task_id.update({
                    candidate.task_id: _no_fresh_data_text for candidate in candidates
                })
                self._coordination_decision_by_task_id.update({
                    candidate.task_id: {
                        "action": "proceed",
                        "peer_task_id": None,
                        "reason": "",
                        "shared_context": "",
                    }
                    for candidate in candidates
                })
            else:
                ordered_task_ids = [task_id for task_id in outcome.ordered_task_ids if task_id in {task.task_id for task in pool}]
                # .update(), not `=`, for all of these -- see this method's own comment above
                # (right before `candidates = [`) for why: these dicts are never reset to {} above
                # anymore, so a merge here is what actually realizes that fix; a bare reassignment
                # would still silently drop entries for any task_id this particular outcome
                # doesn't happen to cover.
                self._negotiation_message_by_task_id.update(outcome.message_by_task_id)
                self._negotiation_rationale_by_task_id.update(outcome.rationale_by_task_id)
                self._declaration_by_task_id.update(outcome.declaration_by_task_id)
                self._declaration_reasoning_summary_by_task_id.update(
                    outcome.declaration_reasoning_summary_by_task_id
                )
                self._reasoning_by_task_id.update(outcome.reasoning_by_task_id)
                self._reasoning_reasoning_summary_by_task_id.update(
                    outcome.reasoning_reasoning_summary_by_task_id
                )
                self._discussion_by_task_id.update(outcome.discussion_by_task_id)
                self._ballot_by_task_id.update(outcome.ballot_by_task_id)
                self._ballot_reasoning_summary_by_task_id.update(
                    outcome.ballot_reasoning_summary_by_task_id
                )
                self._coordination_decision_by_task_id.update(outcome.coordination_decision_by_task_id)
                self._final_duration_by_task_id.update(outcome.final_duration_by_task_id)
                self._final_price_by_task_id.update(outcome.final_price_by_task_id)
                resolved_transcript = list(outcome.transcript)
                # Kept for _active_negotiation_snapshot_locked's live-status read only -- see the
                # comment on resolved_transcript's own declaration above. Not used for recording.
                self._latest_negotiation_transcript = list(outcome.transcript)
        else:
            ordered_task_ids = []

        if not ordered_task_ids:
            ordered_task_ids = [
                candidate.task_id
                for candidate in sorted(
                    candidates,
                    key=lambda candidate: (
                        0 if candidate.priority == "high" else 1,
                        candidate.deadline_remaining_seconds,
                        -candidate.announced_for_seconds,
                        candidate.task_id,
                    ),
                )
            ]

        missing_task_ids = [task.task_id for task in pool if task.task_id not in ordered_task_ids]
        ordered_task_ids.extend(missing_task_ids)
        self._negotiated_task_ids = ordered_task_ids
        for index, task_id in enumerate(self._negotiated_task_ids, start=1):
            self._negotiation_rank_by_task_id[task_id] = index
        for candidate in candidates:
            self._negotiation_message_by_task_id.setdefault(
                candidate.task_id,
                self._default_worker_message(candidate),
            )
            self._negotiation_rationale_by_task_id.setdefault(
                candidate.task_id,
                self._default_consensus_rationale(candidate),
            )
            self._declaration_by_task_id.setdefault(candidate.task_id, self._default_worker_message(candidate))
            self._declaration_reasoning_summary_by_task_id.setdefault(candidate.task_id, "")
            self._reasoning_by_task_id.setdefault(candidate.task_id, "")
            self._reasoning_reasoning_summary_by_task_id.setdefault(candidate.task_id, "")
            self._discussion_by_task_id.setdefault(candidate.task_id, "")
            self._ballot_by_task_id.setdefault(candidate.task_id, "")
            self._ballot_reasoning_summary_by_task_id.setdefault(candidate.task_id, "")
            self._coordination_decision_by_task_id.setdefault(
                candidate.task_id,
                {"action": "proceed", "peer_task_id": None, "reason": "", "shared_context": ""},
            )
        self._synchronize_delegated_waits_locked(candidates)
        self._record_negotiation_history_locked(candidates, ordered_task_ids, now, resolved_transcript)
        # Don't blindly clear dirty -- `pool` above was captured BEFORE releasing self._condition
        # for the (possibly multi-minute) negotiate() call. If a new, unrelated task announced
        # itself while the lock was released (_announce() sets _communication_dirty=True
        # unconditionally, with no idea a negotiation was already in flight for a different
        # pool), that signal must not get silently discarded by this method unconditionally
        # setting dirty back to False on its way out -- otherwise that new arrival sits
        # unnegotiated until some OTHER, unrelated event happens to re-touch dirty again. Recompute
        # the pool now and only clear dirty if every candidate in it was already accounted for by
        # this round's own decision.
        pool_after = self._communication_candidate_pool_locked()
        self._communication_dirty = any(task.task_id not in self._negotiated_task_ids for task in pool_after)

    def _synchronize_delegated_waits_locked(
        self,
        candidates: list[NegotiationCandidate],
    ) -> None:
        if not self.enable_delegated_peer_wait:
            return
        candidate_by_task_id = {candidate.task_id: candidate for candidate in candidates}
        candidate_repo_slugs = {candidate.repo_slug for candidate in candidates}
        current_delegate_repo_slugs: set[str] = set()
        for candidate in candidates:
            decision = self._coordination_decision_by_task_id.get(candidate.task_id) or {}
            action = str(decision.get("action") or "").strip().lower()
            if action != "delegate_to_peer":
                continue
            peer_task_id = decision.get("peer_task_id")
            peer_candidate = candidate_by_task_id.get(int(peer_task_id)) if isinstance(peer_task_id, int) else None
            current_delegate_repo_slugs.add(candidate.repo_slug)
            self._delegated_wait_by_repo_slug[candidate.repo_slug] = _DelegatedWait(
                waiter_repo_name=candidate.repo_name,
                waiter_repo_slug=candidate.repo_slug,
                peer_task_id=peer_candidate.task_id if peer_candidate is not None else None,
                peer_repo_name=peer_candidate.repo_name if peer_candidate is not None else None,
                peer_repo_slug=peer_candidate.repo_slug if peer_candidate is not None else None,
                reason=str(decision.get("reason") or "").strip(),
                expected_handoff=str(decision.get("shared_context") or "").strip(),
                created_at=monotonic(),
            )
        for repo_slug in candidate_repo_slugs:
            if repo_slug in current_delegate_repo_slugs:
                continue
            removed = self._delegated_wait_by_repo_slug.pop(repo_slug, None)
            if removed is not None:
                # A repo that previously had a real delegated wait recorded now resolved to
                # "proceed" (the default action) this round -- log it so a wait that was
                # established, then genuinely dropped (a round for the same repo re-ran and
                # degraded to a fallback/failure path that produced no fresh coordination
                # decision, or the repo legitimately moved on), leaves at least a trace instead of
                # vanishing from this dict with none at all.
                self._log(
                    f"scheduler cleared {removed.waiter_repo_name}'s delegated wait on "
                    f"{removed.peer_repo_name or 'its peer'}: this round's own decision for it is "
                    "'proceed', not 'delegate_to_peer'."
                )

    def _communication_can_admit_locked(self, task: _QueuedTask) -> bool:
        now = monotonic()
        pool = self._communication_candidate_pool_locked()
        if not pool:
            return False
        pool_ids = {item.task_id for item in pool}
        if task.task_id not in pool_ids:
            return False
        already_decided = task.task_id in self._negotiated_task_ids
        if not already_decided:
            # communication_min_k only gates whether a *new* round of negotiation
            # should start. It must not re-block a task that a completed round
            # already decided on, just because the pool shrank as the rest of that
            # same batch admitted before it.
            if not self._communication_min_k_ready_locked(pool):
                return False
            self._ensure_negotiation_state_locked(now)
            if task.task_id not in self._negotiated_task_ids:
                return False
        waive_wait = self._waive_wait_for_task_locked(task, now, pool)
        # Every member of the decided batch admits as soon as it is individually
        # ready, not only the single earliest-ranked one. The group moves into the
        # execution queue together, in the order the negotiation already settled.
        return waive_wait or self._communication_ready_locked(task, now)

    def _waive_wait_for_task_locked(
        self,
        task: _QueuedTask,
        now: float,
        pool: list[_QueuedTask] | None = None,
    ) -> bool:
        if not self.waive_wait_after_contested_negotiation:
            return False
        candidate_pool = pool if pool is not None else self._communication_candidate_pool_locked()
        if len(candidate_pool) < 2:
            return False
        if task.task_id not in {item.task_id for item in candidate_pool}:
            return False
        if self._communication_ready_locked(task, now):
            return False
        return bool(self._negotiated_task_ids)

    def _communication_wait_timeout_locked(self, task: _QueuedTask) -> float | None:
        now = monotonic()
        pool = self._communication_candidate_pool_locked()
        if not self._communication_min_k_ready_locked(pool):
            return None
        if self._waive_wait_for_task_locked(task, now, pool):
            return None
        remaining = self._communication_remaining_locked(task, now)
        if remaining > 0:
            return remaining
        positive_waits = [
            self._communication_remaining_locked(item, now)
            for item in pool
            if self._communication_remaining_locked(item, now) > 0
        ]
        if positive_waits:
            return min(positive_waits)
        return None

    def _communication_explanation_locked(self, task: _QueuedTask, now: float) -> dict[str, Any]:
        self._ensure_negotiation_state_locked(now)
        remaining = self._communication_remaining_locked(task, now)
        pool = self._communication_candidate_pool_locked()
        pool_ids = {item.task_id for item in pool}
        if task.task_id not in pool_ids:
            queue_position = sorted(self._communication_tasks_locked(), key=lambda item: (item.announced_at, item.task_id))
            outer_rank = next(
                (index for index, item in enumerate(queue_position, start=1) if item.task_id == task.task_id),
                None,
            )
            summary = (
                f"Outside the current top {self.communication_top_k} communication candidates. "
                "This request will join a negotiation round after earlier communication entries move out."
            )
            return {
                "negotiation_summary": summary,
                "negotiation_messages": [
                    "This task is waiting for a slot in the next communication round."
                ],
                "communication_rank": outer_rank,
                "negotiated_rank": None,
                "in_negotiation_round": False,
            }

        if not self._communication_min_k_ready_locked(pool):
            needed = self._effective_communication_min_k_locked()
            present = len(pool)
            summary = (
                f"Waiting for at least {needed} communication candidates before negotiation can release tasks "
                f"into the execution queue. {present} candidate(s) are present now."
            )
            return {
                "negotiation_summary": summary,
                "negotiation_messages": [
                    f"Communication min-k is active. Waiting for {needed} candidates; currently {present} present."
                ],
                "communication_rank": next(
                    (index for index, item in enumerate(pool, start=1) if item.task_id == task.task_id),
                    None,
                ),
                "negotiated_rank": None,
                "in_negotiation_round": False,
            }

        negotiated_rank = self._negotiation_rank_by_task_id.get(task.task_id)
        worker_message = self._negotiation_message_by_task_id.get(task.task_id, "")
        consensus_rationale = self._negotiation_rationale_by_task_id.get(task.task_id, "")
        declaration = self._declaration_by_task_id.get(task.task_id, "")
        discussion = self._discussion_by_task_id.get(task.task_id, "")
        ballot = self._ballot_by_task_id.get(task.task_id, "")
        coordination = self._coordination_decision_by_task_id.get(task.task_id, {})
        reasoning = self._reasoning_by_task_id.get(task.task_id, "")
        can_admit = self._communication_can_admit_locked(task)
        coordination_action = str(coordination.get("action") or "").strip().lower()
        wait_waived = self._waive_wait_for_task_locked(task, now, pool)

        if coordination_action == "wait_for_peer" and self.allow_coordination_withdrawal:
            peer_task_id = coordination.get("peer_task_id")
            summary = (
                f"Negotiation rank {negotiated_rank}. This request plans to withdraw from the communication queue and "
                f"re-plan from {self._visible_worker_task_phrase()} {peer_task_id} instead of consuming the next execution turn."
            )
            scheduler_reply = (
                "Scheduler reply: this request may withdraw from communication immediately and continue reasoning from "
                f"the {self._visible_worker_phrase()} clue."
            )
        elif coordination_action == "delegate_to_peer" and self.allow_coordination_withdrawal:
            peer_task_id = coordination.get("peer_task_id")
            summary = (
                f"Negotiation rank {negotiated_rank}. This request plans to withdraw, let {self._visible_worker_task_phrase()} {peer_task_id} "
                f"consume the next shared execution turn, and wait for a structured {self._visible_update_phrase()} before re-entering the slot."
            )
            scheduler_reply = (
                "Scheduler reply: this request may withdraw now, stay out of the shared slot, and re-plan once the "
                f"{self._visible_worker_phrase()} {self._visible_update_phrase()} arrives."
            )
        elif remaining > 0:
            if wait_waived:
                summary = (
                    f"Negotiation rank {negotiated_rank}. A contested negotiation round has already occurred, so the normal "
                    f"{self._communication_wait_seconds_for_priority(task.context.priority):.0f}s communication window is waived for this request."
                )
                scheduler_reply = "Scheduler reply: contested negotiation complete; this request now follows the negotiated order without further communication delay."
            else:
                summary = (
                    f"Negotiation rank {negotiated_rank}. Waiting on the "
                    f"{self._communication_wait_seconds_for_priority(task.context.priority):.0f}s communication window "
                    "before this request can join the execution queue."
                )
                scheduler_reply = (
                    f"Scheduler reply: keep negotiating for {remaining:.1f}s more before this request may enter the execution queue."
                )
        elif can_admit:
            summary = (
                f"Negotiation rank {negotiated_rank}. The communication window is complete and the negotiated order now admits this request."
            )
            scheduler_reply = "Scheduler reply: communication window satisfied; this request can now enter the execution queue."
        elif negotiated_rank is None:
            # Genuinely distinct from the "someone else outranks this request" case below --
            # negotiated_rank is only populated once a round involving this task_id has actually
            # resolved (see _ensure_negotiation_state_locked's own ordered_task_ids loop). Reaching
            # here with no rank at all means no candidate in this round has been ranked yet, not
            # that a real, ranked peer is ahead of this one -- confirmed via monitor_state.json
            # snapshots from live runs showing EVERY candidate in a round with this exact message
            # simultaneously, which is impossible if it genuinely meant "someone else ranks
            # higher." The old wording asserted a specific, false reason regardless of which was
            # actually true.
            summary = (
                "Negotiation rank None. This request's negotiation round has not resolved yet -- "
                "waiting for it to conclude before this request can be admitted."
            )
            scheduler_reply = "Scheduler reply: negotiation for this request's round is still in progress; check back once it resolves."
        else:
            summary = (
                f"Negotiation rank {negotiated_rank}. This request is ready, but a higher-ranked negotiated candidate is still ahead of it."
            )
            scheduler_reply = "Scheduler reply: hold position until the higher-ranked negotiated request moves into the execution queue."

        messages = []
        if worker_message:
            messages.append(worker_message)
        if consensus_rationale:
            messages.append(f"Consensus: {consensus_rationale}")
        messages.append(scheduler_reply)
        return {
            "negotiation_summary": summary,
            "negotiation_messages": messages,
            "declaration_message": declaration,
            "reasoning_message": reasoning,
            "discussion_message": discussion,
            "ballot_message": ballot,
            "coordination_decision": coordination,
            "consensus_message": consensus_rationale,
            "combined_worker_message": worker_message,
            "communication_rank": negotiated_rank,
            "negotiated_rank": negotiated_rank,
            "in_negotiation_round": True,
        }

    def _execution_explanation_locked(self, task: _QueuedTask, now: float) -> dict[str, Any]:
        deadline_remaining = self._deadline_remaining_seconds(task, now)
        deadline_text = ""
        if task.context.deadline_minutes:
            deadline_text = f" Worker deadline has {deadline_remaining:.1f}s remaining."
        if self._is_high(task):
            summary = (
                "High-priority execution work is favored over low-priority evaluation once it reaches "
                f"the front of the execution queue.{deadline_text}"
            )
            messages = [
                "Scheduler reply: this request is in the execution queue and can run as soon as concurrency and queue order allow.",
            ]
            if task.communication_bypassed and task.communication_bypass_reason:
                messages.insert(0, f"Communication bypassed: {task.communication_bypass_reason}")
            return {"negotiation_summary": summary, "negotiation_messages": messages}

        completed_since_enqueue = self._completed_high_tasks - task.completed_high_count_at_enqueue
        missing = max(0, self.high_priority_head_start - completed_since_enqueue)
        highs_remaining = self._active_high_tasks > 0 or any(
            other.execution_queued_at is not None and self._is_high(other) for other in self._queue
        )
        if not self._low_priority_task_is_ready_locked(task):
            summary = (
                f"Evaluation is waiting in the execution queue because it is low priority. "
                f"It still needs {missing} more high-priority completion(s), unless the {self.low_priority_max_wait_seconds:.0f}s "
                f"eval wait expires.{deadline_text}"
            )
            messages = [
                (
                    f"Scheduler reply: hold this evaluation behind cheaper high-priority work; "
                    f"{missing} more high-priority completion(s) are still required."
                ),
            ]
            if highs_remaining:
                messages.append("Other workers still have high-priority execution work pending, so they keep the lane first.")
        else:
            summary = f"Evaluation has cleared the execution-queue policy and can run when it reaches the front of the lane.{deadline_text}"
            messages = [
                "Scheduler reply: this evaluation has satisfied the execution-queue delay policy and is eligible to run.",
            ]
        if task.communication_bypassed and task.communication_bypass_reason:
            messages.insert(0, f"Communication bypassed: {task.communication_bypass_reason}")
        return {"negotiation_summary": summary, "negotiation_messages": messages}

    def _current_docker_cpu_percent(self) -> float | None:
        if self._cpu_percent_provider is None:
            return None
        try:
            value = self._cpu_percent_provider()
        except Exception as exc:
            self._log(f"docker cpu provider failed: {exc}")
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _record_negotiation_history_locked(
        self,
        candidates: list[NegotiationCandidate],
        ordered_task_ids: list[int],
        now: float,
        transcript: list[dict[str, Any]],
    ) -> None:
        # `transcript` is call-scoped, passed in by the caller from its own local
        # resolved_transcript (see _ensure_negotiation_state_locked's own comment on that
        # variable) -- NOT read from a shared instance attribute. This is what makes the merge
        # logic below safe against cross-round contamination: whatever this parameter holds is
        # guaranteed to belong to THIS specific call (this round's own negotiate() attempt, or []
        # if it was solo/timed out/errored), never leftover data from some unrelated round that
        # happened to finish in between two recordings of this one.
        candidate_ids = [candidate.task_id for candidate in candidates]
        signature = tuple(candidate_ids)
        if self._negotiation_history and tuple(self._negotiation_history[-1].get("candidate_task_ids", [])) == signature:
            last = self._negotiation_history[-1]
            last["updated_at_monotonic"] = round(now, 3)
            last["ordered_task_ids"] = list(ordered_task_ids)
            last["rank_by_task_id"] = dict(self._negotiation_rank_by_task_id)
            # Merge, never replace: a re-check of this SAME still-in-progress round (e.g. the
            # communication-stall watchdog forcing a fresh look) can easily produce fewer messages
            # of its own than what this round's entry already recorded, not more -- its own fresh
            # negotiate() attempt may not have gotten as far, or may have timed out entirely
            # (transcript == []). Blindly assigning last["transcript"] = list(transcript) (the old
            # behavior, previously sourced from a shared instance attribute) let a later, smaller
            # snapshot silently overwrite and permanently destroy earlier messages for this exact
            # round -- confirmed live: round 1 of a real run went from 24 recorded messages down to
            # 4 on its second update, round 3 from 22 down to 9, each time genuinely losing real
            # negotiator messages (not duplicates) from both this in-memory history AND the
            # append-only log below, since the log records whatever `last` looks like AFTER this
            # mutation. A transcript is a log of things that already happened; it should only ever
            # grow for a given round, never shrink.
            existing_transcript = last.get("transcript") or []
            existing_keys = {
                (m.get("round"), m.get("phase"), m.get("from_task_id"), m.get("to_task_id"), m.get("message"))
                for m in existing_transcript
            }
            merged_transcript = list(existing_transcript)
            for message in transcript:
                key = (message.get("round"), message.get("phase"), message.get("from_task_id"), message.get("to_task_id"), message.get("message"))
                if key not in existing_keys:
                    merged_transcript.append(message)
                    existing_keys.add(key)
            last["transcript"] = merged_transcript
            last["per_task"] = self._negotiation_per_task_payload_locked(candidate_ids)
            last["execution_outcomes"] = self._execution_outcomes_for_ordered_task_ids_locked(ordered_task_ids)
            last["has_agent_messages"] = bool(merged_transcript)
            last["compact_summary"] = self._compact_history_entry_locked(last)
            self._append_negotiation_history_log_locked(last)
            return

        self._negotiation_round_counter += 1
        entry = {
            "round_id": self._negotiation_round_counter,
            "created_at_monotonic": round(now, 3),
            "updated_at_monotonic": round(now, 3),
            "candidate_task_ids": candidate_ids,
            "ordered_task_ids": list(ordered_task_ids),
            "rank_by_task_id": dict(self._negotiation_rank_by_task_id),
            "candidates": self._serialize_negotiation_candidates(candidates),
            "per_task": self._negotiation_per_task_payload_locked(candidate_ids),
            "transcript": list(transcript),
            "execution_outcomes": self._execution_outcomes_for_ordered_task_ids_locked(ordered_task_ids),
            "has_agent_messages": bool(transcript),
        }
        entry["compact_summary"] = self._compact_history_entry_locked(entry)
        self._negotiation_history.append(entry)
        self._append_negotiation_history_log_locked(entry)
        # self._negotiation_history itself stays capped: one entry per negotiation round, each
        # carrying full transcripts/reasoning text, is what silently bloated monitor_state.json
        # past 20MB+ on long runs, making the periodic monitor write fall tens of minutes behind
        # and making the run look inactive/stale in the dashboard even though it was still
        # running. The full, uncapped record now lives in the append-only log written above
        # instead (see _append_negotiation_history_log_locked) -- an append is O(1) regardless of
        # how large the file gets, so it doesn't reintroduce the rewrite-cost problem this cap
        # was built to avoid.
        if len(self._negotiation_history) > self._NEGOTIATION_HISTORY_MAXLEN:
            del self._negotiation_history[: len(self._negotiation_history) - self._NEGOTIATION_HISTORY_MAXLEN]

    def _append_negotiation_history_log_locked(self, entry: dict[str, Any]) -> None:
        if self._negotiation_history_log_fh is None:
            return
        try:
            self._negotiation_history_log_fh.write(json.dumps(entry, sort_keys=True) + "\n")
            self._negotiation_history_log_fh.flush()
        except Exception:
            # Best-effort: the full-history log is a supplementary record for the dashboard, not
            # load-bearing for scheduling itself. Never let a write/serialization failure here
            # interrupt actual negotiation.
            pass

    def _negotiation_per_task_payload_locked(self, candidate_ids: list[int]) -> dict[int, dict[str, Any]]:
        payload: dict[int, dict[str, Any]] = {}
        for task_id in candidate_ids:
            payload[task_id] = {
                "message": self._negotiation_message_by_task_id.get(task_id, ""),
                "rationale": self._negotiation_rationale_by_task_id.get(task_id, ""),
                "declaration": self._declaration_by_task_id.get(task_id, ""),
                "declaration_reasoning_summary": self._declaration_reasoning_summary_by_task_id.get(task_id, ""),
                "reasoning": self._reasoning_by_task_id.get(task_id, ""),
                "reasoning_reasoning_summary": self._reasoning_reasoning_summary_by_task_id.get(task_id, ""),
                "discussion": self._discussion_by_task_id.get(task_id, ""),
                "ballot": self._ballot_by_task_id.get(task_id, ""),
                "ballot_reasoning_summary": self._ballot_reasoning_summary_by_task_id.get(task_id, ""),
                "coordination_decision": self._coordination_decision_by_task_id.get(task_id, {}),
                # Additive -- present only for callers whose negotiator actually populates
                # NegotiationOutcome.final_duration_by_task_id/final_price_by_task_id (currently
                # only the ProgramBench auction script's response-phase bid revisions feed this;
                # every other caller's negotiator leaves it as the empty dict from
                # NegotiationOutcome's default, so this is always {} for them, same as before).
                "final_duration_seconds": self._final_duration_by_task_id.get(task_id),
                "final_bid_price": self._final_price_by_task_id.get(task_id),
                "rank": self._negotiation_rank_by_task_id.get(task_id),
            }
        return payload

    def _execution_outcomes_for_ordered_task_ids_locked(
        self,
        ordered_task_ids: list[int],
    ) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for task_id in ordered_task_ids:
            outcomes.append(self._execution_outcome_for_task_id_locked(task_id))
        return outcomes

    def _execution_outcome_for_task_id_locked(self, task_id: int) -> dict[str, Any]:
        execution_record = self._execution_record_by_task_id.get(task_id)
        if execution_record is not None:
            return {
                "task_id": task_id,
                "repo_name": execution_record.context.repo_name,
                "repo_slug": execution_record.context.repo_slug,
                "instance_id": execution_record.context.instance_id,
                "task_kind": execution_record.context.task_kind,
                "priority": execution_record.context.priority,
                "state": "completed",
                "execution_time_seconds": round(max(0.0, execution_record.finished_at - execution_record.started_at), 2),
            }

        active = self._active_task_details.get(task_id)
        if active is not None:
            return {
                "task_id": task_id,
                "repo_name": active.context.repo_name,
                "repo_slug": active.context.repo_slug,
                "instance_id": active.context.instance_id,
                "task_kind": active.context.task_kind,
                "priority": active.context.priority,
                "state": "running",
                "execution_time_seconds": None,
            }

        queued = next((item for item in self._queue if item.task_id == task_id), None)
        if queued is not None:
            state = "execution_queue" if queued.execution_queued_at is not None else "communication_queue"
            return {
                "task_id": task_id,
                "repo_name": queued.context.repo_name,
                "repo_slug": queued.context.repo_slug,
                "instance_id": queued.context.instance_id,
                "task_kind": queued.context.task_kind,
                "priority": queued.context.priority,
                "state": state,
                "execution_time_seconds": None,
            }

        withdrawn = self._withdrawal_record_by_task_id.get(task_id)
        if withdrawn is not None:
            return {
                "task_id": task_id,
                "repo_name": withdrawn.context.repo_name,
                "repo_slug": withdrawn.context.repo_slug,
                "instance_id": withdrawn.context.instance_id,
                "task_kind": withdrawn.context.task_kind,
                "priority": withdrawn.context.priority,
                "state": "withdrawn_for_replan",
                "execution_time_seconds": None,
            }

        return {
            "task_id": task_id,
            "repo_name": None,
            "repo_slug": None,
            "instance_id": None,
            "task_kind": None,
            "priority": None,
            "state": "unknown",
            "execution_time_seconds": None,
        }

    def _refresh_execution_outcomes_locked(self, completed_task_id: int | None = None) -> None:
        for entry in self._negotiation_history:
            ordered_task_ids = entry.get("ordered_task_ids") or []
            if completed_task_id is not None and completed_task_id not in ordered_task_ids:
                continue
            entry["execution_outcomes"] = self._execution_outcomes_for_ordered_task_ids_locked(ordered_task_ids)
            entry["compact_summary"] = self._compact_history_entry_locked(entry)
            # This class's own docstring (near _negotiation_history_log_fh's own construction)
            # documents that "later execution-outcome refreshes" are one of the update kinds that
            # should produce another appended line in the permanent, uncapped
            # negotiation_history_full.jsonl log -- this call was missing entirely, so an
            # execution-outcome refresh updated the in-memory/monitor_state.json copy but never
            # made it into the permanent log at all.
            self._append_negotiation_history_log_locked(entry)

    def _prior_communication_history_for_prompt_locked(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for entry in self._negotiation_history:
            if not entry.get("has_agent_messages"):
                continue
            history.append(
                {
                    "round_id": entry.get("round_id"),
                    "candidate_task_ids": list(entry.get("candidate_task_ids") or []),
                    "ordered_task_ids": list(entry.get("ordered_task_ids") or []),
                    "candidates": list(entry.get("candidates") or []),
                    "per_task": dict(entry.get("per_task") or {}),
                    "transcript": list(entry.get("transcript") or []),
                    "execution_outcomes": list(entry.get("execution_outcomes") or []),
                    "compact_summary": entry.get("compact_summary", ""),
                }
            )
        return history

    def _compact_history_entry_locked(self, entry: dict[str, Any]) -> str:
        round_id = entry.get("round_id", "?")
        lines = [f"Round {round_id}"]
        candidates = entry.get("candidates") or []
        if candidates:
            candidate_bits = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate_bits.append(
                    f"Task {candidate.get('task_id')} ({candidate.get('repo_name')}, "
                    f"{candidate.get('priority')}, {candidate.get('task_kind')}, "
                    f"deadline={candidate.get('deadline_remaining_seconds')}s)"
                )
            if candidate_bits:
                lines.append("Candidates: " + "; ".join(candidate_bits))
        ordered = entry.get("ordered_task_ids") or []
        if ordered:
            lines.append("Consensus order: " + " > ".join(str(task_id) for task_id in ordered))
        outcomes = entry.get("execution_outcomes") or []
        if outcomes:
            outcome_bits = []
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                label = f"Task {outcome.get('task_id')} ({outcome.get('repo_name')})"
                state = outcome.get("state") or "unknown"
                duration = outcome.get("execution_time_seconds")
                if duration is None:
                    outcome_bits.append(f"{label}: {state}")
                else:
                    outcome_bits.append(f"{label}: {state} in {duration}s")
            if outcome_bits:
                lines.append("Execution outcomes: " + "; ".join(outcome_bits))
        return " | ".join(lines)

    def _active_negotiation_snapshot_locked(self, now: float) -> dict[str, Any] | None:
        pool = self._communication_candidate_pool_locked()
        if not pool:
            return None
        candidates = [
            NegotiationCandidate(
                task_id=task.task_id,
                repo_name=task.context.repo_name,
                repo_slug=task.context.repo_slug,
                priority=task.context.priority,
                task_kind=task.context.task_kind,
                instance_id=task.context.instance_id,
                description=task.context.description,
                announced_for_seconds=max(0.0, now - task.announced_at),
                communication_wait_seconds=self._communication_wait_seconds_for_priority(task.context.priority),
                deadline_minutes=task.context.deadline_minutes or 0,
                deadline_remaining_seconds=self._deadline_remaining_seconds(task, now),
                memory_summary=self._memory_summary_for_repo_slug(task.context.repo_slug),
            )
            for task in pool
        ]
        return {
            "generated_at_monotonic": round(now, 3),
            "candidate_task_ids": [task.task_id for task in pool],
            "ordered_task_ids": list(self._negotiated_task_ids),
            "rank_by_task_id": dict(self._negotiation_rank_by_task_id),
            "candidates": self._serialize_negotiation_candidates(candidates),
            "per_task": self._negotiation_per_task_payload_locked([task.task_id for task in pool]),
            "transcript": list(self._latest_negotiation_transcript),
        }

    @staticmethod
    def _serialize_negotiation_candidates(candidates: list[NegotiationCandidate]) -> list[dict[str, Any]]:
        return [
            {
                "task_id": candidate.task_id,
                "repo_name": candidate.repo_name,
                "repo_slug": candidate.repo_slug,
                "priority": candidate.priority,
                "task_kind": candidate.task_kind,
                "instance_id": candidate.instance_id,
                "description": candidate.description,
                "announced_for_seconds": round(candidate.announced_for_seconds, 2),
                "communication_wait_seconds": round(candidate.communication_wait_seconds, 2),
                "deadline_minutes": candidate.deadline_minutes,
                "deadline_remaining_seconds": round(candidate.deadline_remaining_seconds, 2),
                "memory_summary": candidate.memory_summary,
            }
            for candidate in candidates
        ]

    def _memory_summary_for_repo_slug(self, repo_slug: str) -> str:
        if self._negotiation_memory_provider is None:
            return ""
        try:
            return str(self._negotiation_memory_provider(repo_slug) or "")
        except Exception as exc:
            self._log(f"negotiation memory provider failed for {repo_slug}: {exc}")
            return ""

    def _default_worker_message(self, candidate: NegotiationCandidate) -> str:
        if candidate.priority == "high":
            return (
                f"{candidate.repo_name} argues for earlier admission because {candidate.task_kind} is cheaper container-side work, "
                f"while its project deadline has {candidate.deadline_remaining_seconds:.1f}s remaining."
            )
        return (
            f"{candidate.repo_name} acknowledges that evaluation is expensive, but asks not to be delayed indefinitely because "
            f"its deadline has {candidate.deadline_remaining_seconds:.1f}s remaining."
        )

    def _default_consensus_rationale(self, candidate: NegotiationCandidate) -> str:
        if candidate.priority == "high":
            return "Worker consensus usually favors cheaper high-priority command/test work unless another worker has materially tighter deadline pressure."
        return "Worker consensus usually keeps low-priority evaluation behind cheaper work unless deadline pressure or wait time becomes dominant."

    def _deadline_remaining_seconds(self, task: _QueuedTask, now: float) -> float:
        return self._deadline_remaining_seconds_from_context(task.context, now)

    def _deadline_remaining_seconds_from_context(self, context: ScheduledTaskContext, now: float) -> float:
        # Prefer the live, registered DeadlineClock over context.deadline_at_monotonic
        # whenever one exists for this repo. context.deadline_at_monotonic is a plain
        # float, frozen at whatever the deadline was the instant this task's context
        # was constructed (announce time) -- it is NEVER updated afterward, even
        # though the real DeadlineClock keeps getting silently credited throughout a
        # communication-queue wait (see run()). Observed live: this is exactly why a
        # dashboard's "Communication" queue clock and its worker-level
        # "Recently Completed" clock showed genuinely different, disagreeing values --
        # they were reading two entirely disconnected sources, only one of which
        # (the registered clock, via monitor.deadline_remaining_seconds) was actually
        # being kept live-accurate. Falling back to the static float below preserves
        # exact prior behavior for every caller that never registers a clock at all.
        clock = self._deadline_clocks_by_repo_slug.get(context.repo_slug)
        if clock is not None:
            return clock.remaining_seconds(now)
        if context.deadline_at_monotonic is None:
            return 0.0
        return max(0.0, context.deadline_at_monotonic - now)

    @staticmethod
    def _is_high(task: _QueuedTask) -> bool:
        return task.context.priority == "high"

    def _coordination_withdrawal_for_task_locked(
        self,
        task: _QueuedTask,
    ) -> SchedulerCoordinationWithdrawal | None:
        if not self.allow_coordination_withdrawal:
            return None
        if task.context.task_kind == "evaluation":
            return None
        persisted_withdrawal = self._delegation_withdrawal_for_repo_locked(task.context.repo_slug)
        if persisted_withdrawal is not None:
            return persisted_withdrawal
        now = monotonic()
        pool = self._communication_candidate_pool_locked()
        if task.task_id not in {item.task_id for item in pool}:
            return None
        self._ensure_negotiation_state_locked(now)
        decision = self._coordination_decision_by_task_id.get(task.task_id) or {}
        action = str(decision.get("action") or "").strip().lower()
        if action not in {"wait_for_peer", "delegate_to_peer"}:
            return None
        peer_task_id = decision.get("peer_task_id")
        peer_task = next((item for item in pool if item.task_id == peer_task_id), None)
        if peer_task is None:
            peer_task = next((item for item in self._active_task_details.values() if item.task_id == peer_task_id), None)
        peer_repo_name = peer_task.context.repo_name if peer_task is not None else None
        peer_repo_slug = peer_task.context.repo_slug if peer_task is not None else None
        reason = str(decision.get("reason") or "").strip()
        shared_context = str(decision.get("shared_context") or "").strip()
        if not (reason or shared_context):
            return None
        if action == "delegate_to_peer" and self.enable_delegated_peer_wait:
            self._delegated_wait_by_repo_slug[task.context.repo_slug] = _DelegatedWait(
                waiter_repo_name=task.context.repo_name,
                waiter_repo_slug=task.context.repo_slug,
                peer_task_id=int(peer_task_id) if peer_task_id is not None else None,
                peer_repo_name=peer_repo_name,
                peer_repo_slug=peer_repo_slug,
                reason=reason,
                expected_handoff=shared_context,
                created_at=now,
            )
        return SchedulerCoordinationWithdrawal(
            task_id=task.task_id,
            repo_name=task.context.repo_name,
            repo_slug=task.context.repo_slug,
            instance_id=task.context.instance_id,
            peer_task_id=int(peer_task_id) if peer_task_id is not None else None,
            peer_repo_name=peer_repo_name,
            peer_repo_slug=peer_repo_slug,
            reason=reason,
            shared_context=shared_context,
            defer_execution=(action == "delegate_to_peer" and self.enable_delegated_peer_wait),
        )

    def delegated_waiters_for_peer(self, peer_repo_slug: str) -> list[dict[str, Any]]:
        with self._condition:
            return [
                {
                    "waiter_repo_name": item.waiter_repo_name,
                    "waiter_repo_slug": item.waiter_repo_slug,
                    "peer_task_id": item.peer_task_id,
                    "peer_repo_name": item.peer_repo_name,
                    "peer_repo_slug": item.peer_repo_slug,
                    "reason": item.reason,
                    "expected_handoff": item.expected_handoff,
                }
                for item in self._delegated_wait_by_repo_slug.values()
                if item.peer_repo_slug == peer_repo_slug
            ]

    def resolve_delegated_waiters(self, peer_repo_slug: str, handoff_text: str) -> list[str]:
        handoff = str(handoff_text or "").strip()
        if not handoff:
            return []
        with self._condition:
            # Deliberately no _prune_stale_delegations_locked() call here: the peer calling this
            # method has, by definition, just finished its own task and already left both
            # self._queue and _active_task_details, so pruning at this point would delete the
            # very waiter entry this call is about to deliver to -- in the normal, successful
            # case, not just as a rare race. Staleness cleanup still happens via the other call
            # sites (withdrawal check, consume, status) for waits whose peer never comes back.
            now = monotonic()
            delivered: list[str] = []
            for waiter_slug, item in list(self._delegated_wait_by_repo_slug.items()):
                if item.peer_repo_slug != peer_repo_slug:
                    continue
                self._delivered_delegation_by_repo_slug[waiter_slug] = _DeliveredDelegation(
                    waiter_repo_name=item.waiter_repo_name,
                    waiter_repo_slug=item.waiter_repo_slug,
                    peer_task_id=item.peer_task_id,
                    peer_repo_name=item.peer_repo_name,
                    peer_repo_slug=item.peer_repo_slug,
                    handoff_text=handoff,
                    delivered_at=now,
                )
                delivered.append(waiter_slug)
                del self._delegated_wait_by_repo_slug[waiter_slug]
            if delivered:
                self._condition.notify_all()
            return delivered

    def consume_delivered_delegation(self, repo_slug: str) -> dict[str, Any] | None:
        with self._condition:
            self._prune_stale_delegations_locked()
            delivered = self._delivered_delegation_by_repo_slug.pop(repo_slug, None)
            if delivered is None:
                return None
            return {
                "waiter_repo_name": delivered.waiter_repo_name,
                "waiter_repo_slug": delivered.waiter_repo_slug,
                "peer_task_id": delivered.peer_task_id,
                "peer_repo_name": delivered.peer_repo_name,
                "peer_repo_slug": delivered.peer_repo_slug,
                "handoff_text": delivered.handoff_text,
                "delivered_at": delivered.delivered_at,
            }

    def delegation_status(self, repo_slug: str) -> dict[str, Any]:
        with self._condition:
            self._prune_stale_delegations_locked()
            delivered = self._delivered_delegation_by_repo_slug.get(repo_slug)
            waiting = self._delegated_wait_by_repo_slug.get(repo_slug)
            return {
                "waiting": waiting is not None,
                "delivered": delivered is not None,
                "peer_repo_slug": (
                    delivered.peer_repo_slug
                    if delivered is not None
                    else (waiting.peer_repo_slug if waiting is not None else None)
                ),
                "peer_repo_name": (
                    delivered.peer_repo_name
                    if delivered is not None
                    else (waiting.peer_repo_name if waiting is not None else None)
                ),
            }

    def _delegation_withdrawal_for_repo_locked(
        self,
        repo_slug: str,
    ) -> SchedulerCoordinationWithdrawal | None:
        self._prune_stale_delegations_locked()
        delivered = self._delivered_delegation_by_repo_slug.get(repo_slug)
        if delivered is not None:
            return SchedulerCoordinationWithdrawal(
                task_id=0,
                repo_name=delivered.waiter_repo_name,
                repo_slug=delivered.waiter_repo_slug,
                instance_id=None,
                peer_task_id=delivered.peer_task_id,
                peer_repo_name=delivered.peer_repo_name,
                peer_repo_slug=delivered.peer_repo_slug,
                reason=(
                    f"Another {self._visible_worker_phrase()} finished its execution turn. Re-plan from the delivered "
                    f"{self._visible_update_phrase()} before using the shared slot again."
                ),
                shared_context=delivered.handoff_text,
            )
        waiting = self._delegated_wait_by_repo_slug.get(repo_slug)
        if waiting is not None:
            return SchedulerCoordinationWithdrawal(
                task_id=0,
                repo_name=waiting.waiter_repo_name,
                repo_slug=waiting.waiter_repo_slug,
                instance_id=None,
                peer_task_id=waiting.peer_task_id,
                peer_repo_name=waiting.peer_repo_name,
                peer_repo_slug=waiting.peer_repo_slug,
                reason=(
                    f"Another {self._visible_worker_phrase()} execution is still in progress. Stay out of the shared slot "
                    f"until that {self._visible_update_phrase()} arrives."
                ),
                shared_context=waiting.expected_handoff,
                # No new information exists yet -- repeating a synthetic "please wait" message on
                # every subsequent command would just burn turns with nothing actionable. Defer
                # the whole instance instead so the driver can move on and retry it once delivered.
                defer_execution=True,
            )
        return None

    def _prune_stale_delegations_locked(self) -> None:
        active_or_queued_repo_slugs = {
            item.context.repo_slug
            for item in self._active_task_details.values()
        }
        active_or_queued_repo_slugs.update(task.context.repo_slug for task in self._queue)
        stale_waiter_slugs = [
            waiter_slug
            for waiter_slug, item in self._delegated_wait_by_repo_slug.items()
            if not item.peer_repo_slug or item.peer_repo_slug not in active_or_queued_repo_slugs
        ]
        if not stale_waiter_slugs:
            return
        for waiter_slug in stale_waiter_slugs:
            item = self._delegated_wait_by_repo_slug[waiter_slug]
            # Log before deleting -- this dict has no snapshot/history exposure anywhere, so
            # without this line a delegated wait vanishes with literally zero trace once its peer
            # is no longer active/queued (confirmed: nothing moves this to
            # _delivered_delegation_by_repo_slug or any withdrawal-style record first).
            self._log(
                f"scheduler pruned a stale delegated wait: {item.waiter_repo_name} was waiting on "
                f"{item.peer_repo_name or 'an unknown peer'}, which is no longer active or queued."
            )
            del self._delegated_wait_by_repo_slug[waiter_slug]
        self._condition.notify_all()

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger.log(message)

    @staticmethod
    def _format_instance_suffix(instance_id: str | None) -> str:
        return f" [{instance_id}]" if instance_id else ""

    @staticmethod
    def _format_description_suffix(description: str) -> str:
        return f": {description}" if description else ""


class ScheduledEnvironment:
    """Wrap a mini-SWE-agent environment so container commands go through the scheduler."""

    def __init__(
        self,
        inner: Any,
        scheduler: DockerTaskScheduler,
        *,
        repo_name: str,
        repo_slug: str,
        instance_id: str,
        test_command_pattern,
        deadline_minutes: int | None = None,
        deadline_at_monotonic: "float | DeadlineClock | None" = None,
        allow_solution_withdrawal: bool = False,
        use_safe_task_descriptions: bool = False,
        announce_deadline_in_output: bool = False,
    ) -> None:
        self._inner = inner
        self._scheduler = scheduler
        self._repo_name = repo_name
        self._repo_slug = repo_slug
        self._instance_id = instance_id
        self._test_command_pattern = test_command_pattern
        self._deadline_minutes = deadline_minutes
        self._deadline_at_monotonic = deadline_at_monotonic
        self._allow_solution_withdrawal = allow_solution_withdrawal
        self._use_safe_task_descriptions = use_safe_task_descriptions
        # Off by default -- only run_swebench_subrun_repo_agents.py opts in. When on, every
        # command's output gets a trailing line with the live, pause-aware remaining time
        # read straight from deadline_at_monotonic (a DeadlineClock when pausing is active),
        # so the solver sees an accurate number on every single turn, not just a one-time
        # snapshot in its initial task text.
        self._announce_deadline_in_output = announce_deadline_in_output

    def _visible_worker_phrase(self) -> str:
        return self._scheduler.visible_worker_label

    def _visible_worker_task_phrase(self) -> str:
        return f"{self._scheduler.visible_worker_label} task"

    def _visible_source_label(self) -> str:
        return f"{self._scheduler.visible_worker_label.capitalize()} source"

    def _format_coordination_update(self, exc: SchedulerCoordinationWithdrawal) -> str:
        lines = [
            "Scheduler coordination update:",
            "Your requested command was not executed in order to save the shared execution slot.",
        ]
        if exc.peer_repo_name:
            lines.append(f"{self._visible_source_label()}: {exc.peer_repo_name}.")
        if exc.reason:
            lines.append(f"Reason: {exc.reason}")
        if exc.shared_context:
            lines.append(f"Reusable clue: {exc.shared_context}")
        lines.append(
            "Re-plan immediately. You can draft or revise the patch now, or choose a narrower next command only if you still need one."
        )
        return "\n".join(lines) + "\n"

    def _coordination_update_result(self, exc: SchedulerCoordinationWithdrawal) -> dict[str, Any]:
        output = self._format_coordination_update(exc)
        return self._with_deadline_announcement({
            "output": output,
            "raw_output": output,
            "returncode": 0,
            "exception_info": "",
        })

    def _with_deadline_announcement(self, result: dict[str, Any]) -> dict[str, Any]:
        """Append a live, pause-aware remaining-time line to a command's output.

        Off unless announce_deadline_in_output was set at construction (only
        run_swebench_subrun_repo_agents.py opts in). Reads deadline_at_monotonic fresh
        on every call, exactly like the ScheduledTaskContext build above, so a
        DeadlineClock's accumulated negotiation-pause credit is reflected on every turn --
        not just in the one-time snapshot baked into the solver's initial task text.
        """
        if not self._announce_deadline_in_output or self._deadline_at_monotonic is None:
            return result
        remaining = (
            self._deadline_at_monotonic.remaining_seconds()
            if isinstance(self._deadline_at_monotonic, DeadlineClock)
            else max(0.0, self._deadline_at_monotonic - monotonic())
        )
        suffix = (
            f"\n\n[deadline: approximately {remaining:.0f}s of execution-time budget "
            f"remain before your hard deadline for this subrun.]"
        )
        updated = dict(result)
        updated["output"] = str(updated.get("output", "")) + suffix
        return updated

    def execute(self, action: dict | str, cwd: str = "", **kwargs) -> dict[str, Any]:
        command = self._extract_command(action)
        task_kind = "unit_test" if self._test_command_pattern.search(command) else "shell_command"
        context = ScheduledTaskContext(
            priority="high",
            task_kind=task_kind,
            repo_name=self._repo_name,
            repo_slug=self._repo_slug,
            instance_id=self._instance_id,
            description=_describe_command(
                command,
                task_kind=task_kind,
                conservative=self._use_safe_task_descriptions,
            ),
            deadline_minutes=self._deadline_minutes,
            # Resolved live on every call (not just once at construction) so a
            # DeadlineClock's accumulated negotiation-time extensions are reflected
            # in what gets shown to the negotiator for every subsequent command.
            # For a plain float this is a no-op.
            deadline_at_monotonic=(
                None if self._deadline_at_monotonic is None else float(self._deadline_at_monotonic)
            ),
        )
        try:
            result = self._scheduler.run(
                lambda: self._inner.execute(action, cwd=cwd, **kwargs),
                context=context,
            )
            return self._with_deadline_announcement(result)
        except SchedulerCoordinationWithdrawal as exc:
            if not self._allow_solution_withdrawal:
                raise
            if exc.defer_execution:
                # A genuine delegate-to-peer relationship: swallowing this into a synthetic tool
                # message would just have the agent burn every subsequent command on the same
                # withdrawn stub until the delegation resolves. Raise SchedulerExecutionDeferred
                # instead so it propagates through mini-swe-agent's run() loop (caught generically
                # by its `except Exception` handler, which records exit_status="SchedulerExecutionDeferred"
                # on the trajectory) up to run_instance_direct's dedicated handler, which cleanly
                # parks this instance for a later attempt instead of continuing to spend turns on it.
                raise SchedulerExecutionDeferred(
                    repo_name=exc.repo_name,
                    repo_slug=exc.repo_slug,
                    instance_id=self._instance_id,
                    peer_task_id=exc.peer_task_id,
                    peer_repo_name=exc.peer_repo_name,
                    peer_repo_slug=exc.peer_repo_slug,
                    reason=exc.reason,
                    shared_context=exc.shared_context,
                ) from exc
            return self._coordination_update_result(exc)

    def cleanup(self) -> Any:
        return self._inner.cleanup()

    def serialize(self) -> dict[str, Any]:
        return self._inner.serialize()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self._inner.get_template_vars(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @staticmethod
    def _extract_command(action: dict | str) -> str:
        if isinstance(action, str):
            return action
        return str(action.get("command") or "")


def _truncate_command(command: str, limit: int = 120) -> str:
    command = " ".join(command.strip().split())
    if len(command) <= limit:
        return command
    return command[: limit - 3] + "..."


_TEST_TARGET_RE = re.compile(r"(tests?/[^\s\"']+::[^\s\"']+|tests?/[^\s\"']+)")
_WRITE_TARGET_PATTERNS = (
    re.compile(r">\s*([^\s\"']+)"),
    re.compile(r">>\s*([^\s\"']+)"),
    re.compile(r"\btee\s+([^\s\"']+)"),
    re.compile(r"\b(?:cat|python|python3)\b.*?\b([A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|jsx|java|rb|go|rs|c|cc|cpp|h|hpp|yaml|yml|json|toml|ini|cfg|txt|md))\b"),
)
_FILE_TOKEN_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|jsx|java|rb|go|rs|c|cc|cpp|h|hpp|yaml|yml|json|toml|ini|cfg|txt|md))")


def _describe_command(command: str, *, task_kind: str, conservative: bool) -> str:
    if not conservative:
        return _truncate_command(command)
    summary = _build_safe_command_summary(command, task_kind=task_kind)
    if summary:
        return summary
    if task_kind == "unit_test":
        return "Run focused validation command"
    return "Run focused shell command"


def _build_safe_command_summary(command: str, *, task_kind: str) -> str:
    normalized = " ".join(command.strip().split())
    if not normalized:
        return ""
    if task_kind == "unit_test":
        test_target = _extract_test_target(normalized)
        if test_target:
            return f"Run focused test {test_target}"
        return "Run focused validation command"
    lower = normalized.lower()
    if _looks_like_search(lower):
        target = _extract_file_reference(normalized)
        return f"Inspect {target}" if target else "Inspect relevant code paths"
    if _looks_like_diff_or_status(lower):
        return "Review local workspace changes"
    write_target = _extract_write_target(normalized)
    if write_target:
        return f"Edit {write_target}"
    file_target = _extract_file_reference(normalized)
    if file_target:
        if _looks_like_view(lower):
            return f"Inspect {file_target}"
        return f"Inspect or update {file_target}"
    return ""


def _extract_test_target(command: str) -> str:
    match = _TEST_TARGET_RE.search(command)
    if match:
        return match.group(1)
    return ""


def _looks_like_search(command: str) -> bool:
    return any(token in command for token in (" rg ", " grep ", "find ", "sed -n", "awk "))


def _looks_like_diff_or_status(command: str) -> bool:
    return "git diff" in command or "git status" in command


def _looks_like_view(command: str) -> bool:
    return any(token in command for token in (" cat ", "sed -n", "head ", "tail ", "less ", "rg ", "grep "))


def _extract_write_target(command: str) -> str:
    for pattern in _WRITE_TARGET_PATTERNS:
        match = pattern.search(command)
        if match:
            target = match.group(1).strip().strip("\"'")
            if _looks_like_file_path(target):
                return target
    return ""


def _extract_file_reference(command: str) -> str:
    for match in _FILE_TOKEN_RE.finditer(command):
        candidate = match.group(1).strip().strip("\"'")
        if _looks_like_file_path(candidate):
            return candidate
    return ""


def _looks_like_file_path(value: str) -> bool:
    return "/" in value or "." in value
