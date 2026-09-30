"""Per-worker, per-subrun MCP server exposing the auction's execution interface as three tools
(request_execution_slot, run_command, save_persistent_memory) instead of mini-swe-agent's single
bash-command loop.

Deliberately thin: every tool call below does nothing except translate its typed arguments into
the EXACT SAME `env.execute({"command": ...})` call mini-swe-agent's own DockerEnvironment.execute
already makes, against the SAME TimeSlotLeaseEnvironment/SolverMemoryEnvironment wrapper chain
built in run_programbench_auction_subrun_harness_agents.py the identical way
run_programbench_auction_subrun_repo_agents.py builds it for mini-swe-agent. None of the
interception, negotiation-triggering, deadline-pausing, or budget logic lives here or is
duplicated here -- this module's only job is being a second caller of that existing, unmodified
machinery, reachable over MCP instead of a Python function call.

One instance of HarnessSession is constructed fresh per subrun (matching how TimeSlotLeaseEnvironment
itself is fresh per subrun in the mini-swe-agent path) and torn down when that subrun's harness
CLI invocation returns.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP

# The mcp package's own internal logger, completely separate from uvicorn's (already quieted via
# log_level="warning" in start() below) and from this project's own TeeLogger console output --
# at its default INFO level it logs "Processing request of type ..." for every single MCP
# protocol exchange (ListToolsRequest, ListPromptsRequest, ListResourcesRequest, CallToolRequest)
# and "Terminating session: None" every time one of those exchanges completes, since
# HarnessMCPServer runs with stateless_http=True (every request opens and immediately closes its
# own stateless session, hence the recurring "None"). None of that is an error -- it just means
# the server is being hit constantly, which is normal -- but it drowns out this project's own
# real per-event console logging (see agent_harness_adapters.py's parse_stream_event) once a run
# actually gets going, so it's quieted to WARNING here rather than left at its noisy default.
logging.getLogger("mcp").setLevel(logging.WARNING)

try:
    from minisweagent.exceptions import Submitted
except ImportError:  # pragma: no cover -- mini-swe-agent is a hard dependency elsewhere already
    class Submitted(Exception):  # type: ignore[no-redef]
        pass

from agent_harness_adapters import COMPLETE_TASK_MARKER

# Hard ceiling on what run_command ever returns to the solver -- confirmed live as the direct
# cause of a full harness-session crash: a solver ran a terminal-animation program (cmatrix)
# directly with no timeout/output bound, it ran for its entire ~180s execution window
# continuously repainting the screen, and the resulting multi-megabyte stream of raw ANSI/cursor
# escape codes got returned as this one tool call's output in full -- the very next API call,
# now needing to include that whole blob in context, failed outright (stopReason=error,
# cost=$0.0000 -- rejected before any generation, consistent with a context-length overflow).
# Nothing downstream ever recovers from that: the harness subprocess exits, and no cleanup path
# existed to un-stick whatever it was still holding (see AuctionScheduleBoard's own docstring for
# the matching fix on that side). Capping here is a blunter instrument than teaching the solver to
# bound its own commands (e.g. `timeout 2s`, `| head`), but it protects against ANY runaway-output
# command -- not just this specific program -- with no reliance on the model remembering to do so
# itself. 20,000 chars is generous for legitimate output (long `git log -p`, file dumps) while
# comfortably short of blowing any model's context window on its own.
_MAX_RUN_COMMAND_OUTPUT_CHARS = 20_000


def _truncate_run_command_output(output: str) -> str:
    if len(output) <= _MAX_RUN_COMMAND_OUTPUT_CHARS:
        return output
    omitted = len(output) - _MAX_RUN_COMMAND_OUTPUT_CHARS
    return (
        output[:_MAX_RUN_COMMAND_OUTPUT_CHARS]
        + f"\n\n[... output truncated: {omitted} more character(s) omitted. This command produced "
        "far more output than can usefully fit in context -- if you need more of it, redirect to a "
        "file and read/grep the specific part you need instead of dumping it all at once.]"
    )


@dataclass
class HarnessSessionState:
    """Tracks what actually happened during one subrun's harness session, read by the
    orchestrator after the CLI process exits to determine exit_status -- mirrors what
    mini-swe-agent's own agent.run() return value (info["exit_status"]) would have told it."""

    submitted: bool = False
    submission_text: str = ""
    error: BaseException | None = None
    error_type_name: str | None = None
    commands_run: int = 0
    last_output: str = ""
    # One entry per real command that reached _run_command_impl (request_execution_slot,
    # run_command, and save_persistent_memory all funnel through it) -- the orchestrator uses
    # this to synthesize a traj.json in the same shape collect_traj_metrics() (in
    # run_swebench_three_agents.py, reused unmodified) expects, so token/cost/activity-feed
    # tracking works for harness-driven subruns the same way it already does for mini-swe-agent
    # ones. Without this, every harness subrun's tokens/cost showed as 0 in the monitor/dashboard
    # -- confirmed via a live run -- because that traj.json had an empty "messages" list and no
    # "info.model_stats", both of which collect_traj_metrics reads to produce real numbers.
    command_history: list[dict[str, Any]] = field(default_factory=list)


class HarnessMCPServer:
    """Wraps one FastMCP instance bound to one worker's one subrun. `env` is the fully-built
    wrapper chain (SolverMemoryEnvironment(TimeSlotLeaseEnvironment(ScheduledEnvironment(...))),
    or a subset if solver memory / scheduling is disabled) -- exactly the same `env` variable
    run_programbench_instance_direct hands to mini-swe-agent's agent.run(), just handed to this
    server's tools instead.
    """

    def __init__(
        self,
        *,
        env: Any,
        repo_name: str,
        host: str = "127.0.0.1",
        port: int,
        enable_memory_tool: bool = False,
        on_command: "Any | None" = None,
        request_registry: "Any | None" = None,
        poll_head_start_seconds: float = 20.0,
    ) -> None:
        self._env = env
        self._repo_name = repo_name
        self.state = HarnessSessionState()
        self._enable_memory_tool = enable_memory_tool
        # How long request_execution_slot blocks server-side (on the background wait thread) before
        # returning a "still waiting, call again" poll reply -- see _request_execution_slot_impl.
        # Default 20s is the ONLY safe value for the Claude Code harness, whose MCP client has a
        # confirmed unfixed bug (anthropics/claude-code#49910) that drops the response of any tool
        # call that runs long, so the wrapper has to return frequently to stay clear of it. Pi and
        # OpenCode are NOT affected by that client bug, so they can safely block for much longer per
        # call -- which is the whole game cost-wise: every time this returns "still waiting", the
        # solver (a reasoning model, reprocessing its entire growing context) burns a full LLM turn
        # just to call the same tool again. At 20s a worker genuinely queued behind others for a few
        # minutes spends ~10-15 no-op reasoning turns per window it waits on; at ~240s that drops to
        # ~1. Blocking longer changes nothing about what the agent does or how the deadline ticks
        # (the real wait runs on the persistent background thread either way, re-polling is
        # idempotent, and only active negotiation is deadline-free) -- it just stops paying the LLM
        # per 20s of a wait it has no reason to participate in. Confirmed live: a pi run spent ~89%
        # of ALL solver turns on these "still waiting" polls.
        self._poll_head_start_seconds = float(poll_head_start_seconds)
        # Called (no args) after every command_history append, so the caller can push live
        # activity into the monitor/dashboard AS THE SUBRUN RUNS -- without this, a subrun's
        # commands only became visible once collect_traj_metrics read the FINAL traj.json after
        # the whole harness process exited, meaning the dashboard's "Solver Activity" panel (and
        # monitor_state.json's own recent_commands) stayed empty for the subrun's entire duration
        # (confirmed via a live run where every worker showed recent_commands=[] while running).
        self._on_command = on_command
        # A SolverRequestRegistry (see run_programbench_auction_subrun_repo_agents.py), optional
        # so this class stays usable without it -- lets request_execution_slot's own `context`
        # argument (see _request_execution_slot_impl) reach the negotiator's own prompt via
        # _negotiation_memory_provider, the same free-text channel the requested duration already
        # flows through. Set/cleared entirely from this class (see _request_execution_slot_impl),
        # not by _handle_request, which has no idea this parameter exists.
        self._request_registry = request_registry
        # Tracks the last poll-timeout reply text recorded to command_history (see
        # _request_execution_slot_impl's own poll-timeout branch) -- without this, a worker
        # still queued behind others produces NO command_history/recent_commands entries at
        # all until its first real command finally executes, since that branch used to just
        # `return` its reply directly with no _record_command call. Confirmed live: a 5-worker
        # run showed command_count=0 and an empty activity feed for every worker still waiting
        # its turn, even though it was alive and polling every ~20s -- indistinguishable from a
        # genuinely stuck/dead worker from the dashboard alone. Recording only on a text CHANGE
        # (not every identical poll) keeps command_history from filling with dozens of duplicate
        # "still negotiating" entries while still surfacing real progress through the queue (a
        # position number ticking down, or the negotiating->waiting-my-turn transition).
        self._last_recorded_poll_reply: str | None = None

        self.mcp = FastMCP(
            f"programbench-auction-{repo_name}",
            host=host,
            port=port,
            stateless_http=True,
        )
        self._register_tools()

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Background-wait state for request_execution_slot -- see _request_execution_slot_impl's
        # own docstring for why this exists at all (a known, unfixed Claude Code MCP client bug,
        # confirmed via anthropics/claude-code#49910, intermittently drops the response for any
        # tool call that runs long, and request_execution_slot's whole point is to block for
        # however long negotiation + queue wait takes -- potentially minutes). Guarded by
        # _pending_lock since MCP tool calls can in principle be dispatched concurrently.
        self._pending_lock = threading.Lock()
        self._pending_thread: threading.Thread | None = None
        self._pending_done = threading.Event()
        self._pending_result: dict[str, Any] | None = None
        # Guards _record_command (including the on_command callback) -- see stop()'s own comment
        # for why this exists: without it, mcp_server.stop() returning doesn't guarantee an
        # in-flight command's on_command callback (which itself writes a live traj.json snapshot)
        # has actually finished, letting it race the orchestrator's own final, authoritative
        # traj.json write that runs immediately after stop() returns and silently overwrite it
        # with a stale, incomplete one.
        self._command_lock = threading.Lock()

    def _record_command(self, command: str, output: str, returncode: int) -> None:
        # Held for the whole call, including on_command() -- see stop()'s own matching
        # acquire/release for why: it's what lets stop() guarantee no on_command callback is
        # still in flight (and thus none can race the orchestrator's own "final, authoritative"
        # traj.json write, which runs immediately after stop() returns) by the time it returns.
        with self._command_lock:
            self.state.command_history.append(
                {"command": command, "output": output, "returncode": returncode, "timestamp": time.time()}
            )
            if self._on_command is not None:
                try:
                    self._on_command()
                except Exception:  # noqa: BLE001 -- a monitor-update failure must never break the
                    # actual command result being returned to the solver; this callback is a
                    # side-effect for dashboard visibility, not part of the real command outcome.
                    pass

    def _run_command_impl(self, command: str) -> str:
        self.state.commands_run += 1
        try:
            result = self._env.execute({"command": command}, cwd="")
        except Submitted as exc:
            self.state.submitted = True
            submission = ""
            try:
                # mini-swe-agent's InterruptAgentFlow.__init__ stores its message dicts in
                # .messages and calls super().__init__() with NO arguments -- .args is always
                # empty (), confirmed by direct inspection. An earlier version of this read
                # exc.args[0] and silently got "" every time as a result.
                messages = getattr(exc, "messages", None) or exc.args
                submission = str(messages[0].get("content", "")) if messages else ""
            except Exception:
                submission = str(exc)
            self.state.submission_text = submission
            reply = (
                "Task submitted successfully -- your COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT "
                "marker was recognized and the submission was recorded. You do not need to call "
                "any more tools; you may end your turn now."
            )
            self._record_command(command, reply, 0)
            return reply
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is the one place
            # that has to catch EVERYTHING (HardDeadlineExceededError, scheduler withdrawals,
            # anything else env.execute() can raise) so it can be reported to the orchestrator
            # after the harness process exits, exactly like run_programbench_instance_direct's
            # own except Exception clause does for mini-swe-agent's agent.run().
            self.state.error = exc
            self.state.error_type_name = type(exc).__name__
            reply = f"Command failed: {type(exc).__name__}: {exc}"
            self._record_command(command, reply, 1)
            return reply
        output = _truncate_run_command_output(str(result.get("output") or ""))
        self.state.last_output = output
        self._record_command(command, output, 0)
        return output

    # Defensive cap on request_execution_slot's own `context` argument -- see its own use below.
    # Not a security boundary (this is the solver's own free text, already fully trusted the same
    # way run_command's own command string is), just bloat control: without a cap, nothing stops
    # a very long context from repeating in every negotiator prompt for the whole round.
    _MAX_CONTEXT_CHARS = 600

    def _request_execution_slot_impl(
        self, seconds: float, context: str = "", _is_stale_active_refresh: bool = False
    ) -> str:
        """Poll-based wrapper around the exact same env.execute("request_execution_slot N")
        call _run_command_impl already makes for run_command/save_persistent_memory -- used ONLY
        for request_execution_slot, since that's the one call that can legitimately block for
        minutes (through negotiation and however many workers are queued ahead of it), which
        run_command/save_persistent_memory never do (they're only ever called once a window is
        already active, or are instant, respectively).

        context, when given on the FIRST call for a request (ignored on repeat polls, same as
        `seconds` -- see the "still waiting" branch below), is stored in the shared
        SolverRequestRegistry so _negotiation_memory_provider can fold it into THIS worker's own
        negotiator's prompt for the round that request triggers -- letting the solver hand the
        negotiator whatever it judges relevant (why this window matters, what it's for, anything
        that might help the negotiator make a stronger or more creative case) instead of the
        negotiator having only a bare number to work with.

        Why this isn't just a plain blocking call like the others: a confirmed, unfixed Claude
        Code MCP client bug (anthropics/claude-code#49910, closed as "not planned") intermittently
        drops the response for ANY tool call that runs long, regardless of the server responding
        correctly -- a live run repeatedly hit this as "transport dropped mid-call; response ...
        was lost" once negotiation needed slow retries. There is no server-side fix for a
        client-side bug; the actual mitigation is to stop making one long call at all. The real,
        unchanged, synchronous TimeSlotLeaseEnvironment.execute() call now runs on a background
        thread; this method itself only ever waits up to a short head start before returning
        either the real result (if it finished in time -- the common case, since most rounds
        resolve well under that) or a "still waiting, call again" message. Each individual poll
        call is short enough to sit safely clear of where the bug tends to fire, and a poll
        response getting dropped is harmless (retrying a poll is idempotent -- no state is lost)
        for the intermediate "still waiting" responses.

        That idempotence claim used to be FALSE for the one response that actually matters most:
        the terminal (granted/submitted/error) result, which is exactly the one most likely to
        need retrying (it can arrive after long negotiation -- the highest-risk window for the
        drop bug to fire). The old code cleared self._pending_result the instant it was read, so
        if THAT specific response got dropped, the client's only instructed recovery (call
        request_execution_slot again) would see self._pending_thread back at None and start an
        entirely new background thread -- a real, duplicate `request_execution_slot N` command
        into the underlying TimeSlotLeaseEnvironment, even though the first one already succeeded.
        The same unconditional clear also meant two genuinely concurrent callers racing the same
        self._pending_done.wait() would have the second one read result=None and hit a bare
        assert, crashing instead of getting a graceful answer.

        Fixed by never clearing self._pending_result here at all -- it's cached as "the last
        terminal answer for the current request" until something can actually prove the caller
        consumed it. The only such proof available is a successful run_command call (see that
        tool's own implementation below), which only makes sense to call once a window was
        genuinely granted -- so that's where this cache gets cleared, allowing a later, genuinely
        NEW request_execution_slot call (once that window is used up) to start a fresh wait. Until
        then, this method safely replays the same cached result to any retry or concurrent caller.
        """
        POLL_HEAD_START_SECONDS = self._poll_head_start_seconds

        with self._pending_lock:
            if self._pending_thread is None and self._pending_result is None:
                self._pending_done.clear()

                if context and self._request_registry is not None:
                    self._request_registry.set_requested_context(
                        self._repo_name, context[: self._MAX_CONTEXT_CHARS]
                    )

                def _background_wait() -> None:
                    command = f"request_execution_slot {seconds:.0f}"
                    try:
                        result = self._env.execute({"command": command}, cwd="")
                        self._pending_result = {"kind": "ok", "command": command, "output": str(result.get("output") or "")}
                    except Submitted as exc:
                        submission = ""
                        try:
                            messages = getattr(exc, "messages", None) or exc.args
                            submission = str(messages[0].get("content", "")) if messages else ""
                        except Exception:
                            submission = str(exc)
                        self._pending_result = {"kind": "submitted", "command": command, "submission": submission}
                    except Exception as exc:  # noqa: BLE001 -- same broad-catch posture as
                        # _run_command_impl's own except Exception clause, for the same reason:
                        # this must never crash the background thread silently.
                        self._pending_result = {"kind": "error", "command": command, "error": exc}
                    finally:
                        self._pending_done.set()

                self._pending_thread = threading.Thread(target=_background_wait, daemon=True)
                self._pending_thread.start()

        finished = self._pending_done.wait(timeout=POLL_HEAD_START_SECONDS)
        if not finished:
            # _handle_request's own wait_for_turn call (see its docstring) sets this once
            # negotiation has ALREADY resolved and this repo is just waiting its decided position
            # -- a state the generic "still negotiating" text below is simply false for. Check it
            # every time this branch is hit so a request that starts out genuinely negotiating and
            # later resolves gets an accurate reply on its very next poll, not just its first one.
            wait_status = (
                self._request_registry.get_wait_for_turn_status(self._repo_name)
                if self._request_registry is not None
                else None
            )
            if wait_status is not None:
                holder = wait_status.get("holder_repo_slug")
                turn_phrase = (
                    "you are next up -- your window should activate any moment"
                    if wait_status.get("position") == 1
                    else (
                        f"waiting for your turn: position {wait_status.get('position')} of "
                        f"{wait_status.get('total')}" + (f" (currently held by {holder})" if holder else "")
                    )
                )
                reply = (
                    f"Negotiation for this round already resolved -- {turn_phrase}. This wait is "
                    "expected and not a failure; it just has no finer-grained progress to report "
                    "than your queue position. The `seconds` value on this call is ignored -- your "
                    "original request is the one being tracked. You MUST keep calling "
                    "request_execution_slot (any seconds value) until it returns a real granted "
                    "result instead of this message. Do not call run_command yet."
                )
            else:
                reply = (
                    "Still negotiating / waiting for your turn -- this request has NOT failed, it "
                    "simply isn't resolved yet (this is normal for a contested round or if other "
                    "workers are queued ahead of you; it can legitimately take a while). The `seconds` "
                    "value on this call is ignored -- your original request is still the one being "
                    "negotiated. You MUST keep calling request_execution_slot (any seconds value) "
                    "until it returns a real granted/rejected result instead of this message -- "
                    "stopping now would abandon a request that may still succeed in the background "
                    "with nobody there to use the window it grants. Do not call run_command yet."
                )
            if reply != self._last_recorded_poll_reply:
                self._last_recorded_poll_reply = reply
                self._record_command(f"request_execution_slot {seconds:.0f}", reply, 0)
            return reply

        with self._pending_lock:
            result = self._pending_result
            # Deliberately NOT clearing self._pending_result here anymore -- see this method's
            # own docstring. It stays cached until run_command successfully consumes it (or a
            # fresh background wait overwrites it), so a dropped response's retry -- or a second,
            # genuinely concurrent caller racing this same read -- gets the same real answer
            # again instead of either triggering a duplicate request_execution_slot command or
            # crashing on a None read.
            self._pending_thread = None
        if self._request_registry is not None:
            self._request_registry.clear_requested_context(self._repo_name)
        # Reset so the NEXT fresh request_execution_slot cycle (once this window is used up)
        # always records its own first poll-timeout reply, rather than having it suppressed as
        # a false "duplicate" of this now-finished request's last recorded poll text.
        self._last_recorded_poll_reply = None

        if result is None:
            # Should not happen in practice (self._pending_done is only ever .set() inside
            # _background_wait's own finally, by which point self._pending_result was already
            # assigned in the try/except above it) -- but this used to be a bare assert, and a
            # crash here is strictly worse than a message asking the model to just poll again.
            return (
                "Still negotiating / waiting for your turn -- this request has NOT failed, it "
                "simply isn't resolved yet. Call request_execution_slot again to check on it."
            )
        if result["kind"] == "submitted":
            self.state.submitted = True
            self.state.submission_text = result["submission"]
            reply = (
                "Task submitted successfully -- your COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT "
                "marker was recognized and the submission was recorded. You do not need to call "
                "any more tools; you may end your turn now."
            )
            self._record_command(result["command"], reply, 0)
            return reply
        if result["kind"] == "error":
            exc = result["error"]
            self.state.error = exc
            self.state.error_type_name = type(exc).__name__
            reply = f"Command failed: {type(exc).__name__}: {exc}"
            self._record_command(result["command"], reply, 1)
            return reply
        output = result["output"]
        # "You already have an active execution window with ~Ns remaining" (see
        # TimeSlotLeaseEnvironment.execute()'s STATE 1 branch) is NOT a terminal result the way a
        # genuine grant or submission is -- it's a cheap, fast, time-sensitive status check whose
        # own remaining-seconds value goes stale the instant real time passes. The cache above
        # exists specifically to protect a genuine terminal answer from being lost to a dropped
        # response or duplicated by a retry -- neither concern applies here, and blindly replaying
        # it does real harm: confirmed live, a solver that keeps calling request_execution_slot
        # instead of run_command saw the identical "~4s remaining" text, verbatim, for 60+ calls
        # over 5 minutes -- a real, billed API call each time -- because nothing ever cleared the
        # cache (only a successful run_command does that) and nothing ever recomputed the actual,
        # by-then-long-expired remaining time. Detected by prefix rather than a dedicated result
        # kind to avoid a wider refactor of _background_wait's own return shape for what is, after
        # all, just a fast re-check -- safe to redo synchronously, unlike the slow negotiation path
        # this caching exists for.
        if output.startswith("You already have an active execution window") and not _is_stale_active_refresh:
            # Recurse through the normal path (clear the cache, let the top of this method start
            # a fresh background thread) rather than calling self._env.execute(...) directly here
            # -- STATE 1 itself is fast, but a direct synchronous call has no guard against the
            # rare case where the window has JUST expired between the original check and this one,
            # which would fall through into STATE 2's real (potentially slow) negotiation path
            # running on the MCP request-handling thread instead of a background one -- exactly
            # the long-running-tool-call problem this whole background-thread design exists to
            # avoid. Recursing keeps that guarantee intact for every code path, not just the
            # common one.
            #
            # _is_stale_active_refresh bounds this to exactly one extra round-trip: the refreshed
            # check below is itself very likely to ALSO land on "still active" (correctly, with a
            # smaller remaining-seconds value) for any window with real time left -- without the
            # flag, that would recurse again, and again, once per poll's worth of remaining
            # window, instead of just returning the one freshly-correct answer the caller asked
            # for.
            with self._pending_lock:
                self._pending_result = None
            return self._request_execution_slot_impl(seconds, context, _is_stale_active_refresh=True)
        # Same bug class as the STATE-1 case just above, hitting FOUR different
        # TimeSlotLeaseEnvironment._handle_request outputs that are explicit "please retry, this
        # is not terminal" instructions: "Lost your queued position unexpectedly", "No new
        # negotiation round resolved for your request", "Your request wasn't reflected in this
        # round's resolved schedule", and "Skipping the queued wait for position ..." (the
        # MIN_VIABLE_WINDOW_AFTER_WAIT_SECONDS withdrawal case -- the caller is EXPLICITLY told
        # "Call request_execution_slot again with a smaller ask, or accept an earlier/cheaper
        # position next round," just like the other three). Each one already tells the caller to
        # call request_execution_slot again -- but with self._pending_result cached and never
        # cleared except by a successful run_command (see this method's own docstring), that
        # instruction was a lie: every subsequent poll just replayed the exact same cached text
        # forever, without ever starting a new _handle_request() attempt, since the
        # top-of-function `self._pending_thread is None and self._pending_result is None` check
        # never passed again. Confirmed live: a worker whose wait_for_turn call was abandoned by
        # its own backstop kept obediently calling request_execution_slot exactly as instructed,
        # thousands of times over more than an hour, and never made a single real second attempt
        # -- every call just replayed the identical cached string. The fourth case (this comment's
        # own addition) was found the same way, independently, later: a worker silently vanished
        # from the communication pool for 7+ hours while still polling every ~20-30s, always
        # replaying the identical "Skipping the queued wait for position 4" text with an
        # unchanging deadline figure -- proof no fresh _handle_request() call was ever actually
        # happening, just this same cached reply on repeat.
        #
        # Unlike STATE-1's fix, this does NOT recurse inline: _handle_request's retry path can
        # itself be genuinely slow (a fresh negotiation/queue wait), which is exactly the
        # long-running-call problem the whole background-thread design exists to keep off the MCP
        # request-handling thread. Clearing the cache and returning (not recursing) lets the
        # model's own next natural poll -- which the message already tells it to make -- start a
        # real fresh background thread through the normal top-of-function path instead.
        if (
            output.startswith("Lost your queued position unexpectedly")
            or output.startswith("No new negotiation round resolved for your request")
            or output.startswith("Your request wasn't reflected in this round's resolved schedule")
            or output.startswith("Skipping the queued wait for position")
        ):
            with self._pending_lock:
                self._pending_result = None
            self._record_command(result["command"], output, 0)
            return output
        self.state.last_output = output
        self._record_command(result["command"], output, 0)
        return output

    def _register_tools(self) -> None:
        server = self

        @server.mcp.tool(
            name="request_execution_slot",
            description=(
                "Request an execution window of approximately N seconds for a specific next "
                "subtask. This is the ONLY way to gain the ability to run ordinary commands -- "
                "you cannot call run_command until this returns a granted window. This call does "
                "NOT block for the full wait -- if negotiation/your queue position isn't resolved "
                "within a short head start, it returns a 'still waiting' message instead, and you "
                "MUST call request_execution_slot again (repeatedly, until it stops saying that) "
                "to keep checking; the `seconds` and `context` values are only used on the FIRST "
                "call for a given request and are ignored on repeat/poll calls. This negotiation "
                "itself is free and does not count against your deadline; queued waiting for your "
                "turn afterward does. May grant less time than requested, and can never exceed "
                "300 seconds regardless of what you ask for. `context` (optional) is passed "
                "directly into your own negotiator's prompt for this round -- use it to share "
                "whatever you judge relevant to making a stronger or more creative case for this "
                "specific request (what the window is actually for, why it matters right now, "
                "anything else worth the negotiator knowing that a bare number doesn't convey). "
                "Leave it empty for a routine request with nothing worth adding."
            ),
        )
        def request_execution_slot(seconds: float, context: str = "") -> str:
            return server._request_execution_slot_impl(seconds, context)

        @server.mcp.tool(
            name="run_command",
            description=(
                "Run one ordinary shell command inside your assigned container. Only works while "
                "you hold an active execution window from request_execution_slot -- otherwise "
                f"this returns a rejection, not an error. When you have fully solved and verified "
                f"the task, run a command whose output begins with the literal line "
                f"{COMPLETE_TASK_MARKER} followed by your final submission content, exactly as "
                "your task instructions describe, to submit."
            ),
        )
        def run_command(command: str) -> str:
            # Defensive guard, not just a solver-instruction request: while a request_execution_slot
            # background wait is still unresolved (see _request_execution_slot_impl), the real
            # TimeSlotLeaseEnvironment.execute() call it started is still running on its own
            # thread -- letting run_command reach the SAME env concurrently from a second thread
            # would be genuinely untested territory (the whole stack was built assuming one
            # caller at a time per subrun), so this is rejected outright rather than risked.
            # Read under the same lock every mutation of _pending_thread uses (creation in
            # _request_execution_slot_impl's own "start fresh" branch, reset once its wait
            # concludes) -- reading it unlocked left a narrow window where this guard could
            # observe a stale None right after _request_execution_slot_impl had entered its
            # critical section but before it had actually assigned the new background thread.
            with server._pending_lock:
                pending = server._pending_thread is not None
            if pending:
                return (
                    "Rejected: a request_execution_slot call is still being negotiated in the "
                    "background. Call request_execution_slot again (not run_command) to check on "
                    "it -- this command was not executed."
                )
            result = server._run_command_impl(command)
            # Consuming the granted window: clear request_execution_slot's cached terminal result
            # (see that method's own docstring) now that something has actually proven the client
            # received and acted on it -- this is what allows a LATER, genuinely new
            # request_execution_slot call (once this window runs out) to start a fresh wait
            # instead of replaying this same now-stale cached answer forever.
            with server._pending_lock:
                server._pending_result = None
            return result

        if self._enable_memory_tool:
            @server.mcp.tool(
                name="save_persistent_memory",
                description=(
                    "Write a note to persistent memory that survives past this subrun -- it will "
                    "be shown to you again, verbatim, at the start of your NEXT subrun. Use it to "
                    "record a lesson, technique, or environment detail worth remembering, not as "
                    "a substitute for actually finishing the current subrun's work."
                ),
            )
            def save_persistent_memory(text: str) -> str:
                heredoc_command = "save_persistent_memory <<'MEMORY_EOF'\n" + text + "\nMEMORY_EOF"
                return server._run_command_impl(heredoc_command)

    def start(self) -> None:
        """Starts the HTTP MCP server in a background thread with its own asyncio event loop.
        Blocks until the server is actually accepting connections (bounded wait) so callers never
        race the harness CLI against a server that isn't listening yet."""
        app = self.mcp.streamable_http_app()
        config = uvicorn.Config(app, host=self.mcp.settings.host, port=self.mcp.settings.port, log_level="warning")
        self._server = uvicorn.Server(config)

        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _serve() -> None:
                assert self._server is not None
                config_task = asyncio.create_task(self._server.serve())
                # uvicorn.Server sets .started once its socket is bound and it's ready to accept
                # connections -- poll briefly rather than a fixed sleep, since real startup time
                # varies with system load.
                for _ in range(200):
                    if self._server.started:
                        break
                    await asyncio.sleep(0.01)
                ready.set()
                await config_task

            loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=10.0):
            raise RuntimeError(f"MCP server for {self._repo_name} did not start within 10s.")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        # Barrier, not a real critical section: acquiring and immediately releasing
        # _command_lock guarantees any _record_command call (and its on_command callback) that
        # was already in flight when should_exit was set has now fully completed -- since the
        # server thread just joined, no NEW command can start after this point either. Without
        # this, a caller of stop() (see run_programbench_auction_subrun_harness_agents.py's
        # finally block) that immediately writes its own "final, authoritative" traj.json snapshot
        # right after stop() returns had no guarantee a still-in-flight on_command callback's own,
        # earlier-snapshot traj.json write wouldn't land AFTER it, silently clobbering the real
        # final data with a stale one.
        with self._command_lock:
            pass

    @property
    def url(self) -> str:
        return f"http://{self.mcp.settings.host}:{self.mcp.settings.port}{self.mcp.settings.streamable_http_path}"


def write_mcp_config(
    *,
    server_url: str,
    config_path: "Any",
    harness_name: str = "claude-code",
    enabled_tool_names: "list[str] | None" = None,
    system_prompt: str = "",
) -> None:
    """Writes the JSON file a harness's --mcp-config-equivalent flag points at (or, for a harness
    with no such flag, the config file it auto-discovers from its own cwd instead). The server
    key here MUST stay MCP_SERVER_NAME (imported from agent_harness_adapters, not redefined here)
    -- both Claude Code's and OpenCode's own MCP tool-name qualification schemes are built from
    this key, confirmed via live smoke tests for each, so this key and
    qualified_mcp_tool_names()/OpenCodeAdapter's own qualification must never drift apart.

    harness_name selects the config SHAPE:
      "claude-code" (default): Claude Code's documented format (a "mcpServers" map). Tool
        restriction and the system prompt are handled entirely via CLI flags in
        ClaudeCodeAdapter.build_command instead, so enabled_tool_names/system_prompt are unused
        here for this branch.
      "opencode": OpenCode has no CLI flag equivalent to Claude Code's --mcp-config/
        --disallowedTools/--system-prompt at all (confirmed against `opencode run --help`) -- ALL
        THREE concerns live in this one file instead, which OpenCode auto-discovers from its own
        session cwd (config_path is expected to be that cwd's "opencode.json", confirmed working
        via a live `opencode run` smoke test). enabled_tool_names is used to build the
        agent.build.tools map (every native tool id in OpenCodeAdapter.DISALLOWED_NATIVE_TOOLS set
        false, every enabled MCP tool name set true under its qualified
        "{MCP_SERVER_NAME}_{tool_name}" form); system_prompt is written verbatim as
        agent.build.prompt, confirmed live to fully govern the session's actual system prompt with
        no baked-in default bleeding through.
      "pi": Pi has NO native MCP support at all (confirmed against its own --help/docs) -- this
        writes a static TypeScript Pi extension (agent_harness_adapters._PI_BRIDGE_EXTENSION_SOURCE
        -- see that constant's own comment for the full story) that acts as a minimal MCP client
        bridging Pi's tool-call mechanism to this real MCP server. The content is identical across
        every call (it discovers the server's URL from an env var at runtime, not from anything
        baked in at write time -- see PiAdapter.build_subprocess_env), so enabled_tool_names/
        system_prompt/server_url are all unused for this branch; system_prompt instead goes
        through PiAdapter.build_command's own --system-prompt CLI flag.
    """
    import json

    from agent_harness_adapters import MCP_SERVER_NAME, OpenCodeAdapter, _PI_BRIDGE_EXTENSION_SOURCE

    if harness_name == "claude-code":
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        MCP_SERVER_NAME: {
                            "type": "http",
                            "url": server_url,
                        }
                    }
                },
                indent=2,
            )
        )
        return

    if harness_name == "opencode":
        tools = {name: False for name in OpenCodeAdapter.DISALLOWED_NATIVE_TOOLS}
        for tool_name in enabled_tool_names or []:
            tools[f"{MCP_SERVER_NAME}_{tool_name}"] = True
        config_path.write_text(
            json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "mcp": {
                        MCP_SERVER_NAME: {
                            "type": "remote",
                            "url": server_url,
                            "enabled": True,
                            # Generous, matching build_claude_code_harness_env's own reasoning:
                            # request_execution_slot can legitimately block for several minutes
                            # during negotiation, and a short default MCP call timeout would abort
                            # it mid-flight. Units (ms vs s) not confirmed against OpenCode's own
                            # docs, but this value is safe under either reading.
                            "timeout": 30 * 60 * 1000,
                        }
                    },
                    "agent": {"build": {"tools": tools, "prompt": system_prompt}},
                },
                indent=2,
            )
        )
        return

    if harness_name == "pi":
        config_path.write_text(_PI_BRIDGE_EXTENSION_SOURCE)
        return

    raise ValueError(f"write_mcp_config: unsupported harness_name {harness_name!r}")
