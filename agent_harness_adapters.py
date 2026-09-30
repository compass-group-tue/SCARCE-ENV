"""Pluggable adapters for driving different agentic coding CLIs ("harnesses") as the solver
in run_programbench_auction_subrun_harness_agents.py, instead of mini-swe-agent's own minimal
bash-loop agent.

Why this exists: mini-swe-agent was the original solver because it's a deliberately thin,
single-bash-tool loop with exactly one choke point (environment.execute()) to intercept -- that's
what let TimeSlotLeaseEnvironment/SolverMemoryEnvironment gate every command through the auction
mechanism in the first place. Claude Code, Codex, and OpenCode are all much richer, more
autonomous agents with their own native tool ecosystems (file edit tools, planning, parallel tool
calls) that don't expose that same single hook natively. The integration strategy used here is:
run the harness with ALL of its own native tools disabled, and give it exactly three custom MCP
tools (request_execution_slot, run_command, save_persistent_memory) that internally call the
EXACT SAME .execute() interface mini-swe-agent already calls -- see
programbench_auction_mcp_server.py. This keeps every harness reduced to the same single-bash-tool
shape mini-swe-agent already has, so the interception logic itself never has to change per harness
-- only how each harness's own CLI gets launched and how its own output gets parsed differs, and
that's exactly what this file's HarnessAdapter interface isolates.

Adding a new harness later means implementing ONE new HarnessAdapter subclass here and registering
it in HARNESS_ADAPTERS -- nothing in the MCP server or the orchestrator script needs to change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Shared between agent_harness_adapters.py and programbench_auction_mcp_server.py: the MCP
# server's registered name (must match write_mcp_config's "mcpServers" key exactly, since
# Claude Code qualifies MCP tool names as mcp__<server_name>__<tool_name> when you scope
# --allowedTools to them) and the full set of tool names the server can expose. Kept here
# rather than duplicated in both files so a rename in one place can't silently desync them.
MCP_SERVER_NAME = "programbench_auction"
AUCTION_TOOL_NAMES = ["request_execution_slot", "run_command", "save_persistent_memory"]


def qualified_mcp_tool_names(*, enabled_tool_names: list[str]) -> list[str]:
    """Claude Code's own qualification scheme for MCP-provided tools, confirmed working via a
    live smoke test: mcp__<server_name>__<tool_name>. NOT a documented public API surface as
    far as this project can confirm; if this changes in a future Claude Code release, this is
    the one place to update it."""
    return [f"mcp__{MCP_SERVER_NAME}__{name}" for name in enabled_tool_names]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class HarnessRunResult:
    """Normalized result of one harness CLI invocation (one subrun's whole solving session)."""

    exit_status: str
    """One of: "Submitted" (the run_command MCP tool observed mini-swe-agent's own
    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT completion marker), "Stopped" (the harness process
    exited normally without ever submitting -- it just stopped acting), or an exception class
    name (e.g. "HardDeadlineExceededError") when something raised during the session."""
    raw_output: str
    cost_usd: float | None = None
    num_turns: int | None = None
    returncode: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    ended_without_action: bool = False
    """True when the session's own final turn produced neither a tool call nor the
    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT marker -- i.e. the model narrated an intended next step
    ("I will now explore...") and then simply stopped, rather than actually taking it or
    genuinely finishing. Confirmed live with gpt-oss-120b via PiAdapter: pi's own --print/
    single-session design exits the instant a turn produces no further tool call, with no
    external nudge to continue, silently discarding the rest of the subrun's time/budget even
    though the deadline clock and workspace state are both still intact. Only PiAdapter currently
    sets this meaningfully; every other adapter leaves it at the default False (an ordinary
    "Stopped" from a genuinely-idle or genuinely-finished session, not this specific failure
    mode)."""


class HarnessAdapter:
    """One implementation per agentic CLI. Stateless -- a single instance is reused across every
    worker/subrun; all per-session state (ports, working dirs, prompts) is passed as arguments."""

    name: str = "base"

    def build_command(
        self,
        *,
        mcp_config_path: Path,
        system_prompt: str,
        cwd: Path,
        model: str | None,
        cost_limit_usd: float | None,
        enabled_tool_names: list[str],
        extra_cli_args: list[str],
    ) -> list[str]:
        """Return the full argv for one non-interactive, single-session CLI invocation. Must
        run to completion and exit on its own (no interactive prompts) -- the orchestrator reads
        its stdout line-by-line as it runs (see run_harness_subprocess_streaming in the
        orchestrator script) rather than blocking on the whole thing like subprocess.run(...)
        would, specifically so parse_stream_event below has something to work with live.
        mcp_config_path points at a JSON file declaring this session's one MCP server (see
        programbench_auction_mcp_server.py) as the harness's only tool source; enabled_tool_names
        is the exact set of tool names actually registered on that server this session (varies --
        save_persistent_memory is conditional). The caller is responsible for restricting the
        harness to ONLY those tools -- see each adapter's own implementation for how that harness
        expresses that. IMPORTANT, confirmed the hard way (see ClaudeCodeAdapter's docstring): a
        flag that looks like "disable native tools" may not actually do that, and a flag that
        looks like it only affects permission PROMPTS may be the only thing that actually removes
        a tool from availability -- always verify empirically (does the target .execute() call
        actually get reached? is a supposedly-removed tool still genuinely callable?) before
        trusting a flag's documented description, for every harness added here."""
        raise NotImplementedError

    def parse_output(self, *, stdout: str, stderr: str, returncode: int) -> HarnessRunResult:
        """Turn whatever the CLI printed (its complete accumulated stdout, once the process has
        exited) into a normalized result. exit_status here is a FALLBACK only -- the
        orchestrator's own MCP server tracks the real exit_status (Submitted /
        HardDeadlineExceededError / etc.) directly from the .execute() calls it proxies, since
        that's the only place that information genuinely originates; this method's exit_status is
        only used if that per-session tracked value is somehow unavailable."""
        raise NotImplementedError

    def parse_stream_event(self, line: str) -> str | None:
        """Given ONE raw line of the CLI's stdout, read live while the process is still running,
        return a short human-readable summary to print to the console/run log, or None if this
        line isn't worth surfacing (e.g. blank, unparseable, or a housekeeping event). Base
        implementation always returns None -- an adapter that doesn't override this just runs
        with no live per-event visibility (parse_output still works fine on the accumulated
        output once the process exits; this method only affects what shows up WHILE it's
        running). Exists because a harness CLI run non-interactively can otherwise look
        completely silent for however long the whole subrun takes -- see ClaudeCodeAdapter's
        implementation, which was added specifically because a live run gave zero visibility into
        what the solver was doing turn-by-turn until the whole subrun ended."""
        return None

    def extract_incremental_usage(self, line: str) -> dict[str, int] | None:
        """Given ONE raw line of the CLI's stdout, return that single turn's OWN token usage
        (keys: input_tokens, output_tokens, cache_creation_input_tokens,
        cache_read_input_tokens -- any subset, missing keys treated as 0), or None if this line
        carries no usage data. Base implementation always returns None. Exists because the only
        place a harness's TOTAL cost/token usage is normally available is the terminal summary
        event once the whole session has already exited (see parse_output) -- but a live run
        showed every in-progress subrun's tokens/cost sitting at a flat 0 in the monitor/dashboard
        for its entire duration as a result, even though real work and real spend were happening
        the whole time. Each individual turn/message in the stream typically carries its OWN
        incremental usage already, well before the session ends; summing those as they arrive is
        what lets a mid-run snapshot show real, growing numbers instead of 0 until the very end."""
        return None

    def build_subprocess_env(
        self,
        *,
        base_env: dict[str, str],
        server_url: str,
        wall_time_limit_seconds: float | None,
    ) -> dict[str, str]:
        """Returns the environment for this harness's subprocess. Base implementation returns
        base_env unchanged. Exists because a harness may need something communicated via the
        environment rather than argv/a config file -- e.g. PiAdapter's bridge extension has no
        CLI-flag way to learn the MCP server's URL, so it reads it from an env var this method
        sets; ClaudeCodeAdapter overrides this for CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT (see that
        override's own docstring). server_url is HarnessMCPServer.url for this session; most
        adapters (anything that gets the URL via a config file instead, e.g. OpenCodeAdapter) can
        ignore it and rely on the base implementation."""
        del server_url, wall_time_limit_seconds
        return dict(base_env)


# Default 30 minutes: comfortably longer than a single request_execution_slot call should ever
# need to sit blocked (negotiation itself is designed to be free/deadline-paused, but the wait
# still burns real wall-clock time waiting on slow GLM-5.3/OpenRouter retries across the
# reasoning/declaration/discussion/ballot phases, confirmed live to occasionally exceed 5 minutes
# end-to-end for one round), while staying comfortably under this project's own
# wall_time_limit_seconds ceiling on the whole harness subprocess -- a genuinely runaway session
# still gets caught by that, not by relying on Claude Code's own idle timeout as the sole guard.
_DEFAULT_MCP_TOOL_IDLE_TIMEOUT_MS = 30 * 60 * 1000


class ClaudeCodeAdapter(HarnessAdapter):
    """Drives the `claude` CLI (Claude Code) in single-shot, non-interactive mode.

    Key flags this relies on (confirmed against `claude --help`, v2.1.212, AND against live
    `claude -p` smoke tests -- see the commit/session notes, not just the documented description):
      -p/--print                 non-interactive: run one full session, print, exit.
      --allowedTools <names>     scoped to exactly the qualified MCP tool names
                                  (mcp__<server>__<tool>, see qualified_mcp_tool_names).
      --disallowedTools <names>  THE ACTUAL RESTRICTION MECHANISM -- see CRITICAL note below.
                                  NOT --tools "". An early attempt used `--tools ""` on the
                                  (reasonable-sounding, but wrong in practice) theory that it
                                  disables the built-in tool set while leaving MCP tools alone.
                                  Empirically it did not: the model stopped making real tool
                                  calls entirely and started narrating fake
                                  "<function_calls>...</function_calls>" text instead.
      --mcp-config <path>        load the one MCP server exposing our auction tools.
      --strict-mcp-config        ignore any other MCP config the environment/user might have
                                  (project .mcp.json, user settings, etc.) -- this session must
                                  only ever see the one server we hand it.
      --system-prompt <text>     full replacement of the default system prompt.
      --dangerously-skip-permissions
                                  no interactive confirmation prompts -- required for unattended
                                  runs, same posture as mini-swe-agent's own --yolo.
      --output-format stream-json
                                  NDJSON, one event per line, as the session runs (system init,
                                  each assistant message/tool call, each tool result, then a final
                                  {"type": "result", "result": ..., "total_cost_usd": ...,
                                  "num_turns": ..., "is_error": ...} line) -- required so
                                  parse_stream_event has per-event visibility live, instead of a
                                  single JSON blob only available once the whole session ends (the
                                  --output-format json this used before gave literally zero
                                  console visibility into what the solver was doing for the whole
                                  duration of a subrun). Requires --verbose (the CLI refuses to
                                  start with --output-format=stream-json otherwise, confirmed via
                                  `claude -p --output-format stream-json "..."` alone erroring
                                  with "requires --verbose"). See parse_output/parse_stream_event.
      --max-budget-usd           built-in hard cost ceiling per session, when configured.

    CRITICAL, confirmed via a live run before this was fixed: `--allowedTools` scoped to just the
    MCP tool names does NOT hide native tools. A live subrun's solver plainly had Bash/Read/Edit/
    Write/Agent/Skill/Workflow/etc. available and, in that instance, refused to engage at all
    because it noticed the mismatch between its real toolset and the auction framing -- but
    nothing would have stopped it from just using Bash directly, completely bypassing the entire
    auction/negotiation mechanism this whole harness exists to gate. Root cause, confirmed by
    inspecting the raw stream-json `system.init` event's own "tools" field: with
    `--dangerously-skip-permissions` set, EVERY tool is usable without a permission prompt
    regardless of `--allowedTools` -- that flag apparently only affects what would otherwise
    require a confirmation prompt, not what's fundamentally callable. `--disallowedTools` is the
    one flag confirmed (via the same "tools" field, now genuinely empty) to actually remove tools
    from availability. NOTE: two tools (Glob, Grep) never appeared in the "tools" list from an
    UNRESTRICTED session at all, yet were still callable and had to be denied explicitly too --
    don't assume a single unrestricted system.init "tools" listing is the complete inventory of
    everything callable; if a future Claude Code release adds another such tool, this list needs
    a matching addition, and the way to notice is the same one used here: run an unrestricted
    session and try to get the model to use something not in DISALLOWED_NATIVE_TOOLS.
    """

    name = "claude-code"

    # Every native tool observed to be callable, confirmed via a real `system.init` stream-json
    # event's own "tools" field PLUS Glob/Grep (present in this Claude Code build, never listed in
    # an unrestricted session's own "tools" field, but confirmed independently reachable). Denying
    # all of these plus scoping --allowedTools to just the MCP tools was confirmed, via a live
    # subprocess-invoked test mirroring this exact production code path, to produce a session
    # where Bash is genuinely unavailable ("no matching deferred tool found") while the MCP tools
    # work normally.
    DISALLOWED_NATIVE_TOOLS = [
        "Task", "Bash", "CronCreate", "CronDelete", "CronList", "DesignSync", "Edit",
        "EnterWorktree", "ExitWorktree", "Monitor", "NotebookEdit", "PushNotification", "Read",
        "RemoteTrigger", "ReportFindings", "ScheduleWakeup", "SendMessage", "ShareOnboardingGuide",
        "Skill", "TaskCreate", "TaskGet", "TaskList", "TaskOutput", "TaskStop", "TaskUpdate",
        "WebFetch", "WebSearch", "Workflow", "Write", "Glob", "Grep",
    ]

    def build_command(
        self,
        *,
        mcp_config_path: Path,
        system_prompt: str,
        cwd: Path,
        model: str | None,
        cost_limit_usd: float | None,
        enabled_tool_names: list[str],
        extra_cli_args: list[str],
    ) -> list[str]:
        allowed = ",".join(qualified_mcp_tool_names(enabled_tool_names=enabled_tool_names))
        cmd = [
            "claude",
            "-p",
            "--mcp-config", str(mcp_config_path),
            "--strict-mcp-config",
            "--disallowedTools", ",".join(self.DISALLOWED_NATIVE_TOOLS),
            "--allowedTools", allowed,
            "--system-prompt", system_prompt,
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
        ]
        if model:
            cmd.extend(["--model", model])
        if cost_limit_usd is not None and cost_limit_usd > 0:
            cmd.extend(["--max-budget-usd", str(cost_limit_usd)])
        cmd.extend(extra_cli_args)
        # The actual user turn: with every real action routed through the MCP server, this
        # opening message is the only thing that kicks the session off -- the substantive task
        # framing already lives in system_prompt.
        cmd.append(
            "Begin working on your assigned reconstruction task now. Call request_execution_slot "
            "first, exactly as your instructions describe."
        )
        return cmd

    def parse_output(self, *, stdout: str, stderr: str, returncode: int) -> HarnessRunResult:
        # stdout is now NDJSON (--output-format stream-json), one event per line, not a single
        # JSON blob -- the terminal {"type": "result", ...} line carries the same fields the old
        # single-blob --output-format json used to return. Scan for it rather than assuming it's
        # the last line: a killed/crashed process can leave trailing partial or blank lines.
        result_payload: dict[str, Any] | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result_payload = event
        if result_payload is None:
            return HarnessRunResult(
                exit_status="Stopped",
                raw_output=stdout,
                returncode=returncode,
                extra={"stderr": stderr, "parse_error": "no stream-json 'result' event found"},
            )
        result_text = str(result_payload.get("result") or "")
        is_error = bool(result_payload.get("is_error"))
        return HarnessRunResult(
            exit_status="Stopped" if not is_error else "ClaudeCodeError",
            raw_output=result_text or stdout,
            cost_usd=result_payload.get("total_cost_usd"),
            num_turns=result_payload.get("num_turns"),
            returncode=returncode,
            extra={"raw_payload": result_payload, "stderr": stderr},
        )

    def parse_stream_event(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(event, dict):
            return None
        etype = event.get("type")

        if etype == "system" and event.get("subtype") == "init":
            mcp_names = [str(s.get("name")) for s in (event.get("mcp_servers") or [])]
            return f"session started (model={event.get('model')}, mcp_servers={mcp_names})"

        if etype == "assistant":
            parts = []
            for block in (event.get("message", {}).get("content") or []):
                if block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        parts.append(f'says "{_truncate(text, 300)}"')
                elif block.get("type") == "tool_use":
                    parts.append(f"calls {block.get('name')}({_truncate(json.dumps(block.get('input') or {}), 200)})")
            return " | ".join(parts) if parts else None

        if etype == "user":
            parts = []
            for block in (event.get("message", {}).get("content") or []):
                if block.get("type") == "tool_result":
                    content = block.get("content")
                    text = content if isinstance(content, str) else json.dumps(content)
                    parts.append(f"tool result: {_truncate(text, 300)}")
            return " | ".join(parts) if parts else None

        if etype == "result":
            cost = event.get("total_cost_usd")
            turns = event.get("num_turns")
            cost_text = f"${cost:.4f}" if isinstance(cost, (int, float)) else "unknown"
            return f"session ended (turns={turns}, cost={cost_text}, is_error={event.get('is_error')})"

        # rate_limit_event and anything else: not interesting enough to surface per-line.
        return None

    def extract_incremental_usage(self, line: str) -> dict[str, int] | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(event, dict) or event.get("type") != "assistant":
            return None
        # Confirmed via a live stream-json capture: each "assistant" event's own
        # message.usage carries THAT turn's own incremental token counts (separate from the
        # terminal "result" event's cumulative total for the whole session) -- not nested inside
        # content blocks, a sibling of "content"/"model" on the message object itself.
        usage = (event.get("message") or {}).get("usage")
        if not isinstance(usage, dict):
            return None
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        }

    def build_subprocess_env(
        self,
        *,
        base_env: dict[str, str],
        server_url: str,
        wall_time_limit_seconds: float | None,
    ) -> dict[str, str]:
        """A copy of base_env, plus CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT raised well past its
        5-minute default.

        Confirmed via anthropics/claude-code GitHub issue #70441 (v2.1.187+, this project's
        version is 2.1.212): remote/HTTP MCP tool calls (HarnessMCPServer uses HTTP transport, so
        this applies regardless of it being on localhost) that go 5 minutes with no response
        abort with an error -- completely separate from MCP_TOOL_TIMEOUT (a hard wall-clock
        ceiling defaulting to roughly 28 hours, irrelevant here). Without this override, a live
        run repeatedly showed request_execution_slot's tool_result coming back as the literal
        text "The operation timed out." once negotiation needed several slow repair-retries and
        pushed past 5 minutes, even though the call was still on track to genuinely resolve given
        more time -- confirmed this is the actual mechanism (not a code bug in this project) by
        reproducing the working case (single/concurrent calls under 150s succeeded every time)
        and finding the documented 5-minute idle-abort behavior, rather than guessing.
        """
        del server_url
        timeout_ms = _DEFAULT_MCP_TOOL_IDLE_TIMEOUT_MS
        if wall_time_limit_seconds is not None:
            timeout_ms = max(timeout_ms, int(wall_time_limit_seconds * 1000))
        env = dict(base_env)
        env["CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT"] = str(timeout_ms)
        return env


class CodexAdapter(HarnessAdapter):
    """NOT YET IMPLEMENTED. `codex exec` is Codex CLI's non-interactive subcommand (confirmed
    present via `codex --help` on this machine) and `codex mcp` manages MCP server config, but
    the exact flag names for disabling native tools, pointing at a one-off MCP config file, and
    supplying a full system-prompt replacement need to be confirmed against `codex exec --help`
    and Codex's own config file format before wiring this up -- don't assume the Claude Code
    flag names transfer. Implement by mirroring ClaudeCodeAdapter's structure once those are
    confirmed, then register an instance in HARNESS_ADAPTERS below."""

    name = "codex"

    def build_command(self, **kwargs: Any) -> list[str]:
        raise NotImplementedError(
            "CodexAdapter is a registered placeholder, not implemented yet. See this class's "
            "docstring for what to confirm from `codex exec --help` before implementing it."
        )

    def parse_output(self, **kwargs: Any) -> HarnessRunResult:
        raise NotImplementedError("CodexAdapter is a registered placeholder, not implemented yet.")


class OpenCodeAdapter(HarnessAdapter):
    """Drives the `opencode run` CLI (OpenCode, confirmed v1.18.18) in single-shot,
    non-interactive mode.

    Unlike Claude Code, OpenCode's CLI has NO flags equivalent to --mcp-config,
    --disallowedTools, or --system-prompt (confirmed against `opencode run --help`/
    `opencode --help`) -- all three concerns (registering the one MCP server, restricting native
    tools, and overriding the system prompt) live entirely in a project-local opencode.json that
    the CLI auto-discovers from its own cwd. That means, for this adapter, the REAL restriction
    mechanism is programbench_auction_mcp_server.write_mcp_config's "opencode" branch, not
    anything in build_command's argv -- see that function. build_command here only needs to
    launch the CLI and supply the opening user turn; mcp_config_path/system_prompt are accepted
    for interface parity with other adapters but not used to build argv (they were already baked
    into the auto-discovered config file by the time this runs).

    Key flags this relies on (confirmed against `opencode run --help` AND live
    `opencode run --format json --auto ...` smoke tests against a real HarnessMCPServer
    instance):
      run [message..]      non-interactive: one full session, print each event, exit.
      --format json         NDJSON, one event per line (step_start/tool_use/text/step_finish),
                             confirmed via a live capture. There is NO terminal "result" summary
                             event like Claude Code's -- see parse_output, which instead sums
                             every step_finish event's own "cost" and counts them for num_turns.
      --auto                 auto-approve permissions not explicitly denied -- required for
                             unattended runs, same posture as Claude Code's
                             --dangerously-skip-permissions.
      --model provider/model confirmed working with e.g. "azure/claude-sonnet-5".
      --agent build          explicit, even though `run` already defaults to the "build" agent --
                             the auto-discovered config's tool/prompt overrides are written under
                             agent.build (see write_mcp_config), so this is passed explicitly to
                             survive a future OpenCode default-agent change rather than relying on
                             an implicit default staying "build" forever.

    CRITICAL, confirmed via a live run: qualified MCP tool names use `{server_key}_{tool_name}`
    (e.g. "programbench_auction_request_execution_slot"), NOT Claude Code's
    `mcp__server__tool` -- confirmed by inspecting a real tool_use event's own "tool" field. Also
    confirmed live: setting agent.build.tools to false for every native tool id (see
    DISALLOWED_NATIVE_TOOLS) genuinely removes them -- a live run explicitly instructed to "try
    any shell/bash tool" made zero attempts at anything other than the MCP-provided tools, i.e.
    no native bash tool was even available to attempt. Also confirmed live: agent.build.prompt
    fully governs the system prompt actually used (a literal "you are a pirate, never break
    character" test prompt was followed exactly, with no sign of a conflicting baked-in default
    system prompt bleeding through) -- this rules out the same framing-mismatch failure mode
    documented in ClaudeCodeAdapter's docstring.

    No OpenCode CLI flag equivalent to Claude Code's --max-budget-usd was found in
    `opencode run --help` (v1.18.18); cost_limit_usd is accepted for interface parity but is NOT
    enforced for this harness.
    """

    name = "opencode"

    # Confirmed via a live `opencode debug agent build` (v1.18.18): the complete "tools" map the
    # default "build" primary agent resolves to before any MCP server is configured -- i.e. every
    # native tool id that needs to be explicitly set to false to remove it. If a future OpenCode
    # release adds another native tool, this list needs a matching addition; the way to notice is
    # the same one used here: run `opencode debug agent build` again and diff its "tools" keys.
    DISALLOWED_NATIVE_TOOLS = [
        "invalid", "question", "bash", "read", "glob", "grep", "edit", "write",
        "task", "webfetch", "todowrite", "skill",
    ]

    def build_command(
        self,
        *,
        mcp_config_path: Path,
        system_prompt: str,
        cwd: Path,
        model: str | None,
        cost_limit_usd: float | None,
        enabled_tool_names: list[str],
        extra_cli_args: list[str],
    ) -> list[str]:
        del mcp_config_path, system_prompt, cost_limit_usd  # see class docstring
        cmd = ["opencode", "run", "--format", "json", "--auto", "--agent", "build"]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(extra_cli_args)
        cmd.append(
            "Begin working on your assigned reconstruction task now. Call request_execution_slot "
            "first, exactly as your instructions describe."
        )
        return cmd

    def parse_output(self, *, stdout: str, stderr: str, returncode: int) -> HarnessRunResult:
        # No terminal "result" event exists in this stream (unlike Claude Code) -- cost and turn
        # count are derived by summing/counting every step_finish event instead, confirmed via a
        # live capture to be the only place per-step cost appears.
        total_cost = 0.0
        cost_seen = False
        num_turns = 0
        last_text: str | None = None
        is_error = False
        error_payload: Any = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype == "step_finish":
                num_turns += 1
                cost = (event.get("part") or {}).get("cost")
                if isinstance(cost, (int, float)):
                    total_cost += cost
                    cost_seen = True
            elif etype == "text":
                text = (event.get("part") or {}).get("text")
                if isinstance(text, str) and text.strip():
                    last_text = text
            elif etype == "error":
                is_error = True
                error_payload = event.get("error")
        return HarnessRunResult(
            exit_status="OpenCodeError" if is_error else "Stopped",
            raw_output=last_text if last_text is not None else stdout,
            cost_usd=total_cost if cost_seen else None,
            num_turns=num_turns or None,
            returncode=returncode,
            extra={"stderr": stderr, "error": error_payload} if is_error else {"stderr": stderr},
        )

    def parse_stream_event(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(event, dict):
            return None
        etype = event.get("type")
        part = event.get("part") or {}

        if etype == "text":
            text = str(part.get("text") or "").strip()
            return f'says "{_truncate(text, 300)}"' if text else None

        if etype == "tool_use":
            state = part.get("state") or {}
            return (
                f"calls {part.get('tool')}({_truncate(json.dumps(state.get('input') or {}), 200)}) "
                f"-> {state.get('status')}"
            )

        if etype == "step_finish":
            tokens = part.get("tokens") or {}
            cost = part.get("cost")
            cost_text = f"${cost:.4f}" if isinstance(cost, (int, float)) else "unknown"
            return f"step ended (reason={part.get('reason')}, cost={cost_text}, tokens={tokens.get('total')})"

        if etype == "error":
            return f"session error: {_truncate(json.dumps(event.get('error') or {}), 300)}"

        return None

    def extract_incremental_usage(self, line: str) -> dict[str, int] | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(event, dict) or event.get("type") != "step_finish":
            return None
        # Confirmed via a live capture: each step_finish's own part.tokens carries THAT step's own
        # incremental counts (not a running total), the same "sum as they arrive" shape Claude
        # Code's per-turn usage has.
        tokens = (event.get("part") or {}).get("tokens")
        if not isinstance(tokens, dict):
            return None
        cache = tokens.get("cache") or {}
        return {
            "input_tokens": int(tokens.get("input") or 0),
            "output_tokens": int(tokens.get("output") or 0),
            "cache_creation_input_tokens": int(cache.get("write") or 0),
            "cache_read_input_tokens": int(cache.get("read") or 0),
        }


# Pi (`@earendil-works/pi-coding-agent`, confirmed v0.74.2 on this machine) has NO native MCP
# support at all -- confirmed against its own --help and docs: "No MCP. Build CLI tools with
# READMEs (see Skills), or build an extension that adds MCP support." Unlike OpenCode (no MCP
# CLI flags either, but still speaks MCP natively once a server is in its config file), Pi has no
# MCP client of any kind built in. This TypeScript source is that missing MCP client: a Pi
# "Extension" (see https://pi.dev/docs/latest/extensions) whose factory function, on load, does
# ONE tools/list JSON-RPC call against the real HarnessMCPServer (URL from the
# PROGRAMBENCH_MCP_URL env var -- see PiAdapter.build_subprocess_env, the only way to hand a Pi
# extension a value with no CLI-flag/config-file channel available) and dynamically
# pi.registerTool()s each tool it finds, forwarding every call as its own tools/call JSON-RPC
# request. This means the extension needs NO knowledge of what tools exist or their schemas --
# confirmed live that Type.Unsafe(tool.inputSchema) (typebox's raw-JSON-schema passthrough)
# accepts the real MCP server's own JSON-schema inputSchema directly, so tool descriptions/schemas
# can never drift from the real server's own (the single source of truth used by every other
# adapter too).
#
# Confirmed live, via curl against a real HarnessMCPServer instance, that its streamable-http MCP
# endpoint is STATELESS: no Mcp-Session-Id handshake is required (an initialize call returns no
# such header), and a bare `tools/list`/`tools/call` POST with no prior call works standalone --
# this is what lets the fetch-based bridge below skip session/handshake bookkeeping entirely and
# just POST each request independently. The response body is SSE-formatted
# ("event: message\ndata: {...}"); parsing just the one "data:" line is sufficient since this
# server only ever emits a single message per request (confirmed live, not assumed).
#
# Confirmed live end-to-end (real HarnessMCPServer + real `pi` subprocess + this exact source,
# against a real Anthropic-messages-API-compatible endpoint): a tool call issued by the model
# correctly reached the real MCP server and the real result came back to the model; with
# --no-builtin-tools passed in PiAdapter.build_command, the model never attempted any native tool
# (bash/read/edit/write) even when explicitly told to try one -- only this bridge's own registered
# tools were ever available.
_PI_BRIDGE_EXTENSION_SOURCE = """\
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

async function callMcp(url: string, method: string, params: Record<string, unknown>): Promise<any> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json, text/event-stream" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  const text = await resp.text();
  const dataLine = text.split("\\n").find((line) => line.startsWith("data:"));
  if (!dataLine) {
    throw new Error(`programbench MCP bridge: no data line in response from ${url}: ${text.slice(0, 500)}`);
  }
  const payload = JSON.parse(dataLine.slice("data:".length).trim());
  if (payload.error) {
    throw new Error(`programbench MCP bridge: JSON-RPC error: ${JSON.stringify(payload.error)}`);
  }
  return payload.result;
}

export default async function (pi: ExtensionAPI) {
  const url = process.env.PROGRAMBENCH_MCP_URL;
  if (!url) {
    throw new Error("programbench MCP bridge: PROGRAMBENCH_MCP_URL is not set");
  }
  const listResult = await callMcp(url, "tools/list", {});
  for (const tool of listResult.tools as Array<{ name: string; description: string; inputSchema: unknown }>) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description,
      parameters: Type.Unsafe(tool.inputSchema),
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        try {
          const callResult = await callMcp(url, "tools/call", { name: tool.name, arguments: params });
          const text =
            callResult?.structuredContent?.result ?? callResult?.content?.[0]?.text ?? JSON.stringify(callResult);
          return { content: [{ type: "text", text }], details: {}, isError: Boolean(callResult?.isError) };
        } catch (err) {
          return {
            content: [{ type: "text", text: err instanceof Error ? err.message : String(err) }],
            details: {},
            isError: true,
          };
        }
      },
    });
  }
}
"""


class PiAdapter(HarnessAdapter):
    """Drives the `pi` CLI (Pi, `@earendil-works/pi-coding-agent`, confirmed v0.74.2) in
    single-shot, non-interactive mode.

    Pi has no native MCP support (see _PI_BRIDGE_EXTENSION_SOURCE's own comment for the full
    story and what's confirmed live) -- the real restriction/tool-exposure mechanism for this
    adapter is that generated extension file (written to mcp_config_path by
    programbench_auction_mcp_server.write_mcp_config's "pi" branch), not anything unusual in
    build_command's argv itself.

    Key flags this relies on (confirmed against `pi --help`, v0.74.2, AND live
    `pi --mode json --print --no-session ...` smoke tests against a real HarnessMCPServer
    instance, using a temporary custom Pi provider pointed at the same Azure Anthropic-compatible
    endpoint OpenCodeAdapter's own smoke test used):
      -e/--extension <path>   load the MCP-bridge extension file (repeatable; only one needed
                               here). Paired with --no-extensions so no other
                               auto-discovered/global extension can slip in.
      --no-extensions, -ne    disable auto-discovery of ~/.pi/agent/extensions and
                               .pi/extensions -- only the explicit -e path above is loaded.
      --no-builtin-tools, -nbt
                               disable Pi's own built-in tools (read/bash/edit/write/grep/find/
                               ls) while leaving extension-registered tools enabled -- confirmed
                               live: with this passed, the model made zero attempts at any native
                               tool even when explicitly told to try one, going straight to the
                               bridge's own tools instead.
      --no-skills, --no-prompt-templates, --no-themes, --no-context-files
                               disable every other auto-discovery surface (skills, prompt
                               templates, themes, AGENTS.md/CLAUDE.md) for the same isolation
                               reason as --no-extensions -- this session should see ONLY what
                               this adapter explicitly hands it.
      --system-prompt <text>  full replacement of the default system prompt, confirmed live via
                               a persona-override smoke test (a "you are a pirate, never break
                               character" prompt was followed exactly across many turns, with no
                               sign of a conflicting baked-in default bleeding through).
      --mode json              NDJSON, one event per line -- confirmed via a live capture the
                               richest of the three harnesses' event streams (session/agent_start/
                               turn_start/message_start/message_update/message_end/
                               tool_execution_start/tool_execution_end/turn_end/agent_end). See
                               parse_output/parse_stream_event/extract_incremental_usage for which
                               of these are actually used.
      --print, -p              non-interactive: run one full session, print, exit.
      --no-session             ephemeral: don't persist a session file for this one-off subrun.
      --model <pattern>        supports "provider/id"; whatever the user has authenticated (via
                               `pi`'s own /login or API-key env vars) determines what's actually
                               callable -- this adapter does not set up credentials itself.

    No `pi` CLI flag equivalent to Claude Code's --max-budget-usd was found in `pi --help`
    (v0.74.2); cost_limit_usd is accepted for interface parity but is NOT enforced for this
    harness.
    """

    name = "pi"

    def build_command(
        self,
        *,
        mcp_config_path: Path,
        system_prompt: str,
        cwd: Path,
        model: str | None,
        cost_limit_usd: float | None,
        enabled_tool_names: list[str],
        extra_cli_args: list[str],
        continuation_message: str | None = None,
    ) -> list[str]:
        # enabled_tool_names is unused here -- the bridge extension at mcp_config_path already
        # only ever registers whatever the real MCP server's own tools/list returns, which
        # already reflects the enabled subset (see HarnessMCPServer's own conditional
        # registration of save_persistent_memory). No separate --tools allowlist is needed on top
        # of --no-builtin-tools; confirmed live there's nothing else it would need to exclude.
        del cwd, cost_limit_usd, enabled_tool_names
        cmd = [
            "pi",
            "-e", str(mcp_config_path),
            "--no-extensions",
            "--no-builtin-tools",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--system-prompt", system_prompt,
            "--mode", "json",
            "--print",
            "--no-session",
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(extra_cli_args)
        # continuation_message, when given, replaces the normal opening prompt below -- used to
        # re-invoke this same session's task after a prior attempt ended with
        # HarnessRunResult.ended_without_action=True (see that field's own docstring). Every other
        # adapter leaves this at its default None; only this one currently exhibits the failure
        # mode it addresses.
        cmd.append(
            continuation_message
            if continuation_message is not None
            else (
                "Begin working on your assigned reconstruction task now. Call request_execution_slot "
                "first, exactly as your instructions describe."
            )
        )
        return cmd

    def build_subprocess_env(
        self,
        *,
        base_env: dict[str, str],
        server_url: str,
        wall_time_limit_seconds: float | None,
    ) -> dict[str, str]:
        # The bridge extension (_PI_BRIDGE_EXTENSION_SOURCE) reads PROGRAMBENCH_MCP_URL from its
        # own process.env at load time -- there is no CLI flag or config file for this adapter to
        # hand it the URL through instead.
        del wall_time_limit_seconds
        env = dict(base_env)
        env["PROGRAMBENCH_MCP_URL"] = server_url
        return env

    def parse_output(self, *, stdout: str, stderr: str, returncode: int) -> HarnessRunResult:
        # No single terminal summary event like Claude Code's "result" -- cost/usage/text are
        # confirmed live to live on each "turn_end" event's own message.usage/message.content,
        # analogous to OpenCode's per-step_finish shape. Summed/collected across every turn_end.
        last_text: str | None = None
        total_cost = 0.0
        cost_seen = False
        num_turns = 0
        is_error = False
        error_message: Any = None
        last_stop_reason: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict) or event.get("type") != "turn_end":
                continue
            num_turns += 1
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            cost = (usage.get("cost") or {}).get("total")
            if isinstance(cost, (int, float)):
                total_cost += cost
                cost_seen = True
            if message.get("stopReason") == "error":
                is_error = True
                error_message = message.get("errorMessage")
            # Overwritten every turn_end -- ends up holding the LAST turn's own stopReason, which
            # is exactly what ended_without_action below needs: "toolUse" means this final turn
            # queued another tool call (pi's loop would have kept going), "stop" means it didn't
            # and this really is where the whole session ended.
            last_stop_reason = message.get("stopReason")
            for block in message.get("content") or []:
                if block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"].strip():
                    last_text = block["text"]
        # See HarnessRunResult.ended_without_action's own docstring for the live incident this
        # detects: gpt-oss-120b, after winning a real execution window, ending its turn with plain
        # text describing an intended action instead of a tool call -- pi then exits the whole
        # session right there, with no further nudge. A genuine completion is a "stop" turn too,
        # but its text contains the marker; only the marker-less case is the failure this flags.
        ended_without_action = (
            last_stop_reason == "stop"
            and not is_error
            and (last_text is None or COMPLETE_TASK_MARKER not in last_text)
        )
        return HarnessRunResult(
            exit_status="PiError" if is_error else "Stopped",
            raw_output=last_text if last_text is not None else stdout,
            cost_usd=total_cost if cost_seen else None,
            num_turns=num_turns or None,
            returncode=returncode,
            extra={"stderr": stderr, "error": error_message} if is_error else {"stderr": stderr},
            ended_without_action=ended_without_action,
        )

    def parse_stream_event(self, line: str) -> str | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(event, dict):
            return None
        etype = event.get("type")

        if etype == "tool_execution_start":
            return f"calls {event.get('toolName')}({_truncate(json.dumps(event.get('args') or {}), 200)})"

        if etype == "tool_execution_end":
            result = event.get("result") or {}
            content = result.get("content")
            text = str(content[0].get("text") or "") if isinstance(content, list) and content else ""
            suffix = " [error]" if event.get("isError") else ""
            return f"{event.get('toolName')} -> {_truncate(text, 300)}{suffix}"

        if etype == "turn_end":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            cost = (usage.get("cost") or {}).get("total")
            cost_text = f"${cost:.4f}" if isinstance(cost, (int, float)) else "unknown"
            text = ""
            for block in message.get("content") or []:
                if block.get("type") == "text":
                    text = str(block.get("text") or "")
            summary = f"turn ended (stopReason={message.get('stopReason')}, cost={cost_text})"
            return f'{summary} says "{_truncate(text, 300)}"' if text else summary

        return None

    def extract_incremental_usage(self, line: str) -> dict[str, int] | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(event, dict) or event.get("type") != "turn_end":
            return None
        # Confirmed via a live capture: each turn_end's own message.usage carries that turn's own
        # final incremental counts (not a session-wide running total) -- the same "sum as they
        # arrive" shape Claude Code's/OpenCode's per-turn usage has.
        usage = (event.get("message") or {}).get("usage")
        if not isinstance(usage, dict):
            return None
        return {
            "input_tokens": int(usage.get("input") or 0),
            "output_tokens": int(usage.get("output") or 0),
            "cache_creation_input_tokens": int(usage.get("cacheWrite") or 0),
            "cache_read_input_tokens": int(usage.get("cacheRead") or 0),
        }


HARNESS_ADAPTERS: dict[str, HarnessAdapter] = {
    "claude-code": ClaudeCodeAdapter(),
    "codex": CodexAdapter(),
    "opencode": OpenCodeAdapter(),
    "pi": PiAdapter(),
}


def get_harness_adapter(name: str) -> HarnessAdapter:
    try:
        return HARNESS_ADAPTERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(HARNESS_ADAPTERS))
        raise ValueError(f"Unknown harness {name!r}. Available: {available}") from exc


# The COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT marker mini-swe-agent's own DockerEnvironment
# recognizes (see minisweagent/environments/docker.py's _check_finished) -- kept here, not
# imported, since it's a plain string constant with no reason to couple this module to
# minisweagent's package layout.
COMPLETE_TASK_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)
