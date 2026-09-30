from __future__ import annotations

import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import litellm
litellm.drop_params = True  # gpt-5 family (e.g. azure/gpt-5.4) rejects temperature!=1 and frequency_penalty; drop only affects providers that reject a param, so no effect on openrouter/glm runs.
import yaml


class LoggerLike(Protocol):
    def log(self, message: str = "") -> None: ...


DEFAULT_PROMPT_CONFIG_PATH = Path(__file__).with_name("queue_negotiation_prompts.yaml")
_DATACLASS_SLOTS_KW: dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **_DATACLASS_SLOTS_KW)
class NegotiationCandidate:
    task_id: int
    repo_name: str
    repo_slug: str
    priority: str
    task_kind: str
    instance_id: str | None
    description: str
    announced_for_seconds: float
    communication_wait_seconds: float
    deadline_minutes: int
    deadline_remaining_seconds: float
    memory_summary: str = ""


@dataclass(**_DATACLASS_SLOTS_KW)
class NegotiationOutcome:
    ordered_task_ids: list[int]
    message_by_task_id: dict[int, str]
    rationale_by_task_id: dict[int, str]
    declaration_by_task_id: dict[int, str]
    declaration_reasoning_summary_by_task_id: dict[int, str]
    reasoning_by_task_id: dict[int, str]
    reasoning_reasoning_summary_by_task_id: dict[int, str]
    discussion_by_task_id: dict[int, str]
    ballot_by_task_id: dict[int, str]
    ballot_reasoning_summary_by_task_id: dict[int, str]
    coordination_decision_by_task_id: dict[int, dict[str, Any]]
    transcript: list[dict[str, Any]]
    # Final, binding bid per candidate after any response-phase revisions -- may differ from
    # what declaration_by_task_id's literal DESIRED_DURATION_SECONDS/BID_PRICE text says if that
    # candidate revised its own bid during discussion (see _run_response_round). Callers that
    # read the resolved schedule should prefer these over parsing declaration text. Defaults to
    # 0.0/0 (same "no bid" posture declaration parsing already uses) for a candidate that never
    # declared a duration/price and never revised one either.
    final_duration_by_task_id: dict[int, float] = field(default_factory=dict)
    final_price_by_task_id: dict[int, int] = field(default_factory=dict)


@dataclass(**_DATACLASS_SLOTS_KW)
class _WorkerBallot:
    ordered_task_ids: list[int]
    ballot_comment: str
    coordination_decision: dict[str, Any]


@dataclass(frozen=True, **_DATACLASS_SLOTS_KW)
class _CompletionTextResult:
    text: str
    reasoning_summary: str


class _AzureReasoningSummarySidecar:
    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any],
        logger: LoggerLike | None,
        summary_level: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        self.logger = logger
        self.summary_level = summary_level
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._disabled = False
        self._disabled_reason = ""

    def summarize(self, *, system_prompt: str, user_prompt: str, assistant_text: str) -> str:
        if self._disabled:
            return ""
        if not str(self.model_name or "").startswith("azure/"):
            self._disable_once("reasoning-summary sidecar only supports Azure model routes.")
            return ""
        api_base = str(self.model_kwargs.get("api_base") or os.getenv("AZURE_API_BASE") or "").strip()
        api_key = (
            str(os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
            or str(os.getenv("AZURE_API_KEY") or "").strip()
        )
        if not api_base or not api_key:
            self._disable_once("reasoning-summary sidecar missing Azure endpoint or API key.")
            return ""
        deployment_name = str(self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name).strip()
        if not deployment_name:
            self._disable_once("reasoning-summary sidecar missing Azure deployment name.")
            return ""

        endpoint = api_base.rstrip("/") + "/openai/v1/responses"
        body = {
            "model": deployment_name,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": assistant_text,
                        }
                    ],
                },
            ],
            "reasoning": {
                "summary": self.summary_level,
            },
            "text": {
                "verbosity": "low",
            },
            "max_output_tokens": 32,
            "store": False,
        }
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            self._disable_once(
                f"reasoning-summary sidecar disabled after Azure HTTP {exc.code}: {message[:240]}"
            )
            return ""
        except Exception as exc:
            self._disable_once(f"reasoning-summary sidecar disabled after request failure: {exc}")
            return ""
        try:
            response_payload = json.loads(raw)
        except Exception as exc:
            self._disable_once(f"reasoning-summary sidecar disabled after invalid JSON response: {exc}")
            return ""
        return QueueNegotiator._extract_reasoning_summary(response_payload)

    def _disable_once(self, reason: str) -> None:
        if self._disabled:
            return
        self._disabled = True
        self._disabled_reason = reason
        if self.logger is not None:
            self.logger.log(reason)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class QueueNegotiator:
    # Backstop for litellm.completion() calls that never return within their own configured
    # `timeout` kwarg. Observed directly on a real OpenRouter run (z-ai/glm-5.3, 2026-08-31): a
    # completion call held an ESTABLISHED TCP connection (confirmed via `lsof`; the general API
    # was independently confirmed reachable via a direct curl at the same moment, so this was not
    # a broader outage) and never returned or raised, well past its configured
    # model_kwargs["timeout"]=120 -- litellm/httpx's own timeout enforcement did not fire for
    # whatever this specific stall was. Because DockerTaskScheduler.run()'s negotiation path
    # (_ensure_negotiation_state_locked) calls QueueNegotiator.negotiate() while holding the
    # scheduler's global lock, a single stuck completion call like this freezes every worker and
    # the monitor thread, not just this one negotiation round -- so a hard, code-level deadline
    # independent of litellm/httpx's own timeout handling is required, not optional. Used only as
    # a fallback when the caller hasn't configured its own `timeout` in model_kwargs.
    _COMPLETION_HARD_TIMEOUT_FALLBACK_SECONDS = 150.0

    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        logger: LoggerLike | None = None,
        prompt_config_path: str | Path | None = None,
        completion_max_retries: int = 0,
        completion_retry_base_delay_seconds: float = 2.0,
        structured_parse_repair_attempts: int = 0,
        visible_coordination_task_field: str = "peer_task_id",
        visible_coordination_actions: list[str] | tuple[str, ...] | None = None,
        reasoning_summary_level: str | None = None,
        memory_summary_redactor: Callable[[str], str] | None = None,
        allow_bid_revision: bool = True,
        supports_history_summarization: bool = True,
        communication_history_disabled: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        self.logger = logger
        self.prompt_config_path = Path(prompt_config_path) if prompt_config_path else DEFAULT_PROMPT_CONFIG_PATH
        self.prompt_config = self._load_prompt_config(self.prompt_config_path)
        self.completion_max_retries = max(0, int(completion_max_retries))
        self.completion_retry_base_delay_seconds = max(0.0, float(completion_retry_base_delay_seconds))
        self.structured_parse_repair_attempts = max(0, int(structured_parse_repair_attempts))
        self.visible_coordination_task_field = str(visible_coordination_task_field or "peer_task_id")
        self.visible_coordination_actions = list(
            visible_coordination_actions
            if visible_coordination_actions is not None
            else ["proceed", "wait_for_peer", "delegate_to_peer", "wait_for_update", "worker_run_first", "peer_run_first"]
        )
        # Optional caller-supplied redactor applied to candidate.memory_summary whenever it's
        # rendered for a DIFFERENT worker than the one it describes (mirrors redact_deadline
        # below). memory_summary is free text the caller controls -- deadline_minutes/
        # deadline_remaining_seconds are structured fields this module already knows how to
        # redact on its own, but private figures embedded in memory_summary's prose (e.g. a
        # ProgramBench-auction variant's remaining-budget sentence) are opaque to this shared
        # module, so the caller that knows that prose's shape must supply the scrub. Left as
        # None (no-op) for any caller that doesn't pass one, so this is a strictly additive,
        # zero-risk change for every other script that already constructs a QueueNegotiator.
        self._memory_summary_redactor = memory_summary_redactor
        # Default True (matches existing behavior for every caller that never passes this --
        # zero-risk, purely additive). When False, direct_message/response never offer
        # "revised_duration_seconds"/"revised_bid_price" as schema keys at all (the model isn't
        # told they exist), and even if a model hallucinates them anyway despite never being
        # asked, the values are ignored -- see _run_direct_message_round/_run_response_round's
        # own gating right below where revisions get applied to current_duration_by_task_id/
        # current_price_by_task_id. This is the mechanism side of --fixed-starting-bid in the
        # ProgramBench auction script: with it off, a round's declared bid is genuinely final,
        # matching the original sealed-bid-once design this project ran with before bid revision
        # existed, specifically so the two conditions can be A/B compared.
        self.allow_bid_revision = bool(allow_bid_revision)
        # Whether this negotiator is allowed to issue a real completion call to condense aging
        # prior-round history once it grows past _PRIOR_HISTORY_TOKEN_THRESHOLD (see
        # _maybe_summarize_prior_history below). Default True for this base class, which always
        # has a real, callable model_name. LiveAgentQueueNegotiator (queue_negotiation_live_agent.py)
        # explicitly passes False here -- it's constructed with a placeholder model_name
        # ("live-agent-solver") specifically because none of its own overridden stage methods ever
        # call self._complete_text, so nothing before this feature existed ever tried to actually
        # resolve that name into a real model. This method is inherited UNCHANGED by that
        # subclass, so without this flag it would be the one path that tries to and fails.
        self._supports_history_summarization = bool(supports_history_summarization)
        # Distinguishes "prior_history is empty because this happens to be round 1, more will
        # accumulate from round 2 on" (the normal case) from "prior_history will be empty EVERY
        # round for the whole subrun, by the caller's own deliberate configuration" (the
        # --disable-communication-history ablation condition, DockerTaskScheduler's
        # communication_history_enabled=False -- see its own docstring). The prompt's own count-
        # based framing ("if 0, no memory of any earlier round exists") is already truthful either
        # way, but it never tells the model WHY the count is 0 -- a model has no way to distinguish
        # "just hasn't happened yet" from "structurally will never happen" from inside a single
        # round. When this flag is set, _no_prior_history_text() below says so explicitly instead
        # of leaving that permanence question unanswered every single round.
        self.communication_history_disabled = bool(communication_history_disabled)
        # Keyed by repo_slug. Each entry: {"digest_text": str, "covered_round_ids": set[int]}.
        # See _maybe_summarize_prior_history for the sliding-window policy this backs.
        self._compacted_history_cache: dict[str, dict[str, Any]] = {}
        normalized_reasoning_summary = str(reasoning_summary_level or "").strip().lower()
        self.reasoning_summary_level = normalized_reasoning_summary or None
        self._is_azure_model = str(self.model_name or "").startswith("azure/")
        self._reasoning_summary_sidecar = (
            _AzureReasoningSummarySidecar(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                logger=self.logger,
                summary_level=self.reasoning_summary_level,
            )
            if self.reasoning_summary_level and self._is_azure_model
            else None
        )
        # Non-Azure providers (OpenRouter and most others litellm routes to) return their
        # reasoning trace inline on the same completion call, but only if the request asks
        # for it explicitly via a "reasoning" body param. Azure/OpenAI's reasoning models
        # reason internally regardless, and their summary is instead fetched separately by
        # the sidecar above, so this generic request param is only added off the Azure path.
        self._generic_reasoning_request_kwargs: dict[str, Any] = (
            {
                "reasoning": {
                    "effort": os.environ.get("NEGOTIATION_REASONING_EFFORT","").strip().lower() or {"concise": "low", "auto": "medium", "detailed": "high"}.get(
                        self.reasoning_summary_level, "medium"
                    ),
                    "enabled": True,
                }
            }
            if self.reasoning_summary_level and not self._is_azure_model
            else {}
        )

    @staticmethod
    def _run_concurrently(items: list[Any], worker: Any) -> list[Any]:
        """Run worker(item) for every item on its own daemon thread and return results in
        the same order as `items`. Each candidate's declaration/reasoning/message/ballot call
        is an independent completion request once the phase it belongs to already has whatever
        shared state it needs (e.g. reasoning needs every declaration, but not another
        candidate's reasoning) -- processing candidates one at a time inside a phase was pure
        serialization with no data dependency requiring it, and it's what made a single slow or
        retried candidate block everyone behind it in the same phase (observed live: two
        candidates visibly resolved while a third sat unprocessed for minutes behind a hard
        timeout + JSON-repair retry on an earlier candidate in the loop). Raw daemon threads,
        not concurrent.futures.ThreadPoolExecutor, for the same reason as
        _call_completion_with_hard_timeout below: ThreadPoolExecutor's atexit hook hangs process
        shutdown, confirmed via a dedicated subprocess test.
        """
        results: list[Any] = [None] * len(items)
        errors: list[BaseException | None] = [None] * len(items)

        def _run_one(index: int, item: Any) -> None:
            try:
                results[index] = worker(item)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
                errors[index] = exc

        threads = [
            threading.Thread(target=_run_one, args=(index, item), daemon=True)
            for index, item in enumerate(items)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for exc in errors:
            if exc is not None:
                raise exc
        return results

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        return (
            "ratelimit" in name
            or "rate limit" in message
            or "too many requests" in message
            or "http 429" in message
            or "status code 429" in message
        )

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> float | None:
        message = str(exc)
        patterns = [
            r"retry(?:ing)? in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"retry[- ]after[: ]+([0-9]+(?:\.[0-9]+)?)",
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if not match:
                continue
            try:
                return max(0.0, float(match.group(1)))
            except ValueError:
                continue
        return None

    def _compute_retry_delay_seconds(self, exc: Exception, attempt_index: int) -> float:
        base_delay = self.completion_retry_base_delay_seconds * (2 ** (attempt_index - 1))
        if not self._is_rate_limit_error(exc):
            return base_delay

        retry_after = self._extract_retry_after_seconds(exc)
        capped_base = min(max(base_delay, retry_after or 0.0, 1.0), 60.0)
        # Add a small jitter so concurrent workers do not all retry in lockstep.
        jitter = random.uniform(0.0, min(1.0, capped_base * 0.25))
        return capped_base + jitter

    def negotiate(
        self,
        candidates: list[NegotiationCandidate],
        *,
        prior_history: list[dict[str, Any]] | None = None,
    ) -> NegotiationOutcome:
        if not candidates:
            return NegotiationOutcome([], {}, {}, {}, {}, {}, {}, {}, {}, [],)
        # Private per-candidate cost-of-delay multiplier: how much a worker's own score
        # falls per minute past its deadline. Rolled fresh every negotiate() call (i.e. every
        # round) and surfaced only in that candidate's own prompt context (_build_prompt_context
        # keys it by the current candidate, never by peers), so it never appears in
        # candidates_json/peer_candidates_json, which serialize NegotiationCandidate itself.
        self._cost_of_delay_multiplier_by_task_id = {
            candidate.task_id: round(random.uniform(1.0, 3.0), 2) for candidate in candidates
        }
        normalized_history = self._normalize_prior_history(prior_history or [])
        if len(candidates) == 1:
            candidate = candidates[0]
            declaration = self._format_declaration_text(
                {
                    "resource_summary": "No competing workers are present in this communication round.",
                    "resources_needed": [candidate.task_kind],
                    "estimated_cost": "minimal contention",
                    "urgency": "Only candidate in the round.",
                }
            )
            discussion = "No direct messages were needed because this is the only communication candidate."
            return NegotiationOutcome(
                [candidate.task_id],
                {
                    candidate.task_id: (
                        f"{declaration} Position reasoning: only active candidate in the communication round. "
                        f"{discussion}"
                    )
                },
                {candidate.task_id: "Only communication candidate at the moment."},
                {candidate.task_id: declaration},
                {candidate.task_id: ""},
                {candidate.task_id: "Preferred position: 1. Position reasoning: only active candidate in the communication round."},
                {candidate.task_id: ""},
                {candidate.task_id: discussion},
                {candidate.task_id: "No ballot contest was required."},
                {candidate.task_id: ""},
                {candidate.task_id: self._default_coordination_decision()},
                [],
                # This class's own field docstring says a candidate that "never declared a
                # duration/price and never revised one either" should default to 0.0/0 -- but
                # relying on NegotiationOutcome's field(default_factory=dict) here left the key
                # entirely ABSENT for the lone candidate, so callers doing
                # final_duration_by_task_id.get(task_id) got None, not the documented 0.0. Set
                # explicitly so this path actually matches its own class's documented contract.
                final_duration_by_task_id={candidate.task_id: 0.0},
                final_price_by_task_id={candidate.task_id: 0},
            )

        candidate_by_id = {candidate.task_id: candidate for candidate in candidates}
        history_by_task_id = {
            candidate.task_id: self._filter_prior_history_for_candidate(normalized_history, candidate)
            for candidate in candidates
        }
        compact_history_by_task_id = {
            candidate.task_id: self._maybe_summarize_prior_history(candidate, history_by_task_id[candidate.task_id])
            for candidate in candidates
        }
        declaration_results = self._run_concurrently(
            candidates,
            lambda candidate: self._generate_declaration(
                candidate,
                candidates,
                history_by_task_id[candidate.task_id],
                compact_history_by_task_id[candidate.task_id],
            ),
        )
        declarations = {
            candidate.task_id: result for candidate, result in zip(candidates, declaration_results)
        }
        reasoning_results = self._run_concurrently(
            candidates,
            lambda candidate: self._generate_reasoning(
                candidate,
                candidates,
                declarations,
                history_by_task_id[candidate.task_id],
                compact_history_by_task_id[candidate.task_id],
            ),
        )
        reasonings = {
            candidate.task_id: result for candidate, result in zip(candidates, reasoning_results)
        }
        declaration_reasoning_summary_by_task_id = {
            task_id: payload.pop("_reasoning_summary", "")
            for task_id, payload in declarations.items()
        }
        reasoning_reasoning_summary_by_task_id = {
            task_id: payload.pop("_reasoning_summary", "")
            for task_id, payload in reasonings.items()
        }

        # Running bid, seeded from each candidate's own declaration and updated in place by
        # that same candidate's own response-phase revisions (see _run_response_round) --
        # never by anyone else's. Whatever this holds by the time discussion rounds end is what
        # final_duration_by_task_id/final_price_by_task_id (below) expose as the round's actual,
        # binding bid -- the mechanism that makes "I'll raise my bid" in a response message a
        # real, consequential act instead of prose with no effect on what gets granted/charged.
        current_duration_by_task_id: dict[int, float] = {}
        current_price_by_task_id: dict[int, int] = {}
        for candidate in candidates:
            declared = declarations[candidate.task_id]
            duration = self._normalize_optional_positive_number(declared.get("desired_duration_seconds"))
            if duration is not None:
                current_duration_by_task_id[candidate.task_id] = duration
            price = self._normalize_optional_non_negative_int(declared.get("bid_price"))
            if price is not None:
                current_price_by_task_id[candidate.task_id] = price

        transcript: list[dict[str, Any]] = []
        rounds = max(1, int(self.prompt_config.get("hyperparameters", {}).get("rounds", 4)))
        quiet_rounds_to_stop = max(
            1,
            int(self.prompt_config.get("hyperparameters", {}).get("quiet_rounds_to_stop", 1)),
        )
        quiet_round_streak = 0
        for round_number in range(1, rounds + 1):
            direct_events, direct_revisions_by_task_id = self._run_direct_message_round(
                round_number,
                rounds,
                candidates,
                candidate_by_id,
                declarations,
                reasonings,
                transcript,
                history_by_task_id,
                compact_history_by_task_id,
            )
            transcript.extend(direct_events)
            for task_id, (revised_duration, revised_price) in direct_revisions_by_task_id.items():
                if revised_duration is not None:
                    current_duration_by_task_id[task_id] = revised_duration
                if revised_price is not None:
                    current_price_by_task_id[task_id] = revised_price
            response_events, response_revisions_by_task_id = self._run_response_round(
                round_number,
                rounds,
                candidates,
                candidate_by_id,
                declarations,
                reasonings,
                transcript,
                history_by_task_id,
                compact_history_by_task_id,
            )
            transcript.extend(response_events)
            for task_id, (revised_duration, revised_price) in response_revisions_by_task_id.items():
                if revised_duration is not None:
                    current_duration_by_task_id[task_id] = revised_duration
                if revised_price is not None:
                    current_price_by_task_id[task_id] = revised_price
            if (
                direct_events
                or response_events
                or direct_revisions_by_task_id
                or response_revisions_by_task_id
            ):
                quiet_round_streak = 0
            else:
                quiet_round_streak += 1
                if quiet_round_streak >= quiet_rounds_to_stop and round_number < rounds:
                    self._log(
                        "queue negotiator ending early after "
                        f"{quiet_round_streak} quiet round(s); no new communication events were produced."
                    )
                    break

        ballot_results = self._run_concurrently(
            candidates,
            lambda candidate: self._generate_ballot(
                candidate,
                candidates,
                declarations,
                reasonings,
                transcript,
                history_by_task_id[candidate.task_id],
                compact_history_by_task_id[candidate.task_id],
            ),
        )
        ballots: dict[int, _WorkerBallot] = {}
        ballot_reasoning_summary_by_task_id: dict[int, str] = {}
        for candidate, (ballot, reasoning_summary) in zip(candidates, ballot_results):
            ballot_reasoning_summary_by_task_id[candidate.task_id] = reasoning_summary
            if ballot is not None:
                ballots[candidate.task_id] = ballot

        ordered_task_ids, rationale_by_task_id = self._aggregate_ballots(candidates, ballots)
        if not ordered_task_ids:
            ordered_task_ids = self._fallback_order(candidates)
        if not rationale_by_task_id:
            rationale_by_task_id = {
                candidate.task_id: "Fallback ordering used because worker ballots could not be parsed."
                for candidate in candidates
            }

        declaration_by_task_id = {
            candidate.task_id: self._format_declaration_text(declarations[candidate.task_id])
            for candidate in candidates
        }
        reasoning_by_task_id = {
            candidate.task_id: self._format_reasoning_text(reasonings[candidate.task_id])
            for candidate in candidates
        }
        discussion_by_task_id = {
            candidate.task_id: self._format_discussion_text(candidate.task_id, transcript, candidate_by_id)
            for candidate in candidates
        }
        ballot_by_task_id = {
            candidate.task_id: ballots.get(candidate.task_id).ballot_comment.strip()
            if candidate.task_id in ballots
            else ""
            for candidate in candidates
        }
        coordination_decision_by_task_id = {
            candidate.task_id: ballots.get(candidate.task_id).coordination_decision
            if candidate.task_id in ballots
            else self._default_coordination_decision()
            for candidate in candidates
        }

        message_by_task_id = {}
        for candidate in candidates:
            parts = [
                declaration_by_task_id[candidate.task_id],
                reasoning_by_task_id[candidate.task_id],
                discussion_by_task_id[candidate.task_id],
                ballot_by_task_id[candidate.task_id],
            ]
            message_by_task_id[candidate.task_id] = " ".join(part for part in parts if part)

        final_duration_by_task_id = {
            candidate.task_id: current_duration_by_task_id.get(candidate.task_id, 0.0) for candidate in candidates
        }
        final_price_by_task_id = {
            candidate.task_id: current_price_by_task_id.get(candidate.task_id, 0) for candidate in candidates
        }
        return NegotiationOutcome(
            ordered_task_ids,
            message_by_task_id,
            rationale_by_task_id,
            declaration_by_task_id,
            declaration_reasoning_summary_by_task_id,
            reasoning_by_task_id,
            reasoning_reasoning_summary_by_task_id,
            discussion_by_task_id,
            ballot_by_task_id,
            ballot_reasoning_summary_by_task_id,
            coordination_decision_by_task_id,
            list(transcript),
            final_duration_by_task_id,
            final_price_by_task_id,
        )

    def _run_direct_message_round(
        self,
        round_number: int,
        total_rounds: int,
        candidates: list[NegotiationCandidate],
        candidate_by_id: dict[int, NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history_by_task_id: dict[int, list[dict[str, Any]]],
        compact_history_by_task_id: dict[int, str],
    ) -> tuple[list[dict[str, Any]], dict[int, tuple[float | None, int | None]]]:
        valid_ids = {candidate.task_id for candidate in candidates}

        def _per_candidate(candidate: NegotiationCandidate) -> tuple[list[dict[str, Any]], float | None, int | None]:
            candidate_events: list[dict[str, Any]] = []
            selected_recipients = self._select_recipients(
                candidate,
                candidates,
                declarations,
                reasonings,
                transcript,
                prior_history_by_task_id[candidate.task_id],
                compact_history_by_task_id[candidate.task_id],
                round_number,
                total_rounds,
            )
            selected_recipients = [
                task_id for task_id in selected_recipients if task_id in valid_ids and task_id != candidate.task_id
            ]
            if not selected_recipients:
                return candidate_events, None, None
            messages, revised_duration, revised_bid_price, reasoning_summary = self._compose_direct_messages(
                candidate,
                candidates,
                declarations,
                reasonings,
                transcript,
                prior_history_by_task_id[candidate.task_id],
                compact_history_by_task_id[candidate.task_id],
                selected_recipients,
                round_number,
                total_rounds,
            )
            for payload in messages:
                recipient_task_id = payload.get("recipient_task_id")
                message = str(payload.get("message") or "").strip()
                if recipient_task_id not in valid_ids or recipient_task_id == candidate.task_id or not message:
                    continue
                recipient = candidate_by_id[recipient_task_id]
                candidate_events.append(
                    {
                        "round": round_number,
                        "phase": "direct_message",
                        "from_task_id": candidate.task_id,
                        "from_repo_name": candidate.repo_name,
                        "to_task_id": recipient_task_id,
                        "to_repo_name": recipient.repo_name,
                        "message": message,
                        "reasoning_summary": reasoning_summary,
                    }
                )
            return candidate_events, revised_duration, revised_bid_price

        events: list[dict[str, Any]] = []
        revisions_by_task_id: dict[int, tuple[float | None, int | None]] = {}
        for candidate, (candidate_events, revised_duration, revised_bid_price) in zip(
            candidates, self._run_concurrently(candidates, _per_candidate)
        ):
            events.extend(candidate_events)
            if revised_duration is not None or revised_bid_price is not None:
                revisions_by_task_id[candidate.task_id] = (revised_duration, revised_bid_price)
        return events, revisions_by_task_id

    def _run_response_round(
        self,
        round_number: int,
        total_rounds: int,
        candidates: list[NegotiationCandidate],
        candidate_by_id: dict[int, NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history_by_task_id: dict[int, list[dict[str, Any]]],
        compact_history_by_task_id: dict[int, str],
    ) -> tuple[list[dict[str, Any]], dict[int, tuple[float | None, int | None]]]:
        valid_ids = {candidate.task_id for candidate in candidates}

        def _per_candidate(candidate: NegotiationCandidate) -> tuple[list[dict[str, Any]], float | None, int | None]:
            candidate_events: list[dict[str, Any]] = []
            inbox = [
                event
                for event in transcript
                if event["round"] == round_number
                and event["phase"] == "direct_message"
                and event["to_task_id"] == candidate.task_id
            ]
            if not inbox:
                return candidate_events, None, None
            responses, revised_duration, revised_bid_price, reasoning_summary = self._generate_responses(
                candidate,
                candidates,
                declarations,
                reasonings,
                transcript,
                prior_history_by_task_id[candidate.task_id],
                compact_history_by_task_id[candidate.task_id],
                inbox,
                round_number,
                total_rounds,
            )
            for payload in responses:
                recipient_task_id = payload.get("recipient_task_id")
                message = str(payload.get("message") or "").strip()
                if recipient_task_id not in valid_ids or recipient_task_id == candidate.task_id or not message:
                    continue
                recipient = candidate_by_id[recipient_task_id]
                candidate_events.append(
                    {
                        "round": round_number,
                        "phase": "response",
                        "from_task_id": candidate.task_id,
                        "from_repo_name": candidate.repo_name,
                        "to_task_id": recipient_task_id,
                        "to_repo_name": recipient.repo_name,
                        "message": message,
                        "reasoning_summary": reasoning_summary,
                    }
                )
            return candidate_events, revised_duration, revised_bid_price

        events: list[dict[str, Any]] = []
        revisions_by_task_id: dict[int, tuple[float | None, int | None]] = {}
        for candidate, (candidate_events, revised_duration, revised_bid_price) in zip(
            candidates, self._run_concurrently(candidates, _per_candidate)
        ):
            events.extend(candidate_events)
            if revised_duration is not None or revised_bid_price is not None:
                revisions_by_task_id[candidate.task_id] = (revised_duration, revised_bid_price)
        return events, revisions_by_task_id

    def _generate_declaration(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        prior_history: list[dict[str, Any]],
        compact_history: str,
    ) -> dict[str, Any]:
        prompt_name = "declaration"
        context = self._build_prompt_context(
            candidate,
            candidates,
            declarations={},
            reasonings={},
            transcript=[],
            prior_history=prior_history,
            compact_history=compact_history,
            prompt_name=prompt_name,
        )
        # desired_position/desired_duration_seconds/bid_price are optional, additive fields --
        # only the sequential time-slot AUCTION prompt variant instructs the model to fill them
        # in (its own system/user prompt text explains what they mean); every other variant's
        # prompt never mentions them, so their models simply omit them and _format_declaration_text
        # below renders nothing extra. This exists because the auction's own schedule-extraction
        # regex (run_programbench_auction_subrun_repo_agents.py's _extract_latest_round_schedule)
        # needs a reliable, structured place to read these three numbers from -- asking the model
        # to remember to embed a literal "DESIRED_DURATION_SECONDS: <n>" line inside the free-text
        # resource_summary field (the only place it could go before this fix) worked in barely 2
        # of 129 real declarations sampled from a live run, silently dropping almost every
        # candidate from its own resolved schedule and leaving every solver stuck in an infinite
        # "wasn't reflected in this round's resolved schedule" retry loop despite negotiation
        # itself completing genuinely every round.
        schema = (
            'Return strict JSON with keys "resource_summary", "resources_needed", '
            '"estimated_cost", "urgency", and optionally "desired_position" (integer), '
            '"desired_duration_seconds" (integer), and "bid_price" (integer) -- include these '
            "three only if your system prompt explains what they mean for this negotiation; "
            "omit them entirely otherwise."
        )
        payload, reasoning_summary = self._complete_json(candidate, prompt_name, context, schema)
        resources_needed = [
            str(item).strip()
            for item in payload.get("resources_needed", [])
            if str(item).strip()
        ]
        if not resources_needed:
            fallback_resources: list[str] = []
            task_kind = str(candidate.task_kind or "").strip()
            if task_kind:
                fallback_resources.append(task_kind.replace("_", " "))
            summary_text = " ".join(
                str(payload.get(key) or "").strip().lower()
                for key in ("resource_summary", "estimated_cost", "urgency")
            )
            if "handoff" in summary_text or "share" in summary_text or "peer" in summary_text:
                fallback_resources.append("peer handoff")
            if "test" in summary_text or "validat" in summary_text:
                fallback_resources.append("narrow validation run")
            if "inspect" in summary_text or "file" in summary_text or "search" in summary_text:
                fallback_resources.append("file inspection")
            seen_resources: set[str] = set()
            resources_needed = []
            for item in fallback_resources:
                normalized = item.strip()
                if not normalized or normalized in seen_resources:
                    continue
                seen_resources.add(normalized)
                resources_needed.append(normalized)
            if not resources_needed:
                resources_needed = [task_kind.replace("_", " ") if task_kind else "shared execution slot"]
        return {
            "resource_summary": str(payload.get("resource_summary") or "").strip(),
            "resources_needed": resources_needed,
            "estimated_cost": str(payload.get("estimated_cost") or "").strip(),
            "urgency": str(payload.get("urgency") or "").strip(),
            # Passed through as-is (None if the model omitted them, which is expected/correct for
            # every non-auction prompt variant) -- _format_declaration_text reads these three
            # directly. Without this, the schema/prompt changes that ask the model for them are
            # inert: the model's raw JSON response could contain them and they'd still never
            # reach the rendered declaration text, since this return statement is what actually
            # becomes declarations[task_id] downstream.
            "desired_position": payload.get("desired_position"),
            "desired_duration_seconds": payload.get("desired_duration_seconds"),
            "bid_price": payload.get("bid_price"),
            "_reasoning_summary": reasoning_summary,
        }

    def _generate_reasoning(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        prior_history: list[dict[str, Any]],
        compact_history: str,
    ) -> dict[str, Any]:
        prompt_name = "reasoning"
        context = self._build_prompt_context(
            candidate,
            candidates,
            declarations=declarations,
            reasonings={},
            transcript=[],
            prior_history=prior_history,
            compact_history=compact_history,
            prompt_name=prompt_name,
        )
        schema = (
            'Return strict JSON with keys "preferred_position", "ordered_task_ids", '
            '"reasoning", and "other_positions". "other_positions" should be a list of '
            'objects with keys "task_id", "suggested_position", and "reason".'
        )
        payload, reasoning_summary = self._complete_json(candidate, prompt_name, context, schema)
        candidate_ids = [item.task_id for item in candidates]
        ordered_task_ids = self._normalize_task_id_order(payload.get("ordered_task_ids", []), candidate_ids)
        preferred_position = self._normalize_position(payload.get("preferred_position"), len(candidate_ids))
        other_positions = []
        for item in payload.get("other_positions", []):
            if not isinstance(item, dict):
                continue
            task_id = self._normalize_task_id(item.get("task_id"))
            if task_id is None or task_id == candidate.task_id or task_id not in candidate_ids:
                continue
            other_positions.append(
                {
                    "task_id": task_id,
                    "suggested_position": self._normalize_position(item.get("suggested_position"), len(candidate_ids)),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        return {
            "preferred_position": preferred_position,
            "ordered_task_ids": ordered_task_ids,
            "reasoning": str(payload.get("reasoning") or "").strip(),
            "other_positions": other_positions,
            "_reasoning_summary": reasoning_summary,
        }

    def _select_recipients(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history: list[dict[str, Any]],
        compact_history: str,
        round_number: int,
        total_rounds: int,
    ) -> list[int]:
        prompt_name = "recipient_selection"
        context = self._build_prompt_context(
            candidate,
            candidates,
            declarations=declarations,
            reasonings=reasonings,
            transcript=transcript,
            prior_history=prior_history,
            compact_history=compact_history,
            prompt_name=prompt_name,
            round_number=round_number,
            total_rounds=total_rounds,
        )
        schema = 'Return strict JSON with key "recipient_task_ids" as a list of task IDs.'
        payload, _ = self._complete_json(candidate, prompt_name, context, schema)
        valid_ids = [item.task_id for item in candidates if item.task_id != candidate.task_id]
        selected = []
        seen: set[int] = set()
        for value in payload.get("recipient_task_ids", []):
            task_id = self._normalize_task_id(value)
            if task_id is None or task_id not in valid_ids or task_id in seen:
                continue
            seen.add(task_id)
            selected.append(task_id)
        return selected

    def _compose_direct_messages(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history: list[dict[str, Any]],
        compact_history: str,
        selected_recipients: list[int],
        round_number: int,
        total_rounds: int,
    ) -> tuple[list[dict[str, Any]], str]:
        prompt_name = "direct_message"
        context = self._build_prompt_context(
            candidate,
            candidates,
            declarations=declarations,
            reasonings=reasonings,
            transcript=transcript,
            prior_history=prior_history,
            compact_history=compact_history,
            prompt_name=prompt_name,
            round_number=round_number,
            total_rounds=total_rounds,
            selected_recipient_ids=selected_recipients,
        )
        schema = 'Return strict JSON with key "messages" as a list of objects with keys "recipient_task_id" and "message".'
        if self.allow_bid_revision:
            schema += (
                ' Optionally also include top-level keys "revised_duration_seconds" (integer) '
                'and "revised_bid_price" (integer) to update YOUR OWN bid for the rest of this '
                "negotiation round -- provide either, both, or neither; omit whichever one you "
                "are not changing. This is real: whatever value is current when the round's "
                "final ballot is cast is what actually gets granted and charged, not your "
                "original declaration."
            )
        payload, reasoning_summary = self._complete_json(candidate, prompt_name, context, schema)
        revised_duration = (
            self._normalize_optional_positive_number(payload.get("revised_duration_seconds"))
            if self.allow_bid_revision
            else None
        )
        revised_bid_price = (
            self._normalize_optional_non_negative_int(payload.get("revised_bid_price"))
            if self.allow_bid_revision
            else None
        )
        return (
            self._normalize_message_payload(payload.get("messages", []), selected_recipients),
            revised_duration,
            revised_bid_price,
            reasoning_summary,
        )

    def _generate_responses(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history: list[dict[str, Any]],
        compact_history: str,
        inbox: list[dict[str, Any]],
        round_number: int,
        total_rounds: int,
    ) -> tuple[list[dict[str, Any]], str]:
        prompt_name = "response"
        context = self._build_prompt_context(
            candidate,
            candidates,
            declarations=declarations,
            reasonings=reasonings,
            transcript=transcript,
            prior_history=prior_history,
            compact_history=compact_history,
            prompt_name=prompt_name,
            round_number=round_number,
            total_rounds=total_rounds,
            inbox=inbox,
        )
        schema = 'Return strict JSON with key "responses" as a list of objects with keys "recipient_task_id" and "message".'
        if self.allow_bid_revision:
            schema += (
                ' Optionally also include top-level keys "revised_duration_seconds" (integer) '
                'and "revised_bid_price" (integer) to update YOUR OWN bid for the rest of this '
                "negotiation round -- provide either, both, or neither; omit whichever one you "
                "are not changing (it keeps its current value, either your original declared "
                "value or an earlier revision, whichever is most recent). This is real: "
                "whatever value is current when the round's final ballot is cast is what "
                "actually gets granted and charged, not your original declaration."
            )
        payload, reasoning_summary = self._complete_json(candidate, prompt_name, context, schema)
        allowed_recipients = sorted({event["from_task_id"] for event in inbox if event["from_task_id"] != candidate.task_id})
        revised_duration = (
            self._normalize_optional_positive_number(payload.get("revised_duration_seconds"))
            if self.allow_bid_revision
            else None
        )
        revised_bid_price = (
            self._normalize_optional_non_negative_int(payload.get("revised_bid_price"))
            if self.allow_bid_revision
            else None
        )
        return (
            self._normalize_message_payload(payload.get("responses", []), allowed_recipients),
            revised_duration,
            revised_bid_price,
            reasoning_summary,
        )

    @staticmethod
    def _normalize_optional_positive_number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _normalize_optional_non_negative_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number >= 0 else None

    def _generate_ballot(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history: list[dict[str, Any]],
        compact_history: str,
    ) -> tuple[_WorkerBallot | None, str]:
        prompt_name = "ballot"
        context = self._build_prompt_context(
            candidate,
            candidates,
            declarations=declarations,
            reasonings=reasonings,
            transcript=transcript,
            prior_history=prior_history,
            compact_history=compact_history,
            prompt_name=prompt_name,
        )
        schema = (
            'Return strict JSON with keys "ordered_task_ids", "ballot_comment", and optional '
            '"coordination_decision". "ordered_task_ids" must include every candidate task ID exactly once. '
            '"coordination_decision", when present, should be an object with keys '
            f'"action", "{self.visible_coordination_task_field}", "reason", and "shared_context". '
            f'Allowed "action" values are {", ".join(json.dumps(item) for item in self.visible_coordination_actions)}.'
        )
        payload, reasoning_summary = self._complete_json(candidate, prompt_name, context, schema)
        try:
            ordered = self._normalize_task_id_order(
                payload.get("ordered_task_ids", []),
                [item.task_id for item in candidates],
            )
            if not ordered:
                return None, reasoning_summary
            return _WorkerBallot(
                ordered_task_ids=ordered,
                ballot_comment=str(payload.get("ballot_comment") or "").strip(),
                coordination_decision=self._normalize_coordination_decision(
                    candidate,
                    payload.get("coordination_decision"),
                    candidates,
                ),
            ), reasoning_summary
        except Exception:
            self._log(
                f"queue negotiator could not parse worker ballot for {candidate.repo_name}; ignoring that ballot."
            )
            return None, reasoning_summary

    def _aggregate_ballots(
        self,
        candidates: list[NegotiationCandidate],
        ballots: dict[int, _WorkerBallot],
    ) -> tuple[list[int], dict[int, str]]:
        candidate_ids = [candidate.task_id for candidate in candidates]
        position_scores = {task_id: 0 for task_id in candidate_ids}
        first_place_votes = {task_id: 0 for task_id in candidate_ids}
        support_examples: dict[int, list[str]] = {task_id: [] for task_id in candidate_ids}
        max_points = len(candidate_ids)

        for voter_id, ballot in ballots.items():
            normalized_order = [task_id for task_id in ballot.ordered_task_ids if task_id in position_scores]
            missing = [task_id for task_id in candidate_ids if task_id not in normalized_order]
            normalized_order.extend(missing)
            for index, task_id in enumerate(normalized_order):
                points = max_points - index
                position_scores[task_id] += points
                if index == 0:
                    first_place_votes[task_id] += 1
                    voter_name = next(candidate.repo_name for candidate in candidates if candidate.task_id == voter_id)
                    support_examples[task_id].append(voter_name)

        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -position_scores[candidate.task_id],
                -first_place_votes[candidate.task_id],
                0 if candidate.priority == "high" else 1,
                candidate.deadline_remaining_seconds,
                -candidate.announced_for_seconds,
                candidate.task_id,
            ),
        )
        ordered_task_ids = [candidate.task_id for candidate in ordered_candidates]

        rationale_by_task_id = {}
        for candidate in ordered_candidates:
            supporters = support_examples[candidate.task_id]
            supporter_text = (
                f"Backed for first place by {', '.join(supporters[:3])}."
                if supporters
                else "Reached this position on aggregate ranking points rather than first-place votes."
            )
            rationale_by_task_id[candidate.task_id] = (
                f"Worker consensus score={position_scores[candidate.task_id]}, "
                f"first_place_votes={first_place_votes[candidate.task_id]}. {supporter_text}"
            )
        return ordered_task_ids, rationale_by_task_id

    def _fallback_order(self, candidates: list[NegotiationCandidate]) -> list[int]:
        return [
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

    def _build_prompt_context(
        self,
        candidate: NegotiationCandidate,
        candidates: list[NegotiationCandidate],
        *,
        declarations: dict[int, dict[str, Any]],
        reasonings: dict[int, dict[str, Any]],
        transcript: list[dict[str, Any]],
        prior_history: list[dict[str, Any]],
        compact_history: str,
        prompt_name: str,
        round_number: int | None = None,
        total_rounds: int | None = None,
        selected_recipient_ids: list[int] | None = None,
        inbox: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str]:
        peers = [peer for peer in candidates if peer.task_id != candidate.task_id]
        candidate_ids = [item.task_id for item in candidates]
        return {
            "candidate_repo_name": candidate.repo_name,
            "candidate_repo_slug": candidate.repo_slug,
            "candidate_task_id": candidate.task_id,
            "candidate_priority": candidate.priority,
            "candidate_task_kind": candidate.task_kind,
            "candidate_instance_id": candidate.instance_id or "repo-level task",
            "candidate_description": candidate.description,
            "candidate_announced_for_seconds": f"{candidate.announced_for_seconds:.1f}",
            "candidate_communication_wait_seconds": f"{candidate.communication_wait_seconds:.1f}",
            "candidate_deadline_minutes": candidate.deadline_minutes,
            "candidate_cost_of_delay_multiplier": (
                f"{self._cost_of_delay_multiplier_by_task_id.get(candidate.task_id, 1.0):.2f}"
            ),
            "candidate_deadline_remaining_seconds": f"{candidate.deadline_remaining_seconds:.1f}",
            "candidate_memory_summary": candidate.memory_summary or "No prior attempt memory recorded yet.",
            # Deadlines are private to each worker and must never be visible to a peer via the
            # data layer -- telling a model not to *argue from* deadline info in its own prompt
            # text is not a real guarantee if the structured context it's fed hands over every
            # peer's exact remaining deadline anyway. Every candidate representation that goes
            # to a DIFFERENT worker than the one it describes is redacted; only "candidate_json"
            # (a worker's view of itself) keeps the real numbers, since a worker legitimately
            # needs to know its own deadline.
            "candidate_json": json.dumps(self._candidate_to_dict(candidate), indent=2),
            "candidates_json": json.dumps(
                [
                    self._candidate_to_dict(item, redact_deadline=item.task_id != candidate.task_id)
                    for item in candidates
                ],
                indent=2,
            ),
            "peer_candidates_json": json.dumps(
                [self._candidate_to_dict(item, redact_deadline=True) for item in peers], indent=2
            ),
            "candidate_ids_json": json.dumps(candidate_ids),
            "available_task_ids_json": json.dumps(candidate_ids),
            "round_number": round_number or 0,
            "total_rounds": total_rounds or 0,
            "selected_recipient_ids_json": json.dumps(selected_recipient_ids or []),
            "selected_recipients_json": json.dumps(
                [
                    self._candidate_to_dict(item, redact_deadline=True)
                    for item in peers
                    if item.task_id in (selected_recipient_ids or [])
                ],
                indent=2,
            ),
            "inbox_json": json.dumps(inbox or [], indent=2),
            "transcript_json": json.dumps(transcript, indent=2),
            "prior_history_json": json.dumps(prior_history, indent=2),
            "prior_history_compact_text": compact_history,
            "prior_history_round_count": len(prior_history),
            "declarations_json": json.dumps(declarations, indent=2),
            "reasonings_json": json.dumps(reasonings, indent=2),
            "peer_declarations_json": json.dumps(
                {task_id: payload for task_id, payload in declarations.items() if task_id != candidate.task_id},
                indent=2,
            ),
            "peer_reasonings_json": json.dumps(
                {task_id: payload for task_id, payload in reasonings.items() if task_id != candidate.task_id},
                indent=2,
            ),
            "candidate_declaration_json": json.dumps(declarations.get(candidate.task_id, {}), indent=2),
            "candidate_reasoning_json": json.dumps(reasonings.get(candidate.task_id, {}), indent=2),
            "component_name": prompt_name,
        }

    @staticmethod
    def _normalize_prior_history(prior_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for entry in prior_history:
            if not isinstance(entry, dict):
                continue
            normalized.append(entry)
        return normalized

    @staticmethod
    def _filter_prior_history_for_candidate(
        prior_history: list[dict[str, Any]],
        candidate: "NegotiationCandidate",
    ) -> list[dict[str, Any]]:
        """Scope cross-round history to this repo's own participation.

        A historical round is included only if this repo was one of that round's
        candidates. Within an included round, "per_task" positions and "transcript"
        messages are further restricted to this repo's own task and its own sent/
        received messages, so an agent cannot see other repos' private declarations,
        reasoning, or direct/response messages from rounds it wasn't a party to.
        Round-level facts about a round the repo DID participate in (who else
        competed, the resulting order, execution outcomes) are kept, since the
        repo genuinely witnessed that outcome when it happened.
        """
        filtered: list[dict[str, Any]] = []
        for entry in prior_history:
            if not isinstance(entry, dict):
                continue
            own_task_ids: set[Any] = set()
            for item in entry.get("candidates") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("repo_slug") == candidate.repo_slug or item.get("repo_name") == candidate.repo_name:
                    task_id = item.get("task_id")
                    if task_id is not None:
                        own_task_ids.add(task_id)
            if not own_task_ids:
                continue
            own_task_id_strs = {str(task_id) for task_id in own_task_ids}
            filtered_transcript = [
                event
                for event in (entry.get("transcript") or [])
                if isinstance(event, dict)
                and (
                    event.get("from_repo_name") == candidate.repo_name
                    or event.get("to_repo_name") == candidate.repo_name
                )
            ]
            filtered_per_task = {
                task_id: details
                for task_id, details in (entry.get("per_task") or {}).items()
                if str(task_id) in own_task_id_strs
            }
            # Round-level facts (who else competed, the order, outcomes) are fine to keep for
            # a round this repo actually participated in -- but a peer's deadline is private
            # regardless of whether this repo witnessed the round, so redact it here the same
            # way _build_prompt_context redacts it for the live round. Without this, a peer's
            # real deadline_minutes/deadline_remaining_seconds survives into every later round's
            # prior_history_compact_text via _compact_prior_history below, undoing that redaction.
            filtered_candidates = []
            for item in entry.get("candidates") or []:
                if not isinstance(item, dict):
                    continue
                is_self = item.get("repo_slug") == candidate.repo_slug or item.get("repo_name") == candidate.repo_name
                if is_self:
                    filtered_candidates.append(item)
                    continue
                redacted_item = dict(item)
                redacted_item["deadline_minutes"] = "private (not disclosed to other workers)"
                redacted_item["deadline_remaining_seconds"] = "private (not disclosed to other workers)"
                filtered_candidates.append(redacted_item)
            filtered.append(
                {
                    **entry,
                    "candidates": filtered_candidates,
                    "transcript": filtered_transcript,
                    "per_task": filtered_per_task,
                }
            )
        return filtered

    def _no_prior_history_text(self) -> str:
        """Shared by _compact_prior_history and _maybe_summarize_prior_history for the
        prior_history=[] case -- see communication_history_disabled's own docstring in __init__
        for why this branches: without it, a permanently-disabled run reads no differently from a
        genuinely early round that just hasn't accumulated history yet, every single round.
        """
        if self.communication_history_disabled:
            return (
                "No prior agent-to-agent communication history is available, and none ever will "
                "be for the rest of this subrun -- cross-round communication history is disabled "
                "for this entire run. Every round is negotiated from scratch with no memory of any "
                "earlier round; do not expect this to change in a later round."
            )
        return "No prior agent-to-agent communication history is available."

    def _compact_prior_history(self, prior_history: list[dict[str, Any]]) -> str:
        if not prior_history:
            return self._no_prior_history_text()

        lines: list[str] = []
        for entry in prior_history:
            round_id = entry.get("round_id", "?")
            lines.append(f"Round {round_id}:")
            candidates = entry.get("candidates") or []
            if candidates:
                candidate_bits = []
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    candidate_bits.append(
                        "Task "
                        f"{candidate.get('task_id')} ({candidate.get('repo_name')}, "
                        f"{candidate.get('priority')}, {candidate.get('task_kind')}, "
                        f"deadline={candidate.get('deadline_remaining_seconds')}s)"
                    )
                if candidate_bits:
                    lines.append("Candidates: " + "; ".join(candidate_bits))
            ordered = entry.get("ordered_task_ids") or []
            execution_outcomes = entry.get("execution_outcomes") or []
            if ordered:
                order_text = " > ".join(str(task_id) for task_id in ordered)
                lines.append(f"Consensus order: {order_text}")
            if execution_outcomes:
                outcome_bits = []
                for outcome in execution_outcomes:
                    if not isinstance(outcome, dict):
                        continue
                    task_label = (
                        f"Task {outcome.get('task_id')} ({outcome.get('repo_name')})"
                    )
                    state = outcome.get("state") or "unknown"
                    duration = outcome.get("execution_time_seconds")
                    if duration is None:
                        outcome_bits.append(f"{task_label}: {state}")
                    else:
                        outcome_bits.append(f"{task_label}: {state} in {duration}s")
                if outcome_bits:
                    lines.append("Execution outcomes: " + "; ".join(outcome_bits))
            per_task = entry.get("per_task") or {}
            if per_task:
                lines.append("Per-task positions:")
                for task_id, details in per_task.items():
                    if not isinstance(details, dict):
                        continue
                    pieces = [f"Task {task_id}"]
                    if details.get("declaration"):
                        pieces.append(f"Declaration: {details['declaration']}")
                    if details.get("reasoning"):
                        pieces.append(f"Reasoning: {details['reasoning']}")
                    if details.get("rationale"):
                        pieces.append(f"Consensus: {details['rationale']}")
                    lines.append(" | ".join(pieces))
            transcript = entry.get("transcript") or []
            if transcript:
                lines.append("Transcript:")
                for event in transcript:
                    if not isinstance(event, dict):
                        continue
                    phase = str(event.get("phase") or "message").replace("_", " ").title()
                    lines.append(
                        f"- {phase} {event.get('from_task_id')}->{event.get('to_task_id')}: "
                        f"{str(event.get('message') or '').strip()}"
                    )
        return "\n".join(lines)

    # Rough, dependency-free size estimate -- good enough to decide "is this getting large,"
    # not meant to match a real tokenizer exactly. No tokenizer is imported anywhere in this
    # module already, and pulling one in just for this threshold check isn't worth the added
    # dependency risk for an estimate that only needs to be roughly right.
    _CHARS_PER_TOKEN_ESTIMATE = 4
    # Confirmed via direct measurement on a real 5-worker run: a single candidate's own
    # uncompacted prior-history text reached ~30K tokens by round 6 with nothing bounding it,
    # having started under 3.5K tokens at round 1 -- this threshold is comfortably inside where
    # that growth was already becoming the largest single component of the prompt.
    _PRIOR_HISTORY_TOKEN_THRESHOLD = 20000

    def _maybe_summarize_prior_history(
        self,
        candidate: NegotiationCandidate,
        prior_history: list[dict[str, Any]],
    ) -> str:
        """Same output contract as _compact_prior_history (a single text block for this
        candidate's own prior-round history) -- this is the entry point negotiate() actually
        calls now; _compact_prior_history itself is unchanged and still does the real per-round
        rendering, both for the "under threshold, use as-is" path below and as the one thing this
        method's own digest-building step (_summarize_older_rounds) still renders the single most
        recent prior round with.

        Policy: below _PRIOR_HISTORY_TOKEN_THRESHOLD (estimated), behavior is completely
        unchanged from before this method existed -- every prior round rendered in full via
        _compact_prior_history, nothing summarized, nothing cached. Once over threshold, every
        prior round EXCEPT the single most recent one gets folded into a cached, running digest
        (one real completion call, only for whichever rounds haven't been folded in yet -- see
        _summarize_older_rounds), and that digest plus the most-recent-round's own full text is
        what gets returned. The most recent round is deliberately never itself summarized -- it
        stays in full detail until a newer round pushes it out of that slot, at which point IT
        becomes one of the rounds the next summarization call folds in.
        """
        if not prior_history:
            return self._no_prior_history_text()
        full_text = self._compact_prior_history(prior_history)
        if (
            len(prior_history) <= 1
            or len(full_text) // self._CHARS_PER_TOKEN_ESTIMATE <= self._PRIOR_HISTORY_TOKEN_THRESHOLD
        ):
            return full_text
        if not self._supports_history_summarization:
            # No real model configured to do the summarizing (see that flag's own docstring in
            # __init__) -- degrade to the old, unbounded-but-always-correct behavior rather than
            # attempting a completion call that cannot succeed.
            return full_text
        if "history_summary" not in (self.prompt_config.get("prompts") or {}):
            # This prompt_config variant hasn't defined the history_summary stage -- most callers
            # of this module use a DIFFERENT prompt_config_path than the one this feature was
            # added to, and every one of them must keep working completely unchanged rather than
            # hitting _prompt_settings_for's own KeyError for a stage they were never given.
            return full_text

        last_round = prior_history[-1]
        older_rounds = prior_history[:-1]
        older_round_ids = {entry.get("round_id") for entry in older_rounds}

        cache_entry = self._compacted_history_cache.get(candidate.repo_slug)
        covered_round_ids: set[Any] = set(cache_entry["covered_round_ids"]) if cache_entry else set()
        new_rounds = [entry for entry in older_rounds if entry.get("round_id") not in covered_round_ids]

        if new_rounds or cache_entry is None:
            try:
                digest_text = self._summarize_older_rounds(
                    candidate,
                    existing_digest=cache_entry["digest_text"] if cache_entry else "",
                    new_rounds=new_rounds,
                )
            except Exception as exc:  # noqa: BLE001 -- a failed summarization must never break a round.
                self._log(
                    f"queue negotiator: history-summary call failed for {candidate.repo_name} "
                    f"({type(exc).__name__}: {exc}); using unsummarized prior history for this "
                    "round instead."
                )
                return full_text
            self._compacted_history_cache[candidate.repo_slug] = {
                "digest_text": digest_text,
                "covered_round_ids": covered_round_ids | older_round_ids,
            }
        else:
            digest_text = cache_entry["digest_text"]

        last_round_text = self._compact_prior_history([last_round])
        return f"{digest_text}\n\n{last_round_text}"

    def _summarize_older_rounds(
        self,
        candidate: NegotiationCandidate,
        *,
        existing_digest: str,
        new_rounds: list[dict[str, Any]],
    ) -> str:
        prompt_settings = self._prompt_settings_for(candidate, "history_summary")
        context = {
            "candidate_repo_name": candidate.repo_name,
            "existing_digest_text": (
                existing_digest.strip()
                if existing_digest and existing_digest.strip()
                else "(none yet -- this is the first time older rounds are being condensed for this worker.)"
            ),
            "new_rounds_json": json.dumps(new_rounds, indent=2),
        }
        system_prompt = str(prompt_settings.get("system") or "").format_map(_SafeFormatDict(context))
        user_prompt = str(prompt_settings.get("user") or "").format_map(_SafeFormatDict(context))
        result = self._complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(prompt_settings.get("max_tokens", 800)),
            completion_kwargs=dict(prompt_settings.get("completion_kwargs") or {}),
        )
        digest = result.text.strip()
        return digest or "(history-summary call returned no content.)"

    def _complete_json(
        self,
        candidate: NegotiationCandidate,
        prompt_name: str,
        context: dict[str, Any],
        schema_instruction: str,
    ) -> dict[str, Any]:
        prompt_settings = self._prompt_settings_for(candidate, prompt_name)
        system_prompt = str(prompt_settings.get("system") or "").format_map(_SafeFormatDict(context))
        user_template = str(prompt_settings.get("user") or "")
        user_prompt = (
            user_template
            + "\n\n"
            + schema_instruction
            + "\nOutput JSON only. Do not wrap it in markdown fences. Do not add commentary before or after the JSON."
        ).format_map(_SafeFormatDict(context))
        raw = self._complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(prompt_settings.get("max_tokens", 400)),
            completion_kwargs=dict(prompt_settings.get("completion_kwargs") or {}),
        )
        try:
            parsed = self._parse_structured_response(raw.text)
            if isinstance(parsed, dict):
                return parsed, raw.reasoning_summary
        except Exception:
            repaired = self._attempt_structured_repair(
                candidate=candidate,
                prompt_name=prompt_name,
                raw_response=raw.text,
                schema_instruction=schema_instruction,
                max_tokens=min(int(prompt_settings.get("max_tokens", 400)), 400),
                completion_kwargs=dict(prompt_settings.get("completion_kwargs") or {}),
            )
            if repaired is not None:
                return repaired, raw.reasoning_summary
            self._log(
                f"queue negotiator could not parse {prompt_name} JSON for {candidate.repo_name}; using defaults."
            )
        return {}, raw.reasoning_summary

    def _attempt_structured_repair(
        self,
        *,
        candidate: NegotiationCandidate,
        prompt_name: str,
        raw_response: str,
        schema_instruction: str,
        max_tokens: int,
        completion_kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.structured_parse_repair_attempts <= 0:
            return None
        repair_system_prompt = (
            "You repair malformed structured outputs. Convert the provided model output into one "
            "strict JSON object that matches the requested schema. Return JSON only."
        )
        repair_user_prompt = (
            f"Component: {prompt_name}\n"
            f"Schema: {schema_instruction}\n\n"
            "Original malformed response:\n"
            f"{raw_response}\n\n"
            "Return a repaired JSON object only."
        )
        for attempt_index in range(1, self.structured_parse_repair_attempts + 1):
            try:
                repaired_raw = self._complete_text(
                    system_prompt=repair_system_prompt,
                    user_prompt=repair_user_prompt,
                    max_tokens=max_tokens,
                    completion_kwargs=completion_kwargs,
                )
                parsed = self._parse_structured_response(repaired_raw.text)
                if isinstance(parsed, dict):
                    self._log(
                        f"queue negotiator repaired malformed {prompt_name} JSON for {candidate.repo_name} "
                        f"on repair attempt {attempt_index}."
                    )
                    return parsed
            except Exception as exc:
                self._log(
                    f"queue negotiator repair attempt {attempt_index} failed for {prompt_name} "
                    f"on {candidate.repo_name}: {exc}"
                )
        return None

    def _call_completion_with_hard_timeout(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        merged_kwargs: dict[str, Any],
    ) -> Any:
        """Runs the actual litellm.completion() call on a background thread and gives up
        waiting after a hard deadline, instead of trusting litellm/httpx's own `timeout` kwarg to
        always fire (see _COMPLETION_HARD_TIMEOUT_FALLBACK_SECONDS's docstring for the real
        incident this was added for). If the call doesn't return in time, this raises
        TimeoutError, which _complete_text's existing retry/backoff loop catches exactly like any
        other transient completion failure.

        Deliberately a raw daemon thread + queue.Queue, NOT concurrent.futures.ThreadPoolExecutor:
        confirmed by testing that ThreadPoolExecutor is the wrong tool here -- its module-level
        atexit hook joins every thread from every executor instance ever created before the
        interpreter is allowed to exit, regardless of executor.shutdown(wait=False), so a call
        that never returns would later hang the ENTIRE PROCESS's shutdown, not just this one
        negotiation round -- reintroducing the exact class of hang this method exists to prevent,
        just deferred to exit time. A plain daemon=True thread has no such hook: the interpreter
        can exit immediately regardless of whether this thread is still blocked.
        """
        hard_timeout_seconds = float(
            merged_kwargs.get("timeout") or self._COMPLETION_HARD_TIMEOUT_FALLBACK_SECONDS
        )
        result_queue: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
        # Explicitly pass a request-level timeout to litellm itself, even in the fallback case
        # where the caller never set one -- without this, giving up on the OUTER wait (below)
        # does nothing to the actual HTTP connection: litellm/httpx would keep waiting on their
        # own (much longer, or unbounded) default, so the abandoned request keeps running and
        # getting billed server-side purely because nothing ever told the transport layer to
        # stop. This doesn't guarantee cancellation -- some providers keep generating after a
        # client disconnects regardless -- but it's a real, first-order lever to actually reduce
        # wasted spend, not just this method's original one (bounding OUR wall-clock wait, which
        # only hides the problem from the caller without touching the request that's still
        # running). A small buffer below hard_timeout_seconds so the transport-level abort has a
        # chance to land before our own queue.get() gives up first.
        call_kwargs = dict(merged_kwargs)
        call_kwargs["timeout"] = max(1.0, hard_timeout_seconds - 1.0)

        def _call() -> None:
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools=[],
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                result_queue.put((True, response))
            except Exception as exc:  # noqa: BLE001 -- forwarded to the waiting thread, not swallowed
                result_queue.put((False, exc))

        threading.Thread(target=_call, daemon=True).start()
        try:
            ok, payload = result_queue.get(timeout=hard_timeout_seconds)
        except queue.Empty:
            # The transport-level timeout above gives this a real chance to actually stop, but
            # if it doesn't, the request keeps running on its daemon thread after we give up
            # waiting here -- there is no way to force-cancel it from this side. Report it
            # immediately: this failure mode previously had ZERO trace of ever having happened
            # (see below for when/if a late result eventually surfaces), which is exactly what
            # made it invisible in cost accounting -- confirmed live: dozens of these per run,
            # with no way to know afterward how much they actually cost.
            self._log(
                f"queue negotiator: completion abandoned after {hard_timeout_seconds:.0f}s hard "
                "timeout -- the request may still be running and billing server-side with no "
                "further trace; if it eventually completes, its outcome (and any usage/cost "
                "figure) will be logged separately, tagged 'late/abandoned', once it arrives."
            )

            def _log_late_arrival() -> None:
                try:
                    ok_late, payload_late = result_queue.get()
                except Exception:
                    return
                if ok_late:
                    usage = getattr(payload_late, "usage", None)
                    cost = None
                    try:
                        cost = litellm.completion_cost(payload_late)
                    except Exception:
                        pass
                    self._log(
                        "queue negotiator: a late/abandoned completion arrived after its hard "
                        f"timeout had already fired and been retried -- usage={usage!r} "
                        f"cost={cost!r}. This is real spend that already happened; it was never "
                        "billed to any tracked negotiation round because the retry already "
                        "took its place."
                    )
                else:
                    self._log(
                        "queue negotiator: a late/abandoned completion ultimately failed after "
                        f"its hard timeout had already fired and been retried: {payload_late!r}"
                    )

            threading.Thread(target=_log_late_arrival, daemon=True).start()
            raise TimeoutError(
                f"litellm.completion() did not return within {hard_timeout_seconds:.0f}s (hard timeout)"
            ) from None
        if ok:
            return payload
        raise payload

    def _complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        completion_kwargs: dict[str, Any],
    ) -> _CompletionTextResult:
        merged_kwargs = dict(self.model_kwargs)
        merged_kwargs.update(self._generic_reasoning_request_kwargs)
        merged_kwargs.update(completion_kwargs)
        attempts = self.completion_max_retries + 1
        last_exc: Exception | None = None
        for attempt_index in range(1, attempts + 1):
            call_started_at = time.monotonic()
            try:
                response = self._call_completion_with_hard_timeout(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    merged_kwargs=merged_kwargs,
                )
                # Real, per-call timing + reasoning-token usage -- added to actually measure
                # whether a given reasoning_summary_level is spending more effort (not just
                # asking for it, see _generic_reasoning_request_kwargs) and how that trades off
                # against the round-level and per-call timeouts, instead of guessing from
                # summary-text length or inferring it indirectly from timeout log lines. One line
                # per real attempt (including ones that error out below), not just successes.
                self._log_completion_timing(call_started_at, response)
                message = response.choices[0].message
                content = message.content or ""
                text_content = (
                    "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in content
                    ).strip()
                    if isinstance(content, list)
                    else str(content).strip()
                )
                reasoning_summary = self._extract_reasoning_summary(response)
                if not reasoning_summary and self._reasoning_summary_sidecar is not None and text_content:
                    reasoning_summary = self._reasoning_summary_sidecar.summarize(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        assistant_text=text_content,
                    )
                if isinstance(content, list):
                    return _CompletionTextResult(
                        text_content,
                        reasoning_summary,
                    )
                return _CompletionTextResult(text_content, reasoning_summary)
            except Exception as exc:
                last_exc = exc
                call_duration = time.monotonic() - call_started_at
                if attempt_index >= attempts:
                    break
                delay_seconds = self._compute_retry_delay_seconds(exc, attempt_index)
                self._log(
                    f"queue negotiator completion failed on attempt {attempt_index}/{attempts} "
                    f"after {call_duration:.1f}s with {type(exc).__name__}: {exc}. Retrying in "
                    f"{delay_seconds:.1f}s."
                )
                time.sleep(delay_seconds)
        assert last_exc is not None
        raise last_exc

    def _log_completion_timing(self, call_started_at: float, response: Any) -> None:
        """One line per successful completion attempt: real wall-clock duration plus, when the
        provider reports it, actual reasoning-token usage -- added specifically to answer "is
        reasoning_summary_level='detailed' actually spending more reasoning compute, or just
        returning an equally-short summary regardless" with real numbers instead of guessing
        from summary-text length (which turned out to look terse even under 'detailed') or
        inferring it indirectly from which timeouts fire. Every real attempt gets a line, not
        just the final successful one, so a slow-but-eventually-fine retry is visible too.
        """
        duration = time.monotonic() - call_started_at
        payload = self._to_plain_data(response)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        completion_tokens = None
        reasoning_tokens = None
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
        self._log(
            f"queue negotiator completion: duration={duration:.1f}s "
            f"reasoning_summary_level={self.reasoning_summary_level or 'none'} "
            f"completion_tokens={completion_tokens if completion_tokens is not None else 'unknown'} "
            f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'unknown'} "
            f"model={self.model_name}"
        )

    @classmethod
    def _extract_reasoning_summary(cls, response: Any) -> str:
        response_payload = cls._to_plain_data(response)
        candidates: list[str] = []

        def add_text(value: Any) -> None:
            text = str(value or "").strip()
            if text:
                candidates.append(text)

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                if "summary_text" in value:
                    add_text(value.get("summary_text"))
                summary = value.get("summary")
                if isinstance(summary, list):
                    for item in summary:
                        walk(item)
                reasoning = value.get("reasoning")
                if isinstance(reasoning, str):
                    # Plain-string reasoning trace: how litellm normalizes most non-Azure
                    # reasoning-capable providers (e.g. OpenRouter models) on message.reasoning.
                    add_text(reasoning)
                elif reasoning is not None:
                    walk(reasoning)
                reasoning_content = value.get("reasoning_content")
                if isinstance(reasoning_content, str):
                    # litellm's other common normalized field name for the same thing
                    # (e.g. message.reasoning_content) across several OpenRouter models.
                    add_text(reasoning_content)
                elif reasoning_content is not None:
                    walk(reasoning_content)
                reasoning_details = value.get("reasoning_details")
                if isinstance(reasoning_details, list):
                    # OpenRouter's structured reasoning array (e.g. message.reasoning_details,
                    # items shaped like {"type": "reasoning.text", "text": "..."}). Usually a
                    # duplicate of message.reasoning above; the dedup below drops the repeat.
                    for item in reasoning_details:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            add_text(item.get("text"))
                output = value.get("output")
                if isinstance(output, list):
                    for item in output:
                        walk(item)
                choices = value.get("choices")
                if isinstance(choices, list):
                    for item in choices:
                        walk(item)
                message = value.get("message")
                if isinstance(message, dict):
                    walk(message)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(response_payload)

        unique: list[str] = []
        seen: set[str] = set()
        for text in candidates:
            normalized = " ".join(text.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return "\n".join(unique)

    @classmethod
    def _to_plain_data(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._to_plain_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_plain_data(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return cls._to_plain_data(model_dump())
            except Exception:
                pass
        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            try:
                return cls._to_plain_data(dict_method())
            except Exception:
                pass
        object_dict = getattr(value, "__dict__", None)
        if isinstance(object_dict, dict):
            return cls._to_plain_data(object_dict)
        return str(value)

    def _prompt_settings_for(self, candidate: NegotiationCandidate, prompt_name: str) -> dict[str, Any]:
        global_prompts = dict(self.prompt_config.get("prompts") or {})
        if prompt_name not in global_prompts:
            raise KeyError(f"Prompt configuration is missing required component '{prompt_name}'.")
        settings = _deep_merge({}, global_prompts[prompt_name])
        settings["completion_kwargs"] = _deep_merge(
            {},
            self.prompt_config.get("completion_kwargs") or {},
            settings.get("completion_kwargs") or {},
        )

        overrides = self.prompt_config.get("agent_overrides") or {}
        for override_key in (candidate.repo_slug, candidate.repo_name):
            if override_key not in overrides:
                continue
            override_payload = overrides[override_key] or {}
            settings["completion_kwargs"] = _deep_merge(
                {},
                settings.get("completion_kwargs") or {},
                override_payload.get("completion_kwargs") or {},
            )
            prompt_overrides = (override_payload.get("prompts") or {}).get(prompt_name) or {}
            settings = _deep_merge(settings, prompt_overrides)
            settings["completion_kwargs"] = _deep_merge(
                {},
                settings.get("completion_kwargs") or {},
                prompt_overrides.get("completion_kwargs") or {},
            )
        return settings

    def _load_prompt_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Queue negotiation prompt config not found: {path}. "
                "Create the YAML file or pass a valid prompt config path."
            )
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Queue negotiation prompt config must be a YAML mapping: {path}")
        payload.setdefault("hyperparameters", {})
        payload.setdefault("completion_kwargs", {})
        payload.setdefault("prompts", {})
        payload.setdefault("agent_overrides", {})
        payload["hyperparameters"]["rounds"] = int(payload["hyperparameters"].get("rounds", 4))
        return payload

    @classmethod
    def _parse_structured_response(cls, raw: str) -> dict[str, Any]:
        for candidate_text in cls._structured_candidates(raw):
            try:
                parsed = json.loads(candidate_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = yaml.safe_load(candidate_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        raise ValueError("No parseable structured object found in model response.")

    @classmethod
    def _structured_candidates(cls, raw: str) -> list[str]:
        stripped = str(raw or "").strip()
        candidates: list[str] = []
        if stripped:
            candidates.append(stripped)
        if stripped.startswith("```"):
            fence_lines = stripped.splitlines()
            if len(fence_lines) >= 3:
                inner = "\n".join(fence_lines[1:-1]).strip()
                if inner:
                    candidates.append(inner)
        balanced = cls._extract_balanced_json_object(stripped)
        if balanced:
            candidates.append(balanced)
        unique: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    @staticmethod
    def _extract_balanced_json_object(raw: str) -> str:
        start = raw.find("{")
        if start == -1:
            return ""
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(raw)):
            ch = raw[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : index + 1]
        return ""

    def _candidate_to_dict(self, candidate: NegotiationCandidate, *, redact_deadline: bool = False) -> dict[str, Any]:
        memory_summary = candidate.memory_summary
        if redact_deadline and self._memory_summary_redactor is not None:
            memory_summary = self._memory_summary_redactor(memory_summary)
        return {
            "task_id": candidate.task_id,
            "repo_name": candidate.repo_name,
            "repo_slug": candidate.repo_slug,
            "priority": candidate.priority,
            "task_kind": candidate.task_kind,
            "instance_id": candidate.instance_id,
            "description": candidate.description,
            "announced_for_seconds": round(candidate.announced_for_seconds, 2),
            "communication_wait_seconds": round(candidate.communication_wait_seconds, 2),
            # Deadlines are private information belonging to each worker's own project --
            # see the callers in _build_prompt_context, which redact these two fields for
            # every candidate representation handed to a DIFFERENT worker than the one it
            # describes, so a worker's negotiation prompt can never see a peer's real numbers.
            "deadline_minutes": "private (not disclosed to other workers)" if redact_deadline else candidate.deadline_minutes,
            "deadline_remaining_seconds": (
                "private (not disclosed to other workers)"
                if redact_deadline
                else round(candidate.deadline_remaining_seconds, 2)
            ),
            "memory_summary": memory_summary,
        }

    @staticmethod
    def _normalize_task_id(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_task_id_order(self, values: list[Any], candidate_ids: list[int]) -> list[int]:
        ordered = []
        seen: set[int] = set()
        for value in values:
            task_id = self._normalize_task_id(value)
            if task_id is None or task_id not in candidate_ids or task_id in seen:
                continue
            seen.add(task_id)
            ordered.append(task_id)
        for task_id in candidate_ids:
            if task_id not in seen:
                ordered.append(task_id)
        return ordered

    @staticmethod
    def _normalize_position(value: Any, candidate_count: int) -> int:
        try:
            position = int(value)
        except (TypeError, ValueError):
            return candidate_count
        return min(max(position, 1), candidate_count)

    def _normalize_message_payload(
        self,
        payload: list[Any],
        allowed_recipient_ids: list[int],
    ) -> list[dict[str, Any]]:
        normalized = []
        seen: set[int] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            recipient_task_id = self._normalize_task_id(item.get("recipient_task_id"))
            message = str(item.get("message") or "").strip()
            if (
                recipient_task_id is None
                or recipient_task_id not in allowed_recipient_ids
                or recipient_task_id in seen
                or not message
            ):
                continue
            seen.add(recipient_task_id)
            normalized.append(
                {
                    "recipient_task_id": recipient_task_id,
                    "message": message,
                }
            )
        return normalized

    def _normalize_coordination_decision(
        self,
        candidate: NegotiationCandidate,
        payload: Any,
        candidates: list[NegotiationCandidate],
    ) -> dict[str, Any]:
        default = self._default_coordination_decision()
        if not isinstance(payload, dict):
            return default
        action = str(payload.get("action") or "").strip().lower()
        action_aliases = {
            "wait_for_update": "wait_for_peer",
            "worker_run_first": "delegate_to_peer",
            "peer_run_first": "delegate_to_peer",
        }
        action = action_aliases.get(action, action)
        if action not in {"proceed", "wait_for_peer", "delegate_to_peer"}:
            return default
        peer_task_id = self._normalize_task_id(
            payload.get("peer_task_id", payload.get("worker_task_id"))
        )
        valid_peer_ids = {item.task_id for item in candidates if item.task_id != candidate.task_id}
        if action in {"wait_for_peer", "delegate_to_peer"} and peer_task_id not in valid_peer_ids:
            return default
        reason = str(payload.get("reason") or "").strip()
        shared_context = str(payload.get("shared_context") or "").strip()
        if action in {"wait_for_peer", "delegate_to_peer"} and not (reason or shared_context):
            return default
        return {
            "action": action,
            "peer_task_id": peer_task_id if action in {"wait_for_peer", "delegate_to_peer"} else None,
            "reason": reason,
            "shared_context": shared_context,
        }

    @staticmethod
    def _default_coordination_decision() -> dict[str, Any]:
        return {
            "action": "proceed",
            "peer_task_id": None,
            "reason": "",
            "shared_context": "",
        }

    @staticmethod
    def _format_declaration_text(payload: dict[str, Any]) -> str:
        resources = payload.get("resources_needed") or []
        resources_text = ", ".join(str(item) for item in resources) if resources else "no explicit resources listed"
        resource_summary = str(payload.get("resource_summary") or "").strip()
        estimated_cost = str(payload.get("estimated_cost") or "").strip()
        urgency = str(payload.get("urgency") or "").strip()
        sentences = [f"Resources needed: {resources_text}."]
        if resource_summary:
            sentences.append(f"Need: {resource_summary}")
        if estimated_cost:
            sentences.append(f"Cost: {estimated_cost}.")
        if urgency:
            sentences.append(f"Urgency: {urgency}.")
        # Rendered as literal labeled lines (not folded into a sentence) because the auction
        # variant's schedule-extraction regex greps for exactly this format -- see the schema
        # comment in _generate_declaration for why these three are now real structured fields
        # instead of prose the model had to remember to embed itself.
        for key, label in (
            ("desired_position", "DESIRED_POSITION"),
            ("desired_duration_seconds", "DESIRED_DURATION_SECONDS"),
            ("bid_price", "BID_PRICE"),
        ):
            raw_value = payload.get(key)
            if raw_value is None or str(raw_value).strip() == "":
                continue
            try:
                numeric_value = int(round(float(raw_value)))
            except (TypeError, ValueError):
                continue
            sentences.append(f"{label}: {numeric_value}")
        return " ".join(sentence.strip() for sentence in sentences if sentence.strip())

    def _format_reasoning_text(self, payload: dict[str, Any]) -> str:
        preferred_position = payload.get("preferred_position")
        reasoning = str(payload.get("reasoning") or "").strip()
        ordered_task_ids = payload.get("ordered_task_ids") or []
        parts = []
        if preferred_position:
            parts.append(f"Ballot preference: this worker asked for position {preferred_position}.")
        if ordered_task_ids:
            parts.append(f"Ballot suggestion: this worker proposed round order {ordered_task_ids}.")
        if reasoning:
            parts.append(f"Position reasoning: {reasoning}")
        other_positions = payload.get("other_positions") or []
        if other_positions:
            label = "worker" if "worker" in self.visible_coordination_task_field.lower() else "peer"
            parts.append(f"Suggested {label} positions: {other_positions}.")
        return " ".join(parts)

    @staticmethod
    def _format_discussion_text(
        task_id: int,
        transcript: list[dict[str, Any]],
        candidate_by_id: dict[int, NegotiationCandidate],
    ) -> str:
        related = [
            event
            for event in transcript
            if event["from_task_id"] == task_id or event["to_task_id"] == task_id
        ]
        if not related:
            return "No direct discussion messages were exchanged."
        lines = []
        for event in related:
            direction = "sent" if event["from_task_id"] == task_id else "received"
            peer_id = event["to_task_id"] if event["from_task_id"] == task_id else event["from_task_id"]
            peer_name = candidate_by_id.get(peer_id).repo_name if peer_id in candidate_by_id else f"task {peer_id}"
            lines.append(
                f"Round {event['round']} {event['phase']} {direction} {direction == 'sent' and 'to' or 'from'} "
                f"{peer_name}: {event['message']}"
            )
        return "Discussion transcript: " + " | ".join(lines)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)


def _deep_merge(*values: Any) -> Any:
    merged: Any = {}
    for value in values:
        if not isinstance(value, dict):
            merged = value
            continue
        if not isinstance(merged, dict):
            merged = {}
        for key, item in value.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(item, dict):
                merged[key] = _deep_merge(merged[key], item)
            else:
                merged[key] = item
    return merged
