"""Optional Anthropic selector. Never owns simulation, execution, or policy RPC."""

import os
import time
from typing import Annotated

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from harness.schemas import Model
from harness.skills.schemas import SkillCommand, VLAParameters
from .base import PlannerSelection
from .prompt_builder import prompt_bundle, user_content
from .response_parser import parse_response, tool_schemas, InvalidPlannerResponse


class AnthropicSettings(Model):
    # Explicit model selection: no guessed alias or silent provider fallback.
    model: Annotated[StrictStr, Field(pattern=r"^claude-[A-Za-z0-9._-]+$", max_length=100)]
    max_tokens: Annotated[StrictInt, Field(ge=128, le=4096)] = 1024
    timeout_seconds: Annotated[StrictFloat, Field(ge=1, le=60)] = 45.0
    max_calls: Annotated[StrictInt, Field(ge=1, le=100)] = 20
    max_vla_attempts: Annotated[StrictInt, Field(ge=1, le=6)] = 3
    vla: VLAParameters = Field(default_factory=lambda: VLAParameters(max_chunks=3, max_policy_actions=24))


class AnthropicPlanner:
    def __init__(self, settings, *, client=None):
        self.settings = AnthropicSettings.model_validate(settings.model_dump())
        self.prompt, self.prompt_hash = prompt_bundle()
        self.calls = self.attempts = 0
        self.history = []
        self.failed = False
        self._owned = client is None
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY_not_set")
            from anthropic import Anthropic
            # Only the SDK receives the key. No .env/shell reads or debug logging.
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                               base_url="https://api.anthropic.com",
                               max_retries=0, timeout=self.settings.timeout_seconds)
        self.client = client

    def select(self, context):
        metadata = {"provider": "anthropic", "model": self.settings.model,
                    "prompt_sha256": self.prompt_hash,
                    "memory_snapshot_sha256": context.memory.snapshot_sha256 if context.memory else None}
        if self.failed or self.calls >= self.settings.max_calls:
            return self._abort("planner_call_budget_or_terminal", metadata)
        if context.previous and (context.previous.episode_terminal or not context.previous.completed
                                 or context.previous.execution_uncertain):
            return self._abort("previous_skill_failed", metadata)
        limits = dict(vla=self.settings.vla.model_dump(),
                      vla_attempts_remaining=self.settings.max_vla_attempts-self.attempts,
                      planner_calls_remaining=self.settings.max_calls-self.calls)
        self.calls += 1  # Failed requests also consume a call.
        tick = time.perf_counter()
        try:
            response = self.client.messages.create(model=self.settings.model,
                max_tokens=self.settings.max_tokens, system=self.prompt,
                tools=tool_schemas(self.settings.vla), tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=[{"role": "user", "content": user_content(context, limits=limits, history=self.history[-8:])}])
            # Retain bounded diagnostics even when parsing rejects this response.
            # No raw content, thinking, error body or tool arguments are logged.
            stop_reason = response.get("stop_reason") if isinstance(response, dict) else response.stop_reason
            metadata["provider_stop_reason"] = (stop_reason if isinstance(stop_reason, str) and stop_reason in
                {"tool_use", "max_tokens", "end_turn", "stop_sequence", "pause_turn", "refusal"} else "other")
            usage = response.get("usage", {}) if isinstance(response, dict) else response.usage.model_dump()
            if isinstance(usage, dict):
                metadata["usage"] = {key: usage[key] for key in ("input_tokens", "output_tokens")
                                     if type(usage.get(key)) is int and usage[key] >= 0}
            command = parse_response(response)
            if command.action == "move_to" and command.parameters.target not in ("current_ee", "mug_pregrasp"):
                raise InvalidPlannerResponse("unsupported_anchor")
            if command.action == "vla_act":
                if self.attempts >= self.settings.max_vla_attempts:
                    raise InvalidPlannerResponse("vla_attempt_budget_exhausted")
                if command.stop != "mug_lifted" or any(getattr(command.parameters, key) > getattr(self.settings.vla, key)
                        for key in ("max_chunks", "chunk_size", "max_policy_actions")):
                    raise InvalidPlannerResponse("vla_call_outside_acquisition_limits")
                self.attempts += 1
        except Exception as exc:
            if isinstance(exc, InvalidPlannerResponse):
                metadata["validation_error"] = exc.code
                metadata["validation_issues"] = exc.issues
            metadata.update(error_type=type(exc).__name__, provider_call=self.calls,
                            elapsed_seconds=time.perf_counter()-tick)
            return self._abort("planner_request_or_validation_failed", metadata)
        metadata.update(provider_call=self.calls, elapsed_seconds=time.perf_counter()-tick)
        self.history.append(command.model_dump(mode="json"))
        self.history = self.history[-8:]
        phase = ("FAILED" if command.action == "abort" else "GRASP_MUG" if command.action == "vla_act"
                 else "APPROACH_MUG" if self.attempts == 0 else "RETRY_STAGE")
        if command.action == "abort":
            self.failed = True
        return PlannerSelection(command, phase, max(0, self.attempts-1), metadata)

    def _abort(self, reason, metadata):
        self.failed = True
        return PlannerSelection(SkillCommand(action="abort", reason=reason),
                                "FAILED", max(0, self.attempts-1), metadata)

    def close(self):
        if self._owned:
            self.client.close()
