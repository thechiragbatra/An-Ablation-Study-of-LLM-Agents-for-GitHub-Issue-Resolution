"""Provider-agnostic LLM access with per-call token and dollar accounting.

``LLMProvider`` is the thin provider seam (Anthropic in production, a scripted fake in
graph-routing tests). ``AccountedLLM`` wraps any provider and is the only thing nodes
call: it enforces the budget *before* each call, prices the usage *after* it, and
records the call in the active trace scope. Accounting is therefore identical for real
and fake providers.

Deliberately not used, for experimental validity: server-side model fallbacks (would
silently swap models mid-experiment). A refusal surfaces as ``LLMRefusal``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anthropic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent.budget import BudgetLedger
from agent.config import DEFAULT_PRICING_PATH, Effort
from agent.tracing import LLMCallRecord, Tracer

# Conservative chars-per-token for the pre-call worst-case estimate. Real tokenizers
# average ~3.5-4 chars/token on code; 3 over-estimates, which errs toward stopping early.
_CHARS_PER_TOKEN_ESTIMATE = 3.0


class LLMError(Exception):
    pass


class LLMRefusal(LLMError):
    pass


class LLMOutputError(LLMError):
    """Response could not be parsed into the requested schema (e.g. hit max_tokens)."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    purpose: str  # e.g. "localize", "architect"; used by fakes and traces
    system: str
    messages: tuple[Message, ...]
    model: str
    max_tokens: int
    effort: Effort


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    parsed: BaseModel | None
    usage: Usage
    stop_reason: str | None
    request_id: str | None
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResult:
    text: str
    parsed: BaseModel | None
    usage: Usage
    cost_usd: float
    latency_ms: float
    stop_reason: str | None


class LLMProvider(Protocol):
    def invoke(self, req: LLMRequest, schema: type[BaseModel] | None) -> ProviderResponse: ...


# --------------------------------------------------------------------------- pricing


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_write: float = Field(ge=0)


class PricingTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: dict[str, ModelPrice]

    @classmethod
    def load(cls, path: Path = DEFAULT_PRICING_PATH) -> PricingTable:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def price(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            raise LLMError(f"no pricing for model {model!r}; refusing to run unpriced") from None


def compute_cost(usage: Usage, price: ModelPrice) -> float:
    per_tok = 1e-6
    return per_tok * (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    )


def worst_case_cost(req: LLMRequest, price: ModelPrice) -> float:
    chars = len(req.system) + sum(len(m.content) for m in req.messages)
    est_input = int(chars / _CHARS_PER_TOKEN_ESTIMATE) + 1
    # Price input at the higher of input / cache-write rates: caching could write it all.
    in_rate = max(price.input, price.cache_write)
    return 1e-6 * (est_input * in_rate + req.max_tokens * price.output)


# --------------------------------------------------------------------------- accounted wrapper


class AccountedLLM:
    def __init__(
        self,
        provider: LLMProvider,
        pricing: PricingTable,
        ledger: BudgetLedger,
        tracer: Tracer,
    ) -> None:
        self.provider = provider
        self.pricing = pricing
        self.ledger = ledger
        self.tracer = tracer

    def complete(self, req: LLMRequest, schema: type[BaseModel] | None = None) -> LLMResult:
        price = self.pricing.price(req.model)
        self.ledger.check(worst_case_cost(req, price))  # raises BudgetExceeded

        t0 = time.perf_counter()
        resp = self.provider.invoke(req, schema)
        latency_ms = (time.perf_counter() - t0) * 1000

        cost = compute_cost(resp.usage, price)
        self.ledger.record(cost)
        artifact = self.tracer.write_artifact(
            "llm",
            {
                "purpose": req.purpose,
                "model": req.model,
                "effort": req.effort,
                "max_tokens": req.max_tokens,
                "system": req.system,
                "messages": [{"role": m.role, "content": m.content} for m in req.messages],
                "response_text": resp.text,
                "stop_reason": resp.stop_reason,
                "usage": resp.usage.__dict__,
                "cost_usd": cost,
            },
        )
        self.tracer.record_llm(
            LLMCallRecord(
                model=resp.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                cache_read_tokens=resp.usage.cache_read_tokens,
                cache_write_tokens=resp.usage.cache_write_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                stop_reason=resp.stop_reason,
                request_id=resp.request_id,
                artifact=artifact,
            )
        )

        if resp.stop_reason == "refusal":
            raise LLMRefusal(f"{req.purpose}: model refused ({resp.raw.get('stop_details')})")
        if schema is not None and resp.parsed is None:
            raise LLMOutputError(
                f"{req.purpose}: no parseable {schema.__name__} (stop_reason={resp.stop_reason})"
            )
        return LLMResult(
            text=resp.text,
            parsed=resp.parsed,
            usage=resp.usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            stop_reason=resp.stop_reason,
        )


# --------------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    """Claude via the official ``anthropic`` SDK. Structured output via ``messages.parse``."""

    def __init__(self, api_key: str | None = None, timeout_s: float = 600.0) -> None:
        # api_key=None lets the SDK resolve credentials from the environment.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)

    def invoke(self, req: LLMRequest, schema: type[BaseModel] | None) -> ProviderResponse:
        messages: list[anthropic.types.MessageParam] = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in req.messages
        ]
        msg: anthropic.types.Message
        try:
            if schema is not None:
                parsed_msg = self._client.messages.parse(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    system=req.system,
                    messages=messages,
                    output_config={"effort": req.effort},
                    output_format=schema,
                )
                parsed: BaseModel | None = parsed_msg.parsed_output
                msg = parsed_msg
            else:
                msg = self._client.messages.create(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    system=req.system,
                    messages=messages,
                    output_config={"effort": req.effort},
                )
                parsed = None
        except anthropic.APIStatusError as exc:
            raise LLMError(f"{req.purpose}: API error {exc.status_code} {exc.type}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"{req.purpose}: connection error") from exc

        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        return ProviderResponse(
            text=text,
            parsed=parsed,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.cache_read_input_tokens or 0,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
            ),
            stop_reason=msg.stop_reason,
            request_id=msg.id,
            model=msg.model,
            raw={"stop_details": msg.stop_details.model_dump() if msg.stop_details else None},
        )
