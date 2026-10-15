"""Scripted LLM provider. Used ONLY by graph-routing tests (tests/test_graph.py).

It replaces the network call and nothing else: pricing, budget checks and tracing still
run through ``AccountedLLM`` exactly as in production, and the sandbox is real Docker.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from pydantic import BaseModel

from agent.llm import LLMRequest, ProviderResponse, Usage


@dataclass(frozen=True)
class Scripted:
    output: BaseModel | None
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=1000, output_tokens=500))


class FakeLLM:
    MODEL = "fake-llm"

    def __init__(self, script: dict[str, list[Scripted | BaseModel]]) -> None:
        self._script: dict[str, deque[Scripted]] = defaultdict(deque)
        for purpose, items in script.items():
            for item in items:
                self._script[purpose].append(
                    item if isinstance(item, Scripted) else Scripted(output=item)
                )
        self.requests: list[LLMRequest] = []

    def invoke(self, req: LLMRequest, schema: type[BaseModel] | None) -> ProviderResponse:
        self.requests.append(req)
        queue = self._script[req.purpose]
        if not queue:
            raise AssertionError(f"FakeLLM script exhausted for purpose {req.purpose!r}")
        item = queue.popleft()
        if item.output is not None and schema is not None:
            assert isinstance(item.output, schema), (type(item.output), schema)
        return ProviderResponse(
            text=item.output.model_dump_json() if item.output is not None else "",
            parsed=item.output,
            usage=item.usage,
            stop_reason=item.stop_reason,
            request_id=f"fake-{len(self.requests)}",
            model=self.MODEL,
        )

    def remaining(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._script.items() if v}
