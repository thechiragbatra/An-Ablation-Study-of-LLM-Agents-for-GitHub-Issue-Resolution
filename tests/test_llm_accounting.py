"""Pure accounting and budget logic. No LLM calls are made here."""

from __future__ import annotations

import math

import pytest

from agent.budget import BudgetExceeded, BudgetLedger
from agent.llm import (
    LLMError,
    LLMRequest,
    Message,
    ModelPrice,
    PricingTable,
    Usage,
    compute_cost,
    worst_case_cost,
)


@pytest.fixture
def pricing() -> PricingTable:
    return PricingTable.load()


def test_opus_pricing_matches_published_rates(pricing: PricingTable) -> None:
    p = pricing.price("claude-opus-5-5")
    assert (p.input, p.output, p.cache_read) == (4.0, 20.0, 0.20)


def test_compute_cost_all_token_kinds(pricing: PricingTable) -> None:
    p = pricing.price("claude-opus-5-5")
    usage = Usage(
        input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=500_000,
        cache_write_tokens=200_000,
    )  # fmt: skip
    expected = 4.0 + 2.0 + 0.10 + 1.0
    assert math.isclose(compute_cost(usage, p), expected)


def test_unknown_model_is_an_error(pricing: PricingTable) -> None:
    with pytest.raises(LLMError, match="no pricing"):
        pricing.price("gpt-who-knows")


def test_worst_case_dominated_by_max_tokens() -> None:
    price = ModelPrice(input=1.0, output=10.0, cache_read=0.1, cache_write=1.25)
    req = LLMRequest(
        purpose="t", system="s" * 300, messages=(Message("user", "u" * 2700),),
        model="m", max_tokens=1000, effort="low",
    )  # fmt: skip
    wc = worst_case_cost(req, price)
    # 3000 chars / 3 = 1000 est. input tokens at the cache-write rate + 1000 output.
    assert math.isclose(wc, 1e-6 * (1001 * 1.25 + 1000 * 10.0))


def test_ledger_blocks_before_spending() -> None:
    led = BudgetLedger(1.0, "run")
    led.check(0.6)
    led.record(0.6)
    with pytest.raises(BudgetExceeded) as ei:
        led.check(0.5)
    assert ei.value.scope == "run"
    assert led.spent == pytest.approx(0.6)


def test_parent_cap_enforced_and_propagated() -> None:
    exp = BudgetLedger(1.0, "experiment")
    run_a = BudgetLedger(0.8, "run", parent=exp)
    run_b = BudgetLedger(0.8, "run", parent=exp)
    run_a.record(0.7)
    assert exp.spent == pytest.approx(0.7)
    assert run_b.remaining == pytest.approx(0.3)
    with pytest.raises(BudgetExceeded) as ei:
        run_b.check(0.4)
    assert ei.value.scope == "experiment"


def test_ledger_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        BudgetLedger(0, "run")
    with pytest.raises(ValueError, match="negative"):
        BudgetLedger(1, "run").record(-0.1)
