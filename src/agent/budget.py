"""Dollar budget guard.

A ledger is checked *before* every LLM call against a worst-case cost estimate, so a
cap can never be overshot by more than the estimation error on input tokens. Ledgers
chain: a per-run ledger has the per-experiment ledger as its parent, and both caps are
enforced on every reservation.
"""

from __future__ import annotations

import threading


class BudgetExceeded(Exception):
    def __init__(self, scope: str, cap: float, spent: float, requested: float) -> None:
        super().__init__(
            f"{scope} budget exceeded: cap=${cap:.4f} spent=${spent:.4f} "
            f"next call worst-case=${requested:.4f}"
        )
        self.scope = scope
        self.cap = cap
        self.spent = spent
        self.requested = requested


class BudgetLedger:
    def __init__(self, cap_usd: float, scope: str, parent: BudgetLedger | None = None) -> None:
        if cap_usd <= 0:
            raise ValueError("cap_usd must be positive")
        self.cap_usd = cap_usd
        self.scope = scope
        self.parent = parent
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        own = self.cap_usd - self.spent
        return own if self.parent is None else min(own, self.parent.remaining)

    def check(self, worst_case_usd: float) -> None:
        """Raise ``BudgetExceeded`` if a call costing up to ``worst_case_usd`` could breach
        this ledger's cap or any ancestor's."""
        with self._lock:
            if self._spent + worst_case_usd > self.cap_usd:
                raise BudgetExceeded(self.scope, self.cap_usd, self._spent, worst_case_usd)
        if self.parent is not None:
            self.parent.check(worst_case_usd)

    def record(self, actual_usd: float) -> None:
        if actual_usd < 0:
            raise ValueError("cost cannot be negative")
        with self._lock:
            self._spent += actual_usd
        if self.parent is not None:
            self.parent.record(actual_usd)
