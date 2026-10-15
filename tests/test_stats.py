from __future__ import annotations

import math

import pytest

from eval.stats import (
    bootstrap_ci,
    holm,
    majority,
    mcnemar,
    mcnemar_exact_p,
    paired_diff_ci,
    stdev,
)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (0, 5, 0.0625),  # 2 * 0.5**5
        (1, 9, 0.021484375),  # 2 * (1 + 10) / 1024
        (10, 2, 0.0385742188),  # 2 * (1 + 12 + 66) / 4096
    ],
)
def test_mcnemar_exact_known_values(b: int, c: int, expected: float) -> None:
    assert math.isclose(mcnemar_exact_p(b, c), expected, rel_tol=1e-6)
    assert mcnemar_exact_p(c, b) == mcnemar_exact_p(b, c)  # symmetric


def test_mcnemar_table_uses_shared_instances_only() -> None:
    a = {"i1": True, "i2": True, "i3": False, "i4": False, "only_a": True}
    b = {"i1": True, "i2": False, "i3": True, "i4": False, "only_b": True}
    r = mcnemar(a, b)
    assert (r.both, r.a_only, r.b_only, r.neither, r.n) == (1, 1, 1, 1, 4)
    assert r.p_value == 1.0


def test_holm_adjustment() -> None:
    adj = holm({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert adj["H1"] == pytest.approx(0.03)
    assert adj["H3"] == pytest.approx(0.06)
    assert adj["H2"] == pytest.approx(0.06)  # monotone: never below the previous step
    assert holm({"x": 0.9, "y": 0.8})["x"] == 1.0  # capped


def test_majority() -> None:
    assert majority([True, True, False])
    assert not majority([True, False, False])
    assert not majority([True, False])  # a tie is not a majority


def test_bootstrap_ci_deterministic_and_sensible() -> None:
    units = [1.0] * 30 + [0.0] * 70
    lo, hi = bootstrap_ci(units, n_resamples=2000, seed=1)
    assert (lo, hi) == bootstrap_ci(units, n_resamples=2000, seed=1)
    assert lo < 0.3 < hi
    assert 0.18 < lo < 0.26
    assert 0.34 < hi < 0.42
    assert bootstrap_ci([1.0, 1.0, 1.0], n_resamples=100) == (1.0, 1.0)
    with pytest.raises(ValueError, match="no units"):
        bootstrap_ci([])


def test_paired_diff_ci() -> None:
    a = {f"i{k}": 0.0 for k in range(40)}
    b = {f"i{k}": 1.0 if k < 20 else 0.0 for k in range(40)}
    diff, lo, hi = paired_diff_ci(a, b, n_resamples=2000)
    assert diff == pytest.approx(0.5)
    assert lo > 0.25
    assert hi < 0.75


def test_stdev() -> None:
    assert stdev([1.0]) is None
    assert stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.13809, rel=1e-4)
