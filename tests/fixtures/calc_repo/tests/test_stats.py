from calc.stats import clamp, mean


def test_mean_basic():
    assert mean([1, 2, 3]) == 2


def test_mean_empty_is_zero():
    assert mean([]) == 0.0


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(2, 0, 3) == 2
