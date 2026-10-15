"""Small statistics helpers."""


def mean(values):
    """Return the arithmetic mean of ``values``.

    An empty sequence has mean 0.0 by convention in this project.
    """
    return sum(values) / len(values)


def clamp(x, lo, hi):
    """Clamp ``x`` into the closed interval [lo, hi]."""
    return max(lo, min(x, hi))
