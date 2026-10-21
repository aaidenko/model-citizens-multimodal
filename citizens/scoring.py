"""The competition pays for correct picks, weighted by BENCHMARK probability."""
import math


def reward(p):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Benchmark probabilities must be finite and in [0, 1]')
    return -math.log2(max(p, .01))


def expected_points(q, p):
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('Forecast probabilities must be finite and in [0, 1]')
    return q * reward(p)


def choose(q, p):
    if len(q) != len(p) or not q or abs(sum(q) - 1) > 1e-7:
        raise ValueError('Forecast probabilities must describe the same complete outcome set')
    # Never normalize benchmark prices here: that would change the contest rewards.
    return max(range(len(q)), key=lambda i: expected_points(q[i], p[i]))


def underdog_threshold(p_underdog, p_favorite=None):
    if p_favorite is None:
        p_favorite = 1 - p_underdog
    a, b = reward(p_underdog), reward(p_favorite)
    return b / (a + b) if a + b else .5
