from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: list[float], b: list[float], *, n_boot: int = 10_000, seed: int = 7
) -> dict[str, float]:
    """Bootstrap the mean paired difference a - b over questions."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired samples must have the same length")
    diff = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": float(len(diff)),
    }


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict[str, float]:
    import math

    if len(a_correct) != len(b_correct):
        raise ValueError("paired samples must have the same length")
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n = b01 + b10
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b01": 0.0, "b10": 0.0}
    statistic = (abs(b01 - b10) - 1) ** 2 / n
    p_value = math.erfc(math.sqrt(statistic / 2))
    return {"statistic": float(statistic), "p_value": float(p_value), "b01": float(b01), "b10": float(b10)}
