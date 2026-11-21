from __future__ import annotations

from typing import Iterable

import numpy as np


def summarize(latencies_ms: list[float], *, slo_ms: float = 6000.0) -> dict[str, float]:
    """Exact percentiles over every recorded sample.

    Full samples are kept, so no histogram bucketing is needed and p99 is not
    rounded by a bucket width.
    """
    if not latencies_ms:
        return {}
    values = np.asarray(latencies_ms, dtype=float)
    good = int((values <= slo_ms).sum())
    return {
        "n": float(values.size),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "goodput": good / float(values.size),
        "slo_ms": slo_ms,
    }
