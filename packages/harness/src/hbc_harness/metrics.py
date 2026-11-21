from __future__ import annotations

from typing import Iterable

import numpy as np
from hdrhistogram import Histogram


def percentile_hist(values: Iterable[float]) -> Histogram:
    hist = Histogram(1, 600_000, 3)
    for value in values:
        hist.record_value(max(1, int(value)))
    return hist


def summarize(latencies_ms: list[float], *, slo_ms: float = 6000.0) -> dict[str, float]:
    if not latencies_ms:
        return {}
    hist = percentile_hist(latencies_ms)
    good = sum(1 for v in latencies_ms if v <= slo_ms)
    return {
        "n": float(len(latencies_ms)),
        "p50_ms": float(hist.percentile_at_percentile(50)),
        "p95_ms": float(hist.percentile_at_percentile(95)),
        "p99_ms": float(hist.percentile_at_percentile(99)),
        "max_ms": float(max(latencies_ms)),
        "mean_ms": float(np.mean(latencies_ms)),
        "goodput": good / len(latencies_ms),
        "slo_ms": slo_ms,
    }
