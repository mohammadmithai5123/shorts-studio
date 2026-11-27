"""Ranking signals that do not depend on how well speech was recognised.

A signal is a per-second series (loudness in dB, or YouTube's "most replayed" heat). For a clip, the
interesting question is "how does this stretch compare with every other stretch of the same length in
the video?" — answered as a percentile (50 = typical, 95 = more intense than 95% of them), which makes
different signals comparable and blendable.
"""
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

PctFn = Callable[[float, float], float]


def window_pct_fn(per_second: Sequence[float]) -> PctFn:
    """pct(start, end) -> 0..100: this window's mean against every window of the same length."""
    series = np.asarray(per_second, dtype=np.float64)
    n = len(series)
    cs = np.concatenate([[0.0], np.cumsum(series)])

    def pct(start: float, end: float) -> float:
        s, e = int(max(0, start)), int(min(n, end))
        length = e - s
        if length < 3 or length >= n:
            return 50.0
        windows = (cs[length:] - cs[:-length]) / length
        return float((windows < (cs[e] - cs[s]) / length).mean() * 100.0)

    return pct


def heatmap_series(heatmap: Optional[List[Dict]], duration: float) -> Optional[np.ndarray]:
    """YouTube 'most replayed' points -> a per-second series (None when the video has none)."""
    if not heatmap or duration <= 0:
        return None
    n = int(np.ceil(duration))
    series = np.zeros(n)
    for point in heatmap:
        a, b = float(point.get("start_time", 0)), float(point.get("end_time", 0))
        series[int(a):max(int(a) + 1, min(n, int(np.ceil(b))))] = float(point.get("value", 0))
    return series


def downsample(series: Optional[Sequence[float]], points: int = 400) -> List[float]:
    """Shrink a long series to ``points`` values (mean per bucket) for drawing a timeline."""
    if series is None or len(series) == 0:
        return []
    arr = np.asarray(series, dtype=np.float64)
    if len(arr) <= points:
        return [round(float(v), 3) for v in arr]
    edges = np.linspace(0, len(arr), points + 1).astype(int)
    return [round(float(arr[a:b].mean()), 3) for a, b in zip(edges[:-1], edges[1:])]
