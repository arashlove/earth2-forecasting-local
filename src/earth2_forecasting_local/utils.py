from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def parse_time(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))
