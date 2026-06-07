"""Alpha-mini: a compact, Alpha158-flavored OHLCV feature set.

The paper merges evolved factors with Qlib's Alpha158.  We can't compute all
158 (and it is OHLCV-derived anyway), so this is a representative ~18-feature
stand-in of the same families Alpha158 uses — K-bar shape, momentum/ROC,
moving-average ratios, volatility, stochastic %K, and volume features — all
strictly backward-looking (no look-ahead).  Documented as a substitution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA_MINI_COLS: list[str] = []   # populated by alpha_mini_features()


def alpha_mini_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return [datetime, instrument, <features>] computed per instrument."""
    df = panel.sort_values(["instrument", "datetime"]).copy()
    g = df.groupby("instrument", group_keys=False)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    feats = {}

    # K-bar shape (Qlib KMID/KLEN/KUP/KLOW/KSFT families)
    rng = (h - l).replace(0, np.nan)
    feats["kmid"] = (c - o) / o
    feats["klen"] = (h - l) / o
    feats["kup"] = (h - np.maximum(o, c)) / o
    feats["klow"] = (np.minimum(o, c) - l) / o
    feats["kpos"] = (c - l) / rng                       # close position in range

    ret1 = g["close"].pct_change()
    feats["ret1"] = ret1
    for k in (5, 10, 20):
        feats[f"roc{k}"] = g["close"].pct_change(k)
        ma = g["close"].transform(lambda s, k=k: s.rolling(k, min_periods=k).mean())
        feats[f"ma{k}"] = c / ma - 1.0
        feats[f"std{k}"] = g["close"].transform(
            lambda s, k=k: s.pct_change().rolling(k, min_periods=k).std())
        hh = g["high"].transform(lambda s, k=k: s.rolling(k, min_periods=k).max())
        ll = g["low"].transform(lambda s, k=k: s.rolling(k, min_periods=k).min())
        feats[f"rsv{k}"] = (c - ll) / (hh - ll).replace(0, np.nan)
    vma5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    feats["vratio5"] = v / vma5.replace(0, np.nan)
    feats["vstd5"] = g["volume"].transform(
        lambda s: s.pct_change().rolling(5, min_periods=5).std())

    out = pd.DataFrame(feats)
    out["datetime"] = df["datetime"].values
    out["instrument"] = df["instrument"].values
    cols = [k for k in feats]
    out = out.replace([np.inf, -np.inf], np.nan)
    ALPHA_MINI_COLS.clear()
    ALPHA_MINI_COLS.extend(cols)
    return out[["datetime", "instrument"] + cols]
