"""Faithful Alpha158 feature set in pure pandas/NumPy — NO qlib package needed.

Reimplements Qlib's Alpha158 (KBAR + price + rolling families) using vectorized
wide-matrix (date x instrument) rolling ops, so it runs without the qlib C/Cython
package.  Covers ~128 of the 158 features: KBAR (9), price ratios (4), and the
vectorizable rolling operators over windows [5,10,20,30,60]:
  ROC, MA, STD, MAX, MIN, QTLU, QTLD, RANK, RSV, CORR, CORD,
  CNTP/CNTN/CNTD, SUMP/SUMN/SUMD, VMA, VSTD, WVMA, VSUMP/VSUMN/VSUMD.
Omits the rolling-OLS (BETA/RSQR/RESI) and argmax-index (IMAX/IMIN/IMXD)
families (30 features) for speed; the rest are the high-signal bulk of Alpha158.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index="datetime", columns="instrument", values=col).sort_index()


def _roll_corr(a: pd.DataFrame, b: pd.DataFrame, d: int) -> pd.DataFrame:
    ma, mb = a.rolling(d).mean(), b.rolling(d).mean()
    cov = (a * b).rolling(d).mean() - ma * mb
    sa = (a * a).rolling(d).mean() - ma * ma
    sb = (b * b).rolling(d).mean() - mb * mb
    return cov / (np.sqrt(sa.clip(lower=0) * sb.clip(lower=0)) + EPS)


def alpha158_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Return (features_df [datetime, instrument, <~128 cols>], cols).

    Memory-efficient: each feature's wide matrix is stacked straight into the
    panel-indexed result and then freed (never holds all 128 wide matrices).
    """
    C = _wide(panel, "close")
    O = _wide(panel, "open")
    H = _wide(panel, "high")
    L = _wide(panel, "low")
    V = _wide(panel, "volume")
    VW = _wide(panel, "vwap") if "vwap" in panel.columns else (H + L + C) / 3.0

    idx = pd.MultiIndex.from_arrays(
        [panel["datetime"].to_numpy(), panel["instrument"].to_numpy()],
        names=["datetime", "instrument"])
    out = {"datetime": panel["datetime"].to_numpy(),
           "instrument": panel["instrument"].to_numpy()}
    cols: list[str] = []

    def add(name, mat):
        cols.append(name)
        s = mat.replace([np.inf, -np.inf], np.nan).stack()       # drops NaN cells
        out[name] = s.reindex(idx).to_numpy(dtype="float32")

    rng = (H - L).replace(0, np.nan)
    maxoc, minoc = np.maximum(O, C), np.minimum(O, C)

    # --- KBAR (9) ---
    add("KMID", (C - O) / O); add("KLEN", (H - L) / O); add("KMID2", (C - O) / (rng + EPS))
    add("KUP", (H - maxoc) / O); add("KUP2", (H - maxoc) / (rng + EPS))
    add("KLOW", (minoc - L) / O); add("KLOW2", (minoc - L) / (rng + EPS))
    add("KSFT", (2 * C - H - L) / O); add("KSFT2", (2 * C - H - L) / (rng + EPS))
    # --- price ratios (4) ---
    add("OPEN0", O / C); add("HIGH0", H / C); add("LOW0", L / C); add("VWAP0", VW / C)

    # reused helpers
    up = (C > C.shift(1)).astype(float)
    dn = (C < C.shift(1)).astype(float)
    dC = C - C.shift(1)
    posC, negC, absC = dC.clip(lower=0), (-dC).clip(lower=0), dC.abs()
    dV = V - V.shift(1)
    posV, negV, absV = dV.clip(lower=0), (-dV).clip(lower=0), dV.abs()
    logV = np.log(V + 1.0)
    dlogV = logV - logV.shift(1)
    wv = (C / C.shift(1) - 1.0).abs() * V

    for d in WINDOWS:
        add(f"ROC{d}", C.shift(d) / C)
        add(f"MA{d}", C.rolling(d).mean() / C)
        add(f"STD{d}", C.rolling(d).std() / C)
        add(f"MAX{d}", H.rolling(d).max() / C)
        add(f"MIN{d}", L.rolling(d).min() / C)
        add(f"QTLU{d}", C.rolling(d).quantile(0.8) / C)
        add(f"QTLD{d}", C.rolling(d).quantile(0.2) / C)
        add(f"RANK{d}", C.rolling(d).rank(pct=True))
        hh, ll = H.rolling(d).max(), L.rolling(d).min()
        add(f"RSV{d}", (C - ll) / (hh - ll + EPS))
        add(f"CORR{d}", _roll_corr(C, logV, d))
        add(f"CORD{d}", _roll_corr(C / C.shift(1), dlogV, d))
        cntp, cntn = up.rolling(d).mean(), dn.rolling(d).mean()
        add(f"CNTP{d}", cntp); add(f"CNTN{d}", cntn); add(f"CNTD{d}", cntp - cntn)
        sp, sn, sa = posC.rolling(d).sum(), negC.rolling(d).sum(), absC.rolling(d).sum()
        add(f"SUMP{d}", sp / (sa + EPS)); add(f"SUMN{d}", sn / (sa + EPS))
        add(f"SUMD{d}", (sp - sn) / (sa + EPS))
        add(f"VMA{d}", V.rolling(d).mean() / (V + EPS))
        add(f"VSTD{d}", V.rolling(d).std() / (V + EPS))
        add(f"WVMA{d}", wv.rolling(d).std() / (wv.rolling(d).mean() + EPS))
        vsp, vsn, vsa = posV.rolling(d).sum(), negV.rolling(d).sum(), absV.rolling(d).sum()
        add(f"VSUMP{d}", vsp / (vsa + EPS)); add(f"VSUMN{d}", vsn / (vsa + EPS))
        add(f"VSUMD{d}", (vsp - vsn) / (vsa + EPS))

    return pd.DataFrame(out), cols
