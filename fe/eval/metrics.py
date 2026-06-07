"""Predictive metrics, faithful to the paper's definitions.

Key fidelity point (paper Appendix A.3):
  * IC   = cross-sectional **Pearson** correlation between the factor score and
           the realized forward return, at each date.
  * RIC  = cross-sectional **Spearman** (rank) correlation  (= "Rank IC").
  * ICIR  = mean(IC) / std(IC)    (Eq.6)
  * RICIR = mean(RIC) / std(RIC)  (Eq.7)

This differs from the US factor-zoo convention (where "IC" often means Spearman);
we follow Qlib / this paper, where IC is Pearson and RankIC is Spearman.

Aggregation (paper §4.2):
  * combined_score aggregates IC/ICIR across lags {1,3,5,10} into one objective.
  * fitness (Eq.5):  FS = (IC*10 + ICIR + RIC*10 + RICIR) / 4
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import LAGS, FS_IC_MULT


# --------------------------------------------------------------------------
# Core: vectorized per-date cross-sectional correlation
# --------------------------------------------------------------------------
def xs_corr_by_date(df: pd.DataFrame, xcol: str, ycol: str,
                    method: str = "pearson",
                    date_col: str = "datetime",
                    min_obs: int = 5) -> pd.Series:
    """Series of per-date cross-sectional correlations (NaNs dropped).

    Vectorized via the grouped-covariance identity, so it is fast enough to
    call inside the evolution inner loop.
    """
    cols = [date_col, xcol, ycol]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.Series(dtype=float)

    if method == "spearman":
        d = d.assign(
            _x=d.groupby(date_col)[xcol].rank(),
            _y=d.groupby(date_col)[ycol].rank(),
        )
        xc, yc = "_x", "_y"
    else:
        xc, yc = xcol, ycol

    g = d.groupby(date_col)
    n = g[xc].transform("size")
    d = d[n >= min_obs]
    if d.empty:
        return pd.Series(dtype=float)

    g = d.groupby(date_col)
    mx = g[xc].transform("mean")
    my = g[yc].transform("mean")
    dx = d[xc] - mx
    dy = d[yc] - my
    tmp = pd.DataFrame({
        date_col: d[date_col].values,
        "_cxy": (dx * dy).values,
        "_cxx": (dx * dx).values,
        "_cyy": (dy * dy).values,
    })
    s = tmp.groupby(date_col)[["_cxy", "_cxx", "_cyy"]].sum()
    denom = np.sqrt(s["_cxx"] * s["_cyy"])
    corr = s["_cxy"] / denom.replace(0.0, np.nan)
    return corr.dropna().sort_index()


def summarize_ic(ic: pd.Series) -> dict:
    """{mean, std, ir, hit} from a daily IC (or RIC) series."""
    ic = ic.dropna()
    if len(ic) < 2:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "hit": np.nan, "n": len(ic)}
    m = float(ic.mean())
    s = float(ic.std(ddof=1))
    return {
        "mean": m,
        "std": s,
        "ir": (m / s) if s > 0 else np.nan,   # ICIR / RICIR
        "hit": float((ic > 0).mean()),
        "n": int(len(ic)),
    }


def fitness_from_components(ic: float, icir: float, ric: float, ricir: float) -> float:
    """Eq.5: FS = (IC*10 + ICIR + RIC*10 + RICIR) / 4."""
    vals = [ic * FS_IC_MULT, icir, ric * FS_IC_MULT, ricir]
    if any(not np.isfinite(v) for v in vals):
        return float("-inf")
    return float(sum(vals) / 4.0)


@dataclass
class FactorMetrics:
    """Full metric bundle for one factor over one panel."""
    per_lag: dict = field(default_factory=dict)   # lag -> {ic,icir,ric,ricir,fitness}
    combined_score: float = float("-inf")          # Bayesian objective (mean fitness over lags)
    fitness: float = float("-inf")                  # fitness at the primary lag
    primary_lag: int = 5
    n_dates: int = 0
    error: str | None = None

    def headline(self) -> dict:
        """The four numbers the paper prints in Table 1 (at the primary lag)."""
        pl = self.per_lag.get(self.primary_lag, {})
        return {
            "IC": pl.get("ic", np.nan),
            "ICIR": pl.get("icir", np.nan),
            "RIC": pl.get("ric", np.nan),
            "RICIR": pl.get("ricir", np.nan),
        }

    def as_row(self) -> dict:
        h = self.headline()
        return {**h, "fitness": self.fitness, "combined_score": self.combined_score,
                "n_dates": self.n_dates}


def evaluate_factor(scored: pd.DataFrame, panel: pd.DataFrame,
                    value_col: str = "value",
                    lags=LAGS, primary_lag: int = 5,
                    split: str | None = None) -> FactorMetrics:
    """Evaluate a factor output against forward returns.

    Args:
      scored : DataFrame [datetime, instrument, value_col] — factor output.
      panel  : DataFrame with [datetime, instrument, fwd_ret_{h}, split].
      split  : if given, restrict to that split ('train'/'valid'/'test').
    """
    ret_cols = [f"fwd_ret_{h}" for h in lags]
    keep = ["datetime", "instrument"] + [c for c in ret_cols if c in panel.columns]
    if split is not None and "split" in panel.columns:
        keep_panel = panel[panel["split"] == split][keep]
    else:
        keep_panel = panel[keep]

    merged = scored.merge(keep_panel, on=["datetime", "instrument"], how="inner")
    if merged.empty or merged[value_col].notna().sum() < 10:
        return FactorMetrics(error="empty/degenerate factor output")

    per_lag = {}
    fitnesses = []
    n_dates = 0
    for h in lags:
        rc = f"fwd_ret_{h}"
        if rc not in merged.columns:
            continue
        ic = xs_corr_by_date(merged, value_col, rc, method="pearson")
        ric = xs_corr_by_date(merged, value_col, rc, method="spearman")
        sic = summarize_ic(ic)
        sric = summarize_ic(ric)
        fit = fitness_from_components(sic["mean"], sic["ir"], sric["mean"], sric["ir"])
        per_lag[h] = {
            "ic": sic["mean"], "icir": sic["ir"], "ic_hit": sic["hit"],
            "ric": sric["mean"], "ricir": sric["ir"],
            "fitness": fit, "n_dates": sic["n"],
        }
        if np.isfinite(fit):
            fitnesses.append(fit)
        n_dates = max(n_dates, sic["n"])

    combined = float(np.mean(fitnesses)) if fitnesses else float("-inf")
    prim = per_lag.get(primary_lag, {})
    return FactorMetrics(
        per_lag=per_lag,
        combined_score=combined,
        fitness=prim.get("fitness", float("-inf")),
        primary_lag=primary_lag,
        n_dates=n_dates,
    )
