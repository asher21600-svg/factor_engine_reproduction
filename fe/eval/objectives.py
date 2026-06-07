"""Search objectives beyond raw IC fitness.

The original reproduction optimized validation ``combined_score`` only.  That is
useful for the paper-faithful ablation, but it can select high-IC factors that
do not survive turnover, yearly instability, or excess-return costs.  The V3
objective remains test-blind: it uses train/validation only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import LAGS
from .metrics import evaluate_factor, xs_corr_by_date, FactorMetrics


@dataclass
class ObjectiveResult:
    score: float
    metrics: FactorMetrics
    components: dict = field(default_factory=dict)


def complexity_counts(code: str, params: dict | None = None) -> tuple[int, int]:
    """Return (#parameters read by code, #read-but-undeclared knobs)."""
    used = set(re.findall(r'parameters\.get\(\s*["\']([^"\']+)', code or ""))
    used.discard("epsilon")
    declared = set((params or {}).keys())
    hidden = [u for u in used if u not in declared]
    return len(used), len(hidden)


def _finite(x, default=0.0) -> float:
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return default
    return x if np.isfinite(x) else default


def rank_turnover(scored: pd.DataFrame, panel: pd.DataFrame, split: str = "valid",
                  max_dates: int = 120) -> float:
    """1 - consecutive-date factor-rank autocorrelation.

    It is a cheap proxy for trading churn.  Values near 0 are stable; values near
    1 reshuffle the cross-section daily.
    """
    if "split" not in panel.columns:
        return 1.0
    keep = panel.loc[panel["split"].eq(split), ["datetime", "instrument"]]
    sub = scored.merge(keep, on=["datetime", "instrument"], how="inner")
    if sub.empty:
        return 1.0
    dates = np.array(sorted(pd.to_datetime(sub["datetime"]).unique()))
    if len(dates) > max_dates:
        idx = np.linspace(0, len(dates) - 1, max_dates).round().astype(int)
        sub = sub[sub["datetime"].isin(dates[idx])]
    wide = sub.pivot_table(index="datetime", columns="instrument", values="value").sort_index()
    if len(wide) < 3:
        return 1.0
    ranks = wide.rank(axis=1)
    ac = ranks.corrwith(ranks.shift(1), axis=1)
    if not ac.notna().any():
        return 1.0
    return float(np.clip(1.0 - ac.mean(), 0.0, 2.0))


def min_yearly_ic(scored: pd.DataFrame, panel: pd.DataFrame, split: str = "valid",
                  primary_lag: int = 5) -> float:
    """Worst yearly Pearson IC in the requested split."""
    rc = f"fwd_ret_{primary_lag}"
    if rc not in panel.columns or "split" not in panel.columns:
        return 0.0
    keep = panel.loc[panel["split"].eq(split), ["datetime", "instrument", rc]]
    sub = scored.merge(keep, on=["datetime", "instrument"], how="inner")
    if sub.empty:
        return 0.0
    ic = xs_corr_by_date(sub, "value", rc, "pearson")
    if ic.empty:
        return 0.0
    df = pd.DataFrame({"ic": ic})
    df["year"] = pd.to_datetime(df.index).year
    yearly = df.groupby("year")["ic"].mean().dropna()
    return float(yearly.min()) if len(yearly) else 0.0


def evaluate_objective(scored: pd.DataFrame, panel: pd.DataFrame, code: str = "",
                       params: dict | None = None, objective: str = "portfolio_v3",
                       split: str = "valid", lags=LAGS,
                       primary_lag: int = 5) -> ObjectiveResult:
    """Evaluate a factor and return the scalar objective used by evolution.

    ``ic_only`` is the original paper-faithful validation objective.
    ``portfolio_v3`` is the new default: train/validation robust, parsimonious,
    low-turnover, and yearly-stability-aware.
    """
    valid_m = evaluate_factor(scored, panel, lags=lags, primary_lag=primary_lag, split=split)
    if objective in ("ic", "ic_only", "combined_score"):
        return ObjectiveResult(valid_m.combined_score, valid_m, {
            "objective": "ic_only",
            "valid_combined": valid_m.combined_score,
            "valid_fitness": valid_m.fitness,
        })

    train_m = evaluate_factor(scored, panel, lags=lags, primary_lag=primary_lag, split="train")
    h_train = train_m.headline()
    h_valid = valid_m.headline()
    n_params, n_hidden = complexity_counts(code, params)
    turnover = rank_turnover(scored, panel, split=split)
    min_yic = min_yearly_ic(scored, panel, split=split, primary_lag=primary_lag)

    valid_quality = _finite(valid_m.combined_score, -1.0)
    train_quality = _finite(train_m.combined_score, -1.0)
    worst_quality = min(valid_quality, train_quality)

    train_ic = _finite(h_train.get("IC"))
    valid_ic = _finite(h_valid.get("IC"))
    valid_ric = _finite(h_valid.get("RIC"))
    sign_penalty = 0.0
    if train_ic <= 0 or valid_ic <= 0 or valid_ric <= 0:
        sign_penalty = 0.25 + 2.0 * abs(min(train_ic, valid_ic, valid_ric, 0.0))

    complexity_penalty = 0.02 * n_params + 0.12 * n_hidden
    turnover_penalty = 0.04 * (turnover * 10.0)
    stability_bonus = 2.0 * min_yic

    score = (
        0.55 * valid_quality
        + 0.45 * worst_quality
        + stability_bonus
        - turnover_penalty
        - complexity_penalty
        - sign_penalty
    )
    if not np.isfinite(score):
        score = -1e9

    return ObjectiveResult(float(score), valid_m, {
        "objective": "portfolio_v3",
        "valid_combined": valid_m.combined_score,
        "train_combined": train_m.combined_score,
        "valid_fitness": valid_m.fitness,
        "train_fitness": train_m.fitness,
        "train_ic": train_ic,
        "valid_ic": valid_ic,
        "valid_ric": valid_ric,
        "min_year_ic": min_yic,
        "turnover": turnover,
        "n_params": n_params,
        "n_hidden": n_hidden,
        "stability_bonus": stability_bonus,
        "turnover_penalty": turnover_penalty,
        "complexity_penalty": complexity_penalty,
        "sign_penalty": sign_penalty,
    })
