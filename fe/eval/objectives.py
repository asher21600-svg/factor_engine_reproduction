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

from ..config import LAGS, COMMISSION, STAMP_DUTY, SLIPPAGE
from .metrics import evaluate_factor, xs_corr_by_date, FactorMetrics

# round-trip trading cost (buy then sell) under the paper's A-share cost model
ROUND_TRIP_COST = 2 * COMMISSION + STAMP_DUTY + 2 * SLIPPAGE   # ~2.4e-3


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


def quantile_spread(scored: pd.DataFrame, panel: pd.DataFrame, split: str = "valid",
                    primary_lag: int = 5, q: float = 0.1, max_dates: int = 150) -> float:
    """Cheap, *return-aware* (not IC) signal proxy for the portfolio_v4 objective:
    the mean daily top-q **excess** forward return (top-decile mean minus the
    cross-sectional mean) on the split. This is the gross return of holding the
    factor's top names vs the market each day — what the backtest actually trades —
    without paying for a full tranche backtest inside the evolution loop."""
    rc = f"fwd_ret_{primary_lag}"
    if rc not in panel.columns or "split" not in panel.columns:
        return 0.0
    keep = panel.loc[panel["split"].eq(split), ["datetime", "instrument", rc]]
    sub = scored.merge(keep, on=["datetime", "instrument"], how="inner").dropna(subset=["value", rc])
    if sub.empty:
        return 0.0
    dates = np.array(sorted(pd.to_datetime(sub["datetime"]).unique()))
    if len(dates) > max_dates:                       # subsample for in-loop speed
        idx = np.linspace(0, len(dates) - 1, max_dates).round().astype(int)
        sub = sub[sub["datetime"].isin(dates[idx])]
    spreads = []
    for _, g in sub.groupby("datetime"):
        if len(g) < 20:
            continue
        thr = g["value"].quantile(1.0 - q)
        top = g.loc[g["value"] >= thr, rc]
        if len(top):
            spreads.append(float(top.mean() - g[rc].mean()))
    return float(np.mean(spreads)) if spreads else 0.0


def quantile_set_turnover(scored: pd.DataFrame, panel: pd.DataFrame, split: str = "valid",
                          q: float = 0.1, max_dates: int = 250) -> float:
    """Daily fraction of the top-q holding set that rotates out, day to day (∈ [0,1]) —
    the portfolio's natural turnover. With the A-share round-trip cost this gives a cheap
    cost estimate for the portfolio_v5 objective, no full backtest needed. Uses CONSECUTIVE
    recent dates (turnover needs adjacency), so it caps to the last `max_dates`."""
    if "split" not in panel.columns:
        return 1.0
    keep = panel.loc[panel["split"].eq(split), ["datetime", "instrument"]]
    sub = scored.merge(keep, on=["datetime", "instrument"], how="inner").dropna(subset=["value"])
    if sub.empty:
        return 1.0
    dates = sorted(pd.to_datetime(sub["datetime"]).unique())[-max_dates:]
    sub = sub[sub["datetime"].isin(dates)]
    prev, fracs = None, []
    for _, g in sub.groupby("datetime"):
        n = max(1, int(round(q * len(g))))
        top = set(g.nlargest(n, "value")["instrument"])
        if prev is not None and top:
            fracs.append(len(top - prev) / len(top))     # newly bought (= sold) fraction
        prev = top
    return float(np.mean(fracs)) if fracs else 1.0


def evaluate_objective(scored: pd.DataFrame, panel: pd.DataFrame, code: str = "",
                       params: dict | None = None, objective: str = "portfolio_v3",
                       split: str = "valid", lags=LAGS,
                       primary_lag: int = 5) -> ObjectiveResult:
    """Evaluate a factor and return the scalar objective used by evolution.

    ``ic_only`` is the original paper-faithful validation objective.
    ``portfolio_v3`` is the new default: train/validation robust, parsimonious,
    low-turnover, and yearly-stability-aware.
    ``portfolio_v4`` is return-aware: it optimizes the annualized gross top-decile
    *excess* return (train AND validation) instead of IC, still penalizing turnover,
    complexity, and sign-inconsistency — for evolving factors that lift excess
    return, not just IC. (Definitive v3-vs-v4 needs a fresh evolution run.)
    ``portfolio_v5`` is cost-aware: it optimizes annualized **net** excess return =
    gross top-decile excess − (top-decile set-turnover × A-share round-trip cost), so
    the search evolves factors that are tradeable net, not just gross-predictive.
    """
    known = {"ic", "ic_only", "combined_score", "portfolio_v3", "portfolio_v4", "portfolio_v5"}
    if objective not in known:
        # fail loudly rather than silently defaulting to portfolio_v3 — a stale build that
        # lacks a new branch would otherwise mislabel a whole run (see Result 8/9 history).
        raise ValueError(f"unknown objective {objective!r}; known: {sorted(known)}")
    valid_m = evaluate_factor(scored, panel, lags=lags, primary_lag=primary_lag, split=split)
    if objective in ("ic", "ic_only", "combined_score"):
        return ObjectiveResult(valid_m.combined_score, valid_m, {
            "objective": "ic_only",
            "valid_combined": valid_m.combined_score,
            "valid_fitness": valid_m.fitness,
        })

    if objective == "portfolio_v4":                  # D1: return-aware objective
        sp_v = quantile_spread(scored, panel, split=split, primary_lag=primary_lag)
        sp_t = quantile_spread(scored, panel, split="train", primary_lag=primary_lag)
        worst_sp = min(sp_v, sp_t)
        turnover = rank_turnover(scored, panel, split=split)
        min_yic = min_yearly_ic(scored, panel, split=split, primary_lag=primary_lag)
        n_params, n_hidden = complexity_counts(code, params)
        sign_pen = 0.0 if (sp_t > 0 and sp_v > 0) else 0.25 + 5.0 * abs(min(sp_t, sp_v, 0.0))
        ann = 252.0
        spread_term = ann * (0.55 * sp_v + 0.45 * worst_sp)   # annualized gross top-decile excess
        score = (spread_term + 1.0 * min_yic
                 - 0.04 * (turnover * 10.0)
                 - (0.02 * n_params + 0.12 * n_hidden) - sign_pen)
        if not np.isfinite(score):
            score = -1e9
        return ObjectiveResult(float(score), valid_m, {
            "objective": "portfolio_v4",
            "ann_spread_valid": ann * sp_v, "ann_spread_train": ann * sp_t,
            "min_year_ic": min_yic, "turnover": turnover,
            "n_params": n_params, "n_hidden": n_hidden, "sign_penalty": sign_pen,
            "valid_combined": valid_m.combined_score, "valid_fitness": valid_m.fitness,
        })

    if objective == "portfolio_v5":                  # cost-net objective (evolve tradeable factors)
        sp_v = quantile_spread(scored, panel, split=split, primary_lag=primary_lag)
        sp_t = quantile_spread(scored, panel, split="train", primary_lag=primary_lag)
        to_v = quantile_set_turnover(scored, panel, split=split)
        to_t = quantile_set_turnover(scored, panel, split="train")
        worst_sp, worst_to = min(sp_v, sp_t), max(to_v, to_t)
        n_params, n_hidden = complexity_counts(code, params)
        min_yic = min_yearly_ic(scored, panel, split=split, primary_lag=primary_lag)
        sign_pen = 0.0 if (sp_t > 0 and sp_v > 0) else 0.25 + 5.0 * abs(min(sp_t, sp_v, 0.0))
        ann = 252.0
        gross = ann * (0.55 * sp_v + 0.45 * worst_sp)                  # annualized gross top-decile excess
        cost = ann * ROUND_TRIP_COST * (0.55 * to_v + 0.45 * worst_to)  # annualized trading cost
        score = (gross - cost) + 1.0 * min_yic - (0.02 * n_params + 0.12 * n_hidden) - sign_pen
        if not np.isfinite(score):
            score = -1e9
        return ObjectiveResult(float(score), valid_m, {
            "objective": "portfolio_v5",
            "ann_gross": gross, "ann_cost": cost, "ann_net": gross - cost,
            "set_turnover_valid": to_v, "min_year_ic": min_yic,
            "n_params": n_params, "n_hidden": n_hidden, "sign_penalty": sign_pen,
            "valid_combined": valid_m.combined_score, "valid_fitness": valid_m.fitness,
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
