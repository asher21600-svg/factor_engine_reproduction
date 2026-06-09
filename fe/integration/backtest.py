"""Portfolio backtest — paper Appendix A.4 trading strategy & cost model,
Appendix A.3 portfolio metrics.

Strategy: rank stocks by the model score at each close; hold the top-50,
equal-weighted, for 5 days via 5 overlapping daily-rebalanced sub-portfolios
(1/5 of capital rotates each day).  Costs: bilateral commission 1.5e-4,
sell-side stamp duty 5e-4, proportional slippage 8e-4.

Metrics on the test period:
  AR   — annualized portfolio return (Eq.8)
  AER  — annualized excess return vs benchmark (Eq.9)
  IR   — annualized mean/std of the excess-return series
  SR   — annualized Sharpe of the portfolio return (rf=0, Eq.14)
  MDD  — max drawdown of the portfolio (Eq.10); RMDD relative to benchmark (Eq.12)

Simplifications (documented): equal-weight (no cap weighting); lot-size, 10%-ADV
and price-limit constraints are acknowledged but not enforced on the return
series (they bound capacity, not the per-day return much).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import (TOP_K, HOLDING_DAYS, N_SUBPORTFOLIOS, COMMISSION,
                      STAMP_DUTY, SLIPPAGE, TRADING_DAYS_PER_YEAR)


@dataclass
class BacktestResult:
    metrics: dict
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)  # date, port, bench, excess (cumulative)
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)


def _max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(np.nanmax(dd)) if len(dd) else float("nan")


def backtest(preds: pd.DataFrame, panel: pd.DataFrame,
             top_k: int = TOP_K, holding: int = HOLDING_DAYS,
             split: str = "test", benchmark: pd.Series | None = None,
             weighting: str = "equal", hysteresis: float = 0.0) -> BacktestResult:
    """preds: [datetime, instrument, pred] (any split); panel must have ret_1.

    benchmark : optional daily-return Series indexed by datetime (the real
        CSI300/CSI500 index return — paper-faithful). If None, falls back to the
        equal-weight market return (mean across the investable universe).
    weighting : 'equal' (paper default), 'rank' (weight ∝ in-basket score rank)
        or 'score' (weight ∝ positive mean-subtracted score) — conviction
        weighting (excess-return plan C1).
    hysteresis : no-trade band in [0, 1); retain currently-held names still
        within top_k*(1+band) before adding fresh names, cutting turnover/cost
        (plan C2). hysteresis=0 reproduces the paper's strict top-k exactly.
    """
    # restrict to the evaluation split
    dates_split = panel.loc[panel["split"] == split, "datetime"].unique()
    dmin = pd.Timestamp(min(dates_split))
    # daily returns wide matrix (over the split window)
    pr = panel[panel["datetime"] >= dmin][["datetime", "instrument", "ret_1"]]
    ret_wide = pr.pivot(index="datetime", columns="instrument", values="ret_1").sort_index()

    pv = preds.merge(panel[["datetime", "instrument", "split"]],
                     on=["datetime", "instrument"], how="left")
    pv = pv[pv["datetime"] >= dmin]
    pred_wide = pv.pivot_table(index="datetime", columns="instrument", values="pred").reindex(
        index=ret_wide.index, columns=ret_wide.columns)

    dates = list(ret_wide.index)
    insts = list(ret_wide.columns)
    col_ix = {c: j for j, c in enumerate(insts)}
    T, Nn = len(dates), len(insts)
    W = np.zeros((T, Nn))                      # target weights per day

    # For each rebalance day d, the tranche holds top-k for days d+1..d+holding.
    tranche = 1.0 / N_SUBPORTFOLIOS
    per_name = tranche / top_k
    buffer_k = max(top_k, int(round(top_k * (1.0 + hysteresis))))
    pred_arr = pred_wide.values
    prev: list = []
    for di in range(T):
        row = pred_arr[di]
        valid = np.where(np.isfinite(row))[0]
        if len(valid) < top_k:
            prev = []
            continue
        order = valid[np.argsort(row[valid])]          # ascending by score
        fresh = list(order[-top_k:][::-1])             # best-first
        if hysteresis > 0.0 and prev:                  # C2: no-trade band
            buf = set(order[-buffer_k:].tolist())
            sel = [i for i in prev if i in buf]        # keep still-good holdings
            for i in fresh:
                if len(sel) >= top_k:
                    break
                if i not in sel:
                    sel.append(i)
            sel = sel[:top_k]
        else:
            sel = fresh
        prev = sel
        sel_arr = np.asarray(sel, dtype=int)
        if weighting == "equal":                       # paper default
            w = np.full(len(sel_arr), per_name)
        elif weighting == "rank":                      # C1: conviction by rank
            sc = row[sel_arr]
            rk = sc.argsort().argsort().astype(float) + 1.0   # 1=worst .. n=best
            w = tranche * rk / rk.sum()
        else:                                          # 'score': positive mean-subtracted
            sc = row[sel_arr].astype(float)
            sc = sc - sc.min()
            w = tranche * (sc / sc.sum()) if sc.sum() > 0 else np.full(len(sel_arr), per_name)
        for t in range(di + 1, min(di + 1 + holding, T)):
            W[t, sel_arr] += w

    R = np.nan_to_num(ret_wide.values, nan=0.0)
    gross = np.einsum("ti,ti->t", W, R)        # daily gross portfolio return
    # benchmark: real index return if supplied, else equal-weight market
    if benchmark is not None:
        bench = (pd.Series(benchmark).reindex(pd.Index(dates))
                 .astype(float).fillna(0.0).to_numpy())
    else:
        bench = np.nanmean(ret_wide.values, axis=1)

    # costs from day-over-day weight turnover
    dW = np.diff(W, axis=0, prepend=np.zeros((1, Nn)))
    buys = np.clip(dW, 0, None).sum(axis=1)
    sells = np.clip(-dW, 0, None).sum(axis=1)
    cost = buys * (COMMISSION + SLIPPAGE) + sells * (COMMISSION + STAMP_DUTY + SLIPPAGE)
    net = gross - cost

    # drop the warm-up day 0 (no positions) and align
    mask = np.arange(T) >= 1
    net, bench, gross, cost = net[mask], bench[mask], gross[mask], cost[mask]
    dts = [d for i, d in enumerate(dates) if mask[i]]
    excess = net - bench

    n = len(net)
    ann = TRADING_DAYS_PER_YEAR
    cum_port = np.cumprod(1.0 + net)
    cum_bench = np.cumprod(1.0 + bench)
    ar = float(cum_port[-1] ** (ann / n) - 1.0) if n else float("nan")
    aer = float((cum_port[-1] / cum_bench[-1]) ** (ann / n) - 1.0) if n else float("nan")
    ir = float(np.sqrt(ann) * np.mean(excess) / (np.std(excess, ddof=1) + 1e-12))
    sr = float(np.sqrt(ann) * np.mean(net) / (np.std(net, ddof=1) + 1e-12))
    mdd = _max_drawdown(cum_port)
    rmdd = _max_drawdown(cum_port / cum_bench)

    metrics = {
        "AR": ar, "AER": aer, "IR": ir, "SR": sr,
        "MDD": mdd, "RMDD": rmdd,
        "ann_excess": float(np.mean(excess) * ann),
        "ann_turnover": float((buys[mask] + sells[mask]).mean() * ann),
        "ann_cost": float(cost.mean() * ann),
        "n_days": n,
    }
    equity = pd.DataFrame({
        "datetime": dts,
        "port": cum_port, "bench": cum_bench,
        "excess": cum_port / cum_bench,
    })
    daily = pd.DataFrame({"datetime": dts, "net": net, "bench": bench, "excess": excess})
    return BacktestResult(metrics, equity, daily)
