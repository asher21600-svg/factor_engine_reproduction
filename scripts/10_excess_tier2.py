#!/usr/bin/env python
"""Excess-return plan tier 2 — B-group (docs/EXCESS_RETURN_PLAN.md).

  B1b  size + beta neutralization: residualize the combined signal vs a size proxy
       AND market beta (vs size only in B1). Reuses the cached preds (fast).
  B2   IC-weighted linear combine: instead of the LightGBM tree-merge, weight each
       feature (Alpha158 + the A3 mutually-orthogonalized elites) by its validation
       IC, z-scored per date, and sum -> a return-aware linear composite. Rebuilds
       the A3 feature matrix.

(Industry neutralization is omitted: the free Qlib bundle ships no sector
classification; beta is the available systematic-tilt control.)

Compared against the V3 baseline / A3 / A3+B1 numbers in outputs/excess_return.json.
Saves outputs/excess_tier2.json.

    python scripts/10_excess_tier2.py [--universes csi300 csi500]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fe import config
from fe.factors import score_factor
from fe.eval import xs_corr_by_date
from fe.integration import build_feature_matrix, backtest
from fe.integration.robust_elite import orthogonalize_vs_features

IDX = ["datetime", "instrument"]
BT_KEYS = ("AER", "IR", "AR", "SR", "MDD", "RMDD", "ann_turnover", "ann_cost")


def load_benchmark(uni):
    try:
        from fe.data.qlib_loader import load_benchmark as _lb
        return _lb(uni)
    except Exception as e:  # noqa: BLE001
        print(f"  [benchmark unavailable: {e}] -> equal-weight market")
        return None


def size_proxy(panel):
    df = panel[["datetime", "instrument", "close", "volume"]].copy()
    df["size"] = np.log1p((df["close"] * df["volume"]).clip(lower=0))
    df = df.sort_values(["instrument", "datetime"])
    df["size"] = df.groupby("instrument")["size"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    return df[["datetime", "instrument", "size"]].dropna()


def beta_proxy(panel, bench, win=60):
    """Rolling market beta per stock = cov(ret, mkt)/var(mkt). Uses the index if
    available, else the equal-weight market return."""
    df = panel[["datetime", "instrument", "ret_1"]].copy()
    if bench is not None:
        mkt = pd.Series(bench).astype(float)
        df["mkt"] = df["datetime"].map(mkt)
    else:
        mkt = df.groupby("datetime")["ret_1"].transform("mean")
        df["mkt"] = mkt
    df = df.sort_values(["instrument", "datetime"])

    def _beta(g):
        cov = g["ret_1"].rolling(win, min_periods=20).cov(g["mkt"])
        var = g["mkt"].rolling(win, min_periods=20).var()
        return cov / var.replace(0, np.nan)

    df["beta"] = df.groupby("instrument", group_keys=False).apply(_beta)
    return df[["datetime", "instrument", "beta"]].dropna()


def neutralize(preds, base_df, cols):
    scored = preds.rename(columns={"pred": "value"})[IDX + ["value"]]
    r = orthogonalize_vs_features(scored, base_df, list(cols))
    return r.rename(columns={"value": "pred"})


def bt(preds, panel, bench, **kw):
    m = backtest(preds, panel, split="test", benchmark=bench, **kw).metrics
    return {k: round(float(m[k]), 5) for k in BT_KEYS}


def a3_features(panel, elite_specs, baseline="alpha158"):
    """Alpha158 + the elites mutually orthogonalized (each vs Alpha158 + prior elites)."""
    base, base_cols = build_feature_matrix(panel, [], with_baseline=True, baseline=baseline)
    run_long, run_cols = base[IDX + list(base_cols)].copy(), list(base_cols)
    parts, cols = [base.set_index(IDX)], list(base_cols)
    for i, (name, code, params) in enumerate(elite_specs):
        sc = score_factor(code, panel, params)
        r = orthogonalize_vs_features(sc, run_long, list(run_cols)).rename(columns={"value": f"m_{i}"})
        parts.append(r.set_index(IDX))
        cols.append(f"m_{i}")
        run_long = run_long.merge(r, on=IDX, how="left")
        run_long[f"m_{i}"] = run_long[f"m_{i}"].fillna(0.0)
        run_cols.append(f"m_{i}")
    feat = pd.concat(parts, axis=1).reset_index()
    return feat, cols


def ic_weighted_pred(panel, feat, cols):
    """B2: per-date z-score each feature, weight by validation IC (signed), sum."""
    vkeep = panel.loc[panel["split"].eq("valid"), IDX + ["fwd_ret_5"]]
    mv = feat.merge(vkeep, on=IDX, how="inner")
    w = {}
    for c in cols:
        ic = xs_corr_by_date(mv, c, "fwd_ret_5", "pearson")
        w[c] = float(ic.mean()) if len(ic) else 0.0
    g = feat.groupby("datetime")
    comp = np.zeros(len(feat), dtype="float64")
    for c in cols:
        if not np.isfinite(w[c]) or w[c] == 0.0:
            continue
        mean = g[c].transform("mean")
        std = g[c].transform("std").replace(0, np.nan)
        z = ((feat[c] - mean) / std).fillna(0.0).to_numpy()
        comp += w[c] * z
    return feat[IDX].assign(pred=comp)


def run_universe(uni):
    panel = pd.read_parquet(config.OUTPUTS / f"{uni}_panel.parquet")
    bench = load_benchmark(uni)
    with open(config.OUTPUTS / "evolution.json") as f:
        evo = json.load(f)
    elite = evo.get("elite") or []
    elite_specs = [(f"fe_elite_{i}", e["code"], e["params"]) for i, e in enumerate(elite)] \
        or [("fe_evolved", evo["best_code"], evo["best_params"])]
    prev = json.load(open(config.OUTPUTS / "excess_return.json")).get(uni, {}).get("levers", {})
    print(f"  {uni}: {len(elite_specs)} elites, benchmark={'index' if bench is not None else 'eqw'}")

    size_df = size_proxy(panel)
    beta_df = beta_proxy(panel, bench)
    sb = size_df.merge(beta_df, on=IDX, how="inner")            # size + beta

    levers = {}
    # carry forward the key tier-1 references for an apples-to-apples table
    for k in ("V3_baseline", "A3_mutual_ortho", "A3_plus_B1"):
        if k in prev:
            levers[k] = prev[k]

    # --- B1b: size+beta neutralization on the cached base & A3 predictions ---
    pb = config.OUTPUTS / f"preds_{uni}_base.parquet"
    pa = config.OUTPUTS / f"preds_{uni}_a3.parquet"
    if pb.exists():
        levers["B1b_base_size+beta"] = bt(neutralize(pd.read_parquet(pb), sb, ["size", "beta"]), panel, bench)
    if pa.exists():
        levers["A3+B1b_size+beta"] = bt(neutralize(pd.read_parquet(pa), sb, ["size", "beta"]), panel, bench)
    else:
        print("  [no cached A3 preds — run scripts/09_excess_return.py first for B1b]")

    # --- B2: IC-weighted linear combine over the A3 feature set (no LGBM) ---
    feat, cols = a3_features(panel, elite_specs)
    preds_b2 = ic_weighted_pred(panel, feat, cols)
    levers["B2_ic_weighted"] = bt(preds_b2, panel, bench)
    levers["B2_ic_weighted+B1"] = bt(neutralize(preds_b2, sb, ["size", "beta"]), panel, bench)

    return {"benchmark": "index" if bench is not None else "equal_weight", "levers": levers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*", default=["csi300", "csi500"])
    args = ap.parse_args()
    out = {}
    for uni in args.universes:
        print(f"\n== {uni} ==")
        out[uni] = run_universe(uni)
        lv = out[uni]["levers"]
        base = lv.get("V3_baseline", {})
        b_aer = base.get("AER", 0)
        print(f"  {'lever':22s} {'AER':>8} {'IR':>7} {'AR':>8} {'SR':>7} {'turnover':>9}  dAER")
        for name, m in lv.items():
            d = m["AER"] - b_aer
            print(f"  {name:22s} {m['AER']:+8.4f} {m['IR']:+7.3f} {m['AR']:+8.4f} "
                  f"{m['SR']:+7.3f} {m['ann_turnover']:9.2f}  {d:+.4f}")
    with open(config.OUTPUTS / "excess_tier2.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'excess_tier2.json'}")


if __name__ == "__main__":
    main()
