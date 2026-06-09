#!/usr/bin/env python
"""Excess-return quick wins (docs/EXCESS_RETURN_PLAN.md) — A/B vs the V3 baseline.

Levers, each measured against the V3 augmented model (Alpha158 + elite factors,
each residualized vs the baseline) backtested with the real index benchmark:

  B1  size-neutralize the combined signal (residualize pred vs a dollar-volume
      size proxy, per date) -> strip the A-share size tilt the index already prices.
  C1  rank-weighted holdings instead of equal-weight top-50 (conviction weighting).
  C2  turnover hysteresis (no-trade band) -> cut the A-share cost drag.
  A3  mutual orthogonalization: residualize each elite vs Alpha158 + the already
      selected elites (not just vs Alpha158) -> complementary, non-redundant alpha.
  stack  B1 + C1 + C2 together.

Primary metrics: AER (annualized excess return vs index) and IR (excess/TE), net
of A-share costs; reported on the 2017-2024 test split for both universes.

    python scripts/09_excess_return.py [--universes csi300 csi500]
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
from fe.integration import build_feature_matrix, train_predict, backtest
from fe.integration.robust_elite import orthogonalize_vs_features

IDX = ["datetime", "instrument"]
LABEL, LABEL_MODE = "label", "date_demeaned"   # the V3 model target
BT_KEYS = ("AER", "IR", "AR", "SR", "MDD", "RMDD", "ann_turnover", "ann_cost")


def load_benchmark(uni):
    """Real CSI index daily-return series, or None (-> equal-weight market)."""
    try:
        from fe.data.qlib_loader import load_benchmark as _lb
        return _lb(uni)
    except Exception as e:  # noqa: BLE001
        print(f"  [benchmark unavailable: {e}] -> equal-weight market")
        return None


def size_proxy(panel):
    """A-share size/liquidity proxy: 20-day mean of log dollar volume (no market
    cap in the free bundle). Returns [datetime, instrument, size]."""
    df = panel[["datetime", "instrument", "close", "volume"]].copy()
    df["size"] = np.log1p((df["close"] * df["volume"]).clip(lower=0))
    df = df.sort_values(["instrument", "datetime"])
    df["size"] = df.groupby("instrument")["size"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    return df[["datetime", "instrument", "size"]].dropna()


def build_arms(panel, elite_specs, baseline="alpha158"):
    """Build the V3-baseline feature matrix (elites independently orthogonalized vs
    Alpha158) and the A3 matrix (elites mutually orthogonalized), sharing one
    Alpha158 build. Returns (feat_base, cols_base), (feat_a3, cols_a3)."""
    base, base_cols = build_feature_matrix(panel, [], with_baseline=True, baseline=baseline)
    base_long = base[IDX + list(base_cols)]
    scores = [(f"e{i}", score_factor(code, panel, params))
              for i, (name, code, params) in enumerate(elite_specs)]

    # --- V3 baseline: each elite residualized vs Alpha158 only (independent) ---
    parts, cols = [base.set_index(IDX)], list(base_cols)
    for i, (nm, sc) in enumerate(scores):
        r = orthogonalize_vs_features(sc, base_long, list(base_cols)).rename(columns={"value": f"o_{i}"})
        parts.append(r.set_index(IDX))
        cols.append(f"o_{i}")
    feat_base = pd.concat(parts, axis=1).reset_index()

    # --- A3: each elite residualized vs Alpha158 + the prior elites (mutual) ---
    run_long, run_cols = base_long.copy(), list(base_cols)
    parts, cols_a3 = [base.set_index(IDX)], list(base_cols)
    for i, (nm, sc) in enumerate(scores):
        r = orthogonalize_vs_features(sc, run_long, list(run_cols)).rename(columns={"value": f"m_{i}"})
        parts.append(r.set_index(IDX))
        cols_a3.append(f"m_{i}")
        run_long = run_long.merge(r, on=IDX, how="left")
        run_long[f"m_{i}"] = run_long[f"m_{i}"].fillna(0.0)   # neutral where undefined
        run_cols.append(f"m_{i}")
    feat_a3 = pd.concat(parts, axis=1).reset_index()
    return (feat_base, cols), (feat_a3, cols_a3)


def model_preds(panel, feat, cols):
    mr = train_predict(panel, feat, cols, label=LABEL, label_mode=LABEL_MODE)
    return mr.preds, float(mr.test_metrics.headline()["IC"])


def neutralize(preds, size_df):
    """B1: per-date residual of the prediction vs the size proxy."""
    scored = preds.rename(columns={"pred": "value"})[IDX + ["value"]]
    r = orthogonalize_vs_features(scored, size_df, ["size"])
    return r.rename(columns={"value": "pred"})


def bt(preds, panel, bench, **kw):
    m = backtest(preds, panel, split="test", benchmark=bench, **kw).metrics
    return {k: round(float(m[k]), 5) for k in BT_KEYS}


def run_universe(uni, reuse=False):
    panel = pd.read_parquet(config.OUTPUTS / f"{uni}_panel.parquet")
    bench = load_benchmark(uni)
    with open(config.OUTPUTS / "evolution.json") as f:
        evo = json.load(f)
    elite = evo.get("elite") or []
    elite_specs = [(f"fe_elite_{i}", e["code"], e["params"]) for i, e in enumerate(elite)] \
        or [("fe_evolved", evo["best_code"], evo["best_params"])]
    print(f"  {uni}: {len(elite_specs)} elite factors, benchmark={'index' if bench is not None else 'eqw'}")

    size_df = size_proxy(panel)
    pb_path = config.OUTPUTS / f"preds_{uni}_base.parquet"
    pa_path = config.OUTPUTS / f"preds_{uni}_a3.parquet"
    if reuse and pb_path.exists() and pa_path.exists():
        preds_base, ic_base = pd.read_parquet(pb_path), float("nan")
        preds_a3, ic_a3 = pd.read_parquet(pa_path), float("nan")
    else:
        (feat_base, cols_b), (feat_a3, cols_a3) = build_arms(panel, elite_specs)
        preds_base, ic_base = model_preds(panel, feat_base, cols_b)
        preds_a3, ic_a3 = model_preds(panel, feat_a3, cols_a3)
        preds_base.to_parquet(pb_path, index=False)
        preds_a3.to_parquet(pa_path, index=False)

    b1 = neutralize(preds_base, size_df)        # size-neutralized V3 signal
    a3b1 = neutralize(preds_a3, size_df)        # size-neutralized A3 signal (the combo)

    levers = {
        "V3_baseline":     bt(preds_base, panel, bench),
        # individual quick wins
        "B1_size_neutral": bt(b1,         panel, bench),
        "C1_rank_weight":  bt(preds_base, panel, bench, weighting="rank"),
        "C2_hyst_0.5":     bt(preds_base, panel, bench, hysteresis=0.5),
        "A3_mutual_ortho": bt(preds_a3,   panel, bench),
        # winning combinations (the two that helped, stacked)
        "A3_plus_B1":      bt(a3b1,       panel, bench),
        "A3_B1_C2":        bt(a3b1,       panel, bench, hysteresis=0.5),
    }
    return {"benchmark": "index" if bench is not None else "equal_weight",
            "model_ic": {"V3_baseline": round(ic_base, 4), "A3_mutual_ortho": round(ic_a3, 4)},
            "levers": levers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*", default=["csi300", "csi500"])
    ap.add_argument("--reuse", action="store_true",
                    help="reload cached preds_<uni>_{base,a3}.parquet instead of retraining")
    args = ap.parse_args()

    # merge into any existing file so a partial-universe run never clobbers others
    out_path = config.OUTPUTS / "excess_return.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for uni in args.universes:
        print(f"\n== {uni} ==")
        out[uni] = run_universe(uni, reuse=args.reuse)
        lv = out[uni]["levers"]
        base = lv["V3_baseline"]
        print(f"  {'lever':18s} {'AER':>8} {'IR':>7} {'AR':>8} {'SR':>7} {'turnover':>9}  dAER")
        for name, m in lv.items():
            d = m["AER"] - base["AER"]
            print(f"  {name:18s} {m['AER']:+8.4f} {m['IR']:+7.3f} {m['AR']:+8.4f} "
                  f"{m['SR']:+7.3f} {m['ann_turnover']:9.2f}  {d:+.4f}")

    with open(config.OUTPUTS / "excess_return.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'excess_return.json'}")


if __name__ == "__main__":
    main()
