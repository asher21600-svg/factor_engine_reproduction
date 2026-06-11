#!/usr/bin/env python
"""Make the augmented model's CUMULATIVE EXCESS RETURN beat the Alpha158 baseline
AND the index — addressing "augmented ≈ baseline, and both trail the index".

Two problems, two levers:
  (1) augmented ≈ baseline  — the evolved OHLCV factors are largely SPANNED by
      Alpha158 (128 price/volume technicals), so LightGBM gives them ~0 importance.
      Fix = RESIDUAL STACKING: train the baseline first, then train a small FE model
      ONLY on what the baseline missed (target − baseline_pred). The FE factors can
      then only ADD orthogonal signal, never dilute the baseline.
  (2) both trail the index (excess < 1.0) — the 5-day book turns over ~75-84×/yr and
      cost (~9-10%) erases the gross edge (Result 9). Fix = the optimal holding period.

For each universe we backtest baseline / augmented(LGBM-merge) / stack at BOTH the
paper's 5-day hold and the Result-9 optimal hold, and report the final cumulative
excess (portfolio / index) and AER. Equity curves are saved for the report.

    python scripts/12_beat_baseline.py [--universes csi300 csi500]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fe import config
from fe.integration import build_feature_matrix, train_predict, backtest

IDX = ["datetime", "instrument"]
LABEL, LABEL_MODE = "label", "date_demeaned"
OPT_HOLD = {"csi300": 40, "csi500": 25}        # Result-9 optima (band on)
OPT_BAND = 1.0


def load_benchmark(uni):
    try:
        from fe.data.qlib_loader import load_benchmark as _lb
        return _lb(uni)
    except Exception as e:  # noqa: BLE001
        print(f"  [benchmark unavailable: {e}] -> equal-weight market")
        return None


def residual_stack(panel, base_pred, fe_feat, fe_cols, seed=0):
    """Train a small FE model on the baseline's residual (date-demeaned label −
    baseline_pred); return combined = baseline_pred + fe_residual_pred."""
    import lightgbm as lgb
    d = fe_feat.merge(panel[["datetime", "instrument", LABEL, "split"]], on=IDX, how="inner")
    d = d.merge(base_pred[IDX + ["pred"]].rename(columns={"pred": "base_pred"}), on=IDX, how="inner")
    d = d.replace([np.inf, -np.inf], np.nan)
    dem = d[LABEL] - d.groupby("datetime")[LABEL].transform("mean")
    d["tgt"] = dem - d["base_pred"]                      # what the baseline left on the table
    tr = d[(d["split"] == "train") & d["tgt"].notna()].dropna(subset=fe_cols)
    va = d[(d["split"] == "valid") & d["tgt"].notna()].dropna(subset=fe_cols)
    if len(tr) < 100:
        return base_pred
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          min_child_samples=80, reg_lambda=2.0, random_state=seed,
                          n_jobs=-1, verbosity=-1)
    m.fit(tr[fe_cols], tr["tgt"], eval_set=[(va[fe_cols], va["tgt"])] if len(va) else None,
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)] if len(va) else None)
    pr = d.dropna(subset=fe_cols).copy()
    pr["pred"] = pr["base_pred"] + m.predict(pr[fe_cols])
    return pr[IDX + ["pred", "split"]]


def run_universe(uni):
    panel = pd.read_parquet(config.OUTPUTS / f"{uni}_panel.parquet")
    bench = load_benchmark(uni)
    evo = json.load(open(config.OUTPUTS / "evolution.json"))
    elite = evo.get("elite") or []
    elite_specs = [(f"fe_{i}", e["code"], e["params"]) for i, e in enumerate(elite)] \
        or [("fe", evo["best_code"], evo["best_params"])]

    # baseline (Alpha158 only) and augmented (Alpha158 + per-date-orthogonalized FE)
    base_feat, base_cols = build_feature_matrix(panel, [], with_baseline=True, baseline="alpha158")
    aug_feat, aug_cols = build_feature_matrix(panel, elite_specs, with_baseline=True,
                                              baseline="alpha158", orthogonalize_factors=True)
    base_pred = train_predict(panel, base_feat, base_cols, label=LABEL, label_mode=LABEL_MODE).preds
    aug_pred = train_predict(panel, aug_feat, aug_cols, label=LABEL, label_mode=LABEL_MODE).preds

    # residual stack uses only the orthogonalized FE columns
    fe_cols = [c for c in aug_cols if c.startswith("o_")]
    stack_pred = residual_stack(panel, base_pred, aug_feat[IDX + fe_cols], fe_cols)

    arms = {"baseline": base_pred, "augmented": aug_pred, "stack": stack_pred}
    out = {"opt_hold": OPT_HOLD[uni], "holds": {}}
    for tag, hold, band in (("h5", 5, 0.0), ("opt", OPT_HOLD[uni], OPT_BAND)):
        row = {}
        for arm, pred in arms.items():
            bt = backtest(pred, panel, split="test", benchmark=bench, holding=hold, hysteresis=band)
            m = bt.metrics
            row[arm] = {"AER": round(m["AER"], 5), "IR": round(m["IR"], 4), "SR": round(m["SR"], 4),
                        "turnover": round(m["ann_turnover"], 2),
                        "cum_excess": round(float(bt.equity["excess"].iloc[-1]), 4),
                        "beats_index": bool(bt.equity["excess"].iloc[-1] > 1.0)}
            bt.equity.to_csv(config.OUTPUTS / f"equity_{uni}_{arm}_{tag}.csv", index=False)
        out["holds"][tag] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*", default=["csi300", "csi500"])
    args = ap.parse_args()
    res = {}
    for uni in args.universes:
        print(f"\n== {uni} (optimal hold {OPT_HOLD[uni]}d) ==")
        res[uni] = run_universe(uni)
        for tag, label in (("h5", "5-day hold (paper)"), ("opt", f"{OPT_HOLD[uni]}-day hold (Result 9)")):
            row = res[uni]["holds"][tag]
            print(f"  --- {label} ---")
            print(f"  {'arm':10} {'cum_excess':>11} {'beats_idx':>10} {'AER':>8} {'IR':>7} {'turnover':>9}")
            for arm in ("baseline", "augmented", "stack"):
                m = row[arm]
                star = "  <== beats index" if m["beats_index"] else ""
                print(f"  {arm:10} {m['cum_excess']:11.3f} {str(m['beats_index']):>10} {m['AER']:+8.4f} "
                      f"{m['IR']:+7.3f} {m['turnover']:9.1f}{star}")
    with open(config.OUTPUTS / "beat_baseline.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'beat_baseline.json'}")


if __name__ == "__main__":
    main()
