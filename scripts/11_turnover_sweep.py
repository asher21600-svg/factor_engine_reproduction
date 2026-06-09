#!/usr/bin/env python
"""Tier-1 turnover sweep — make net excess return positive by trading the EXISTING
signal more cheaply (docs/EXCESS_RETURN_PLAN.md "Next phase").

The v4 augmented model earns +8-10% GROSS excess return but ~9-10% annualized
transaction cost (turnover ~75-84x) eats it to ~0 net. This sweep attacks turnover
at the source, on the cached v4 predictions (backtest-only — no model rebuild):

  smoothing : EWM the daily prediction per stock (span 1=none, 5, 10) -> stable
              cross-sectional ranking -> the held set churns less.
  holding   : 5 (paper) / 10 / 20-day holds (backtest renormalizes per holding).
  band      : hysteresis sell-band (0 = strict top-k; 1.0 = keep a held name until it
              exits top-2k) -> stops marginal names from round-tripping.

Reports net AER, IR, SR, turnover, ann cost and the gross-AER proxy (net + cost) for
each config, on the test split vs the real index, per universe and per arm
(base = standard augmented model; a3 = mutually-orthogonalized elites).

    python scripts/11_turnover_sweep.py [--universes csi300 csi500]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from fe import config
from fe.integration import backtest

IDX = ["datetime", "instrument"]
SPANS = [1, 5, 10]
HOLDS = [5, 10, 20]
BANDS = [0.0, 1.0]


def load_benchmark(uni):
    try:
        from fe.data.qlib_loader import load_benchmark as _lb
        return _lb(uni)
    except Exception as e:  # noqa: BLE001
        print(f"  [benchmark unavailable: {e}] -> equal-weight market")
        return None


def smooth(preds, span):
    if span <= 1:
        return preds
    p = preds.sort_values(["instrument", "datetime"]).copy()
    p["pred"] = p.groupby("instrument")["pred"].transform(lambda s: s.ewm(span=span, min_periods=1).mean())
    return p


def run_universe(uni):
    panel = pd.read_parquet(config.OUTPUTS / f"{uni}_panel.parquet")
    bench = load_benchmark(uni)
    grid = {}
    for arm in ("base", "a3"):
        path = config.OUTPUTS / f"preds_{uni}_{arm}.parquet"
        if not path.exists():
            print(f"  [no cached preds_{uni}_{arm} — run scripts/09 first]")
            continue
        preds0 = pd.read_parquet(path)
        for span in SPANS:
            ps = smooth(preds0, span)[IDX + ["pred"]]
            for hold in HOLDS:
                for band in BANDS:
                    m = backtest(ps, panel, holding=hold, split="test",
                                 benchmark=bench, hysteresis=band).metrics
                    grid[f"{arm}|s{span}|h{hold}|b{band:g}"] = {
                        "AER": round(m["AER"], 5), "IR": round(m["IR"], 4),
                        "SR": round(m["SR"], 4), "turnover": round(m["ann_turnover"], 2),
                        "ann_cost": round(m["ann_cost"], 5),
                        "gross_AER": round(m["AER"] + m["ann_cost"], 5)}
    return {"benchmark": "index" if bench is not None else "equal_weight", "grid": grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*", default=["csi300", "csi500"])
    args = ap.parse_args()

    out = {}
    for uni in args.universes:
        print(f"\n== {uni} ==")
        out[uni] = run_universe(uni)
        grid = out[uni]["grid"]
        if not grid:
            continue
        ref = grid.get("base|s1|h5|b0", {})            # current default (= V3_baseline arm)
        r_aer = ref.get("AER", 0.0)
        order = sorted(grid, key=lambda k: grid[k]["AER"], reverse=True)
        if ref:
            print(f"  default base|s1|h5|b0: AER {r_aer:+.4f}  turnover {ref['turnover']:.1f}  "
                  f"cost {ref['ann_cost']:.4f}  gross {ref['gross_AER']:+.4f}")
        print(f"  {'config':18} {'AER':>8} {'IR':>7} {'SR':>7} {'turnover':>9} {'cost':>8} {'gross':>8}  dAER")
        for k in order[:12]:                            # top 12 by net AER
            m = grid[k]
            flag = "  <== POSITIVE" if m["AER"] > 0 else ""
            print(f"  {k:18} {m['AER']:+8.4f} {m['IR']:+7.3f} {m['SR']:+7.3f} {m['turnover']:9.1f} "
                  f"{m['ann_cost']:8.4f} {m['gross_AER']:+8.4f}  {m['AER']-r_aer:+.4f}{flag}")
        best = order[0]
        print(f"  BEST: {best}  AER {grid[best]['AER']:+.4f} (default {r_aer:+.4f}, "
              f"Δ{grid[best]['AER']-r_aer:+.4f})")

    with open(config.OUTPUTS / "turnover_sweep.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'turnover_sweep.json'}")


if __name__ == "__main__":
    main()
