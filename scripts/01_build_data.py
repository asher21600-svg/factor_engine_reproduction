#!/usr/bin/env python
"""Phase 2 entry point: build (or load) the OHLCV panels and save to outputs/.

Tries real Qlib first (per `--market`); falls back to the synthetic panel.
Saves outputs/<profile>_panel.parquet and prints sanity checks, including the
seed/evolved factor IC so we can confirm the data is calibrated.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from fe import config
from fe.data import build_synthetic_panel, GenParams
from fe.data.qlib_loader import load_qlib_panel, QlibUnavailable
from fe.factors import SEED_SRC, EVOLVED_SRC, score_factor
from fe.eval import evaluate_factor


def sanity_check(panel: pd.DataFrame, name: str):
    print(f"\n[{name}] shape={panel.shape}  dates={panel.datetime.nunique()}  "
          f"stocks={panel.instrument.nunique()}")
    print("  split day-counts:",
          panel.groupby("split")["datetime"].nunique().to_dict())
    # price/volume sanity
    px = panel["close"]
    print(f"  close median={px.median():.1f}  fwd_ret_5 mean={panel['fwd_ret_5'].mean():+.4f} "
          f"std={panel['fwd_ret_5'].std():.4f}")
    for fname, src in [("seed", SEED_SRC), ("evolved", EVOLVED_SRC)]:
        try:
            m = evaluate_factor(score_factor(src, panel), panel, primary_lag=5, split="test")
            h = m.headline()
            print(f"  {fname:8s} (TEST): IC={h['IC']:+.4f} ICIR={h['ICIR']:+.2f} "
                  f"RIC={h['RIC']:+.4f} fit={m.fitness:+.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {fname}: ERROR {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*", default=["fast", "csi300", "csi500"])
    ap.add_argument("--market", default=None,
                    help="if set, try real Qlib for this market (csi300/csi500)")
    ap.add_argument("--qlib-uri", default=None,
                    help="path to the Qlib .bin bundle (default ~/.qlib/qlib_data/cn_data); "
                         "read directly with the pure-NumPy reader, no qlib package needed")
    args = ap.parse_args()

    for prof_name in args.profiles:
        prof = config.PROFILES[prof_name]
        panel = None
        if args.market and prof_name in ("csi300", "csi500"):
            try:
                panel = load_qlib_panel(market=prof_name, provider_uri=args.qlib_uri)
                print(f"[{prof_name}] loaded REAL Qlib data"
                      f"{' from ' + args.qlib_uri if args.qlib_uri else ''}")
            except QlibUnavailable as e:
                print(f"[{prof_name}] Qlib unavailable ({e}); using synthetic")
        if panel is None:
            panel = build_synthetic_panel(
                n_stocks=prof.n_stocks, n_days=prof.n_days, seed=prof.seed,
                params=GenParams())
        out = config.OUTPUTS / f"{prof_name}_panel.parquet"
        panel.to_parquet(out, index=False)
        sanity_check(panel, prof_name)
        print(f"  -> saved {out}")


if __name__ == "__main__":
    main()
