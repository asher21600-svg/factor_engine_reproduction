#!/usr/bin/env python
"""Verify the real Qlib CN bundle loads — pure-NumPy reader, NO qlib package.

    python tests/test_data.py                 # checks ~/.qlib/qlib_data/cn_data
    python tests/test_data.py --uri /data/cn_data

Exits 0 if all checks pass, 1 otherwise. Runnable as a plain script or via pytest.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

URI = os.environ.get("QLIB_URI", os.path.expanduser("~/.qlib/qlib_data/cn_data"))


def _check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def run(uri=URI) -> bool:
    from fe.data import qlib_bin
    from fe.data.qlib_loader import load_qlib_panel, load_benchmark
    from fe.factors import SEED_SRC, score_factor
    from fe.eval import evaluate_factor

    print(f"== verifying Qlib bundle at {uri} ==")
    ok = True

    avail = qlib_bin.available(uri)
    ok &= _check("bundle present (calendars + features)", avail)
    if not avail:
        print("  -> run: python data/setup_qlib.py")
        return False

    cal = qlib_bin.read_calendar(uri)
    ok &= _check("calendar loads", len(cal) > 1000,
                 f"{len(cal)} days, {cal.min().date()}..{cal.max().date()}")
    ok &= _check("covers paper test window (>=2017..2024)",
                 cal.min() <= pd.Timestamp("2008-06-30") and cal.max() >= pd.Timestamp("2024-01-01"),
                 f"{cal.min().date()}..{cal.max().date()}")

    for mkt in ("csi300", "csi500"):
        try:
            members = qlib_bin.read_instruments(mkt, uri)
            ok &= _check(f"{mkt} instruments", len(members) >= 100, f"{len(members)} symbols")
        except Exception as e:  # noqa: BLE001
            ok &= _check(f"{mkt} instruments", False, str(e))

    # load a real panel slice and sanity-check it
    try:
        panel = load_qlib_panel(market="csi300", start="2017-01-01", end="2018-12-31",
                                provider_uri=uri)
        need = ["open", "high", "low", "close", "volume", "ret_1",
                "fwd_ret_5", "label", "split"]
        ok &= _check("panel columns", all(c in panel.columns for c in need),
                     f"{panel.shape[0]} rows, {panel.instrument.nunique()} stocks")
        ok &= _check("prices positive & finite",
                     bool((panel["close"] > 0).all() and np.isfinite(panel["close"]).all()))
        med = panel["close"].median()
        ok &= _check("close median in plausible A-share range", 1 < med < 1000, f"median={med:.1f}")
    except Exception as e:  # noqa: BLE001
        ok &= _check("load_qlib_panel(csi300)", False, str(e))
        panel = None

    # benchmark index
    try:
        bench = load_benchmark("csi300", "2017-01-01", "2018-12-31", provider_uri=uri)
        ok &= _check("benchmark SH000300 returns", bench.notna().sum() > 100,
                     f"{bench.notna().sum()} daily returns")
    except Exception as e:  # noqa: BLE001
        ok &= _check("benchmark SH000300", False, str(e))

    # smoke: the seed factor runs and yields a finite IC on real data
    if panel is not None:
        try:
            m = evaluate_factor(score_factor(SEED_SRC, panel, {}), panel, primary_lag=5, split="test")
            ic = m.headline()["IC"]
            ok &= _check("seed factor computes on real data", np.isfinite(ic), f"test IC={ic:+.4f}")
        except Exception as e:  # noqa: BLE001
            ok &= _check("seed factor on real data", False, str(e))

    print("\n" + ("ALL CHECKS PASSED — real data is ready." if ok else "SOME CHECKS FAILED."))
    return ok


def test_data():  # pytest entry
    assert run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=URI)
    a = ap.parse_args()
    sys.exit(0 if run(a.uri) else 1)
