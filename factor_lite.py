#!/usr/bin/env python
"""factor_lite.py — a clean, dependency-light FactorEngine evaluator.

Reads the local Qlib ``.bin`` data DIRECTLY (NumPy) and scores factors by
IC / RankIC / ICIR — with **no** external quant packages: no ``qlib``, no
LightGBM / Optuna / GPLearn, and no backtest framework. Just ``numpy`` + ``pandas``.

Why IC instead of a backtest? The full reproduction found that a top-50, 5-day
tranche backtest mostly measures *turnover and A-share cost*, not signal: the
augmented model earns +8-10% GROSS excess but ~9-10% cost erases it, and a longer
hold flips it positive without changing the factor at all. So the honest, portable
headline for a factor is its out-of-sample **IC** (and a cheap top-decile long-short
return as a return-flavored sanity check). That is all this file computes.

Usage:
    python factor_lite.py                          # csi300, ~/.qlib/qlib_data/cn_data
    python factor_lite.py --market csi500 --split test
    python factor_lite.py --provider /path/to/cn_data --start 2008-01-01

Plug in your own factor: write ``panel["f_myidea"] = ...`` in add_factors() (any
per-instrument / per-date pandas expression) and it is scored automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Read the Qlib .bin bundle directly (no qlib package).
#    Layout:  calendars/day.txt · instruments/<market>.txt · features/<sym>/<field>.day.bin
#    Each <field>.day.bin is float32 LE: element 0 = start index into the calendar.
# ----------------------------------------------------------------------------
def _uri(provider: str | None) -> Path:
    return Path(provider or "~/.qlib/qlib_data/cn_data").expanduser()


def read_calendar(provider: str | None) -> pd.DatetimeIndex:
    lines = (_uri(provider) / "calendars" / "day.txt").read_text().splitlines()
    return pd.DatetimeIndex(pd.to_datetime([ln.split()[0] for ln in lines if ln.strip()])).normalize()


def read_instruments(market: str, provider: str | None) -> dict:
    """symbol -> [(start, end), ...] point-in-time membership windows."""
    out: dict[str, list] = {}
    for ln in (_uri(provider) / "instruments" / f"{market}.txt").read_text().splitlines():
        p = ln.split("\t") if "\t" in ln else ln.split()
        if not p:
            continue
        out.setdefault(p[0], []).append((pd.Timestamp(p[1]) if len(p) > 1 else None,
                                         pd.Timestamp(p[2]) if len(p) > 2 else None))
    return out


def read_feature(sym: str, field: str, n: int, provider: str | None) -> np.ndarray:
    f = _uri(provider) / "features" / sym.lower() / f"{field}.day.bin"
    out = np.full(n, np.nan)
    if not f.exists():
        return out
    a = np.fromfile(f, dtype="<f4")
    if a.size < 1:
        return out
    start = int(a[0])
    vals = a[1:].astype(float)
    end = min(start + len(vals), n)
    out[start:end] = vals[: end - start]
    return out


def load_panel(market="csi300", start="2008-01-01", end="2024-12-31", provider=None,
               fields=("open", "high", "low", "close", "volume", "vwap"), adjust=True) -> pd.DataFrame:
    """Long panel [datetime, instrument, OHLCV, vwap] for point-in-time constituents."""
    cal = read_calendar(provider)
    mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
    frames = []
    for sym, windows in read_instruments(market, provider).items():
        cols = {f: read_feature(sym, f, len(cal), provider) for f in fields}
        if adjust:                                       # Qlib forward-adjusted prices
            fac = read_feature(sym, "factor", len(cal), provider)
            if np.isfinite(fac).any():
                for f in ("open", "high", "low", "close", "vwap"):
                    if f in cols:
                        cols[f] = cols[f] * fac
        df = pd.DataFrame(cols)
        df["datetime"], df["instrument"] = cal, sym
        active = np.zeros(len(cal), bool)
        for s, e in windows:
            w = np.ones(len(cal), bool)
            if s is not None:
                w &= cal >= s
            if e is not None:
                w &= cal <= e
            active |= w
        df = df[active & mask & np.isfinite(df["close"])]
        if len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no data for {market!r} at {_uri(provider)} — is the .bin bundle there?")
    return pd.concat(frames, ignore_index=True).sort_values(["instrument", "datetime"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Forward returns + a fixed train/valid/test split (no look-ahead).
# ----------------------------------------------------------------------------
def add_returns(panel: pd.DataFrame, horizons=(1, 5, 10), clip=0.2) -> pd.DataFrame:
    gi = panel.groupby("instrument", group_keys=False)
    panel["ret1"] = gi["close"].transform(lambda s: s.pct_change()).clip(-clip, clip)
    for h in horizons:                                   # label at date t uses only future prices
        panel[f"fwd{h}"] = gi["close"].transform(lambda s: s.shift(-h) / s - 1.0).clip(-clip * h, clip * h)
    yr = panel["datetime"].dt.year
    panel["split"] = np.where(yr <= 2014, "train", np.where(yr <= 2016, "valid", "test"))
    return panel


# ----------------------------------------------------------------------------
# 3. Metrics — the paper's convention: IC = cross-sectional Pearson, RankIC = Spearman.
# ----------------------------------------------------------------------------
def _daily_corr(df: pd.DataFrame, fcol: str, rcol: str, method: str) -> pd.Series:
    d = df[["datetime", fcol, rcol]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.Series(dtype=float)
    if method == "spearman":                             # rank within each date, then Pearson
        d = d.assign(**{fcol: d.groupby("datetime")[fcol].rank(),
                        rcol: d.groupby("datetime")[rcol].rank()})
    return d.groupby("datetime")[[fcol, rcol]].apply(lambda x: x[fcol].corr(x[rcol])).dropna()


def evaluate(panel: pd.DataFrame, fcol: str, rcol="fwd5", split="test") -> dict:
    """IC / ICIR / RankIC / RankICIR for one factor over one split (Qlib convention)."""
    sub = panel[panel["split"] == split]
    ic = _daily_corr(sub, fcol, rcol, "pearson")
    ric = _daily_corr(sub, fcol, rcol, "spearman")
    sd = lambda s: s.std(ddof=1)
    return {"IC": ic.mean(), "ICIR": ic.mean() / sd(ic) if sd(ic) else np.nan,
            "RankIC": ric.mean(), "RankICIR": ric.mean() / sd(ric) if sd(ric) else np.nan,
            "n_days": int(len(ic))}


def long_short(panel: pd.DataFrame, fcol: str, rcol="fwd5", split="test", q=0.1) -> float:
    """A cheap return proxy (NO backtest framework): mean daily top-q minus bottom-q
    forward return, scaled to annual. It ignores cost/turnover by design — see Result 9."""
    sub = panel[panel["split"] == split][["datetime", fcol, rcol]].replace([np.inf, -np.inf], np.nan).dropna()

    def spread(x):
        hi, lo = x[fcol].quantile(1 - q), x[fcol].quantile(q)
        return x.loc[x[fcol] >= hi, rcol].mean() - x.loc[x[fcol] <= lo, rcol].mean()

    s = sub.groupby("datetime")[[fcol, rcol]].apply(spread).dropna()
    return float(s.mean() * 252 / 5) if len(s) else float("nan")     # fwd5 → ~annual


# ----------------------------------------------------------------------------
# 4. Example factors (plain functions of the panel). Add your own here.
#    These are the families the full engine kept rediscovering: reversal, low-vol,
#    Amihud illiquidity — the ones with signal beyond plain momentum.
# ----------------------------------------------------------------------------
def add_factors(panel: pd.DataFrame) -> list[str]:
    gi = panel.groupby("instrument", group_keys=False)
    panel["amount"] = panel["close"] * panel["volume"]
    panel["illiq"] = panel["ret1"].abs() / panel["amount"].replace(0, np.nan)
    panel["hl"] = (panel["high"] - panel["low"]) / panel["close"]

    panel["seed_momentum"] = gi["close"].transform(lambda s: s / s.shift(20) - 1.0)        # weak baseline
    panel["f_reversal"] = -gi["ret1"].transform(lambda s: s.rolling(5).sum())              # 5-day reversal
    panel["f_lowvol"] = -gi["ret1"].transform(lambda s: s.rolling(20).std())               # low realized vol
    panel["f_amihud"] = -gi["illiq"].transform(lambda s: s.rolling(20).mean())             # illiquidity reversal
    panel["f_range_rev"] = -gi["hl"].transform(lambda s: s.rolling(10).mean())             # high-range → reversal
    # a tiny "combo": equal-weight z-scored blend of the three good families
    def _z(col):
        d = panel.groupby("datetime")[col]
        return (panel[col] - d.transform("mean")) / d.transform("std").replace(0, np.nan)
    panel["f_combo"] = (_z("f_reversal") + _z("f_lowvol") + _z("f_amihud")) / 3.0
    return ["seed_momentum", "f_reversal", "f_lowvol", "f_amihud", "f_range_rev", "f_combo"]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Dependency-light factor IC evaluator (numpy+pandas only).")
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--provider", default=None, help="Qlib .bin dir (default ~/.qlib/qlib_data/cn_data)")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--ret", default="fwd5", help="forward-return horizon column (fwd1/fwd5/fwd10)")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = ap.parse_args()

    print(f"loading {args.market} from {_uri(args.provider)} ...")
    panel = add_returns(load_panel(args.market, args.start, args.end, args.provider))
    factors = add_factors(panel)
    print(f"{panel['instrument'].nunique()} names · {panel['datetime'].nunique()} dates · "
          f"{len(panel):,} rows · scoring on {args.split} (label {args.ret})\n")

    print(f"  {'factor':16} {'IC':>9} {'ICIR':>8} {'RankIC':>9} {'RankICIR':>9} {'LS ann%':>9}")
    rows = []
    for f in factors:
        m = evaluate(panel, f, rcol=args.ret, split=args.split)
        ls = long_short(panel, f, rcol=args.ret, split=args.split)
        rows.append((f, m, ls))
        print(f"  {f:16} {m['IC']:+9.4f} {m['ICIR']:+8.3f} {m['RankIC']:+9.4f} "
              f"{m['RankICIR']:+9.3f} {100*ls:+9.2f}")
    best = max(rows, key=lambda r: r[1]["RankIC"])
    print(f"\nbest by RankIC: {best[0]} (IC {best[1]['IC']:+.4f}, RankIC {best[1]['RankIC']:+.4f})")


if __name__ == "__main__":
    main()
