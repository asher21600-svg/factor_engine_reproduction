"""Pure-NumPy reader for Qlib's on-disk .bin data format — NO qlib package needed.

Qlib's `pip install pyqlib` requires a C/Cython toolchain that isn't available
in every environment (e.g. this one).  But the data bundle itself is a simple,
stable layout that we can read directly:

    <provider_uri>/
      calendars/day.txt                 # one trading date per line, sorted
      instruments/<market>.txt          # lines: SYMBOL<TAB>START<TAB>END  (point-in-time)
      features/<symbol_lower>/<field>.day.bin   # float32: [start_cal_index, v0, v1, ...]

Each `<field>.day.bin` stores float32 little-endian: element 0 is the index into
the calendar of the first stored observation; the rest are values for
consecutive calendar dates from that index.

This lets the reproduction consume real Qlib A-share data without installing
qlib.  Returns the same long panel schema as `fe.data.synthetic`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _uri(provider_uri: str | None) -> Path:
    return Path(provider_uri or "~/.qlib/qlib_data/cn_data").expanduser()


def available(provider_uri: str | None = None) -> bool:
    """True if a readable Qlib bin bundle exists at provider_uri."""
    p = _uri(provider_uri)
    return (p / "calendars" / "day.txt").exists() and (p / "features").is_dir()


def read_calendar(provider_uri: str | None = None) -> pd.DatetimeIndex:
    p = _uri(provider_uri) / "calendars" / "day.txt"
    dates = [ln.strip().split()[0] for ln in p.read_text().splitlines() if ln.strip()]
    return pd.to_datetime(pd.Index(dates)).normalize()


def read_instruments(market: str, provider_uri: str | None = None) -> dict:
    """symbol -> list of (start, end) active windows (point-in-time membership)."""
    p = _uri(provider_uri) / "instruments" / f"{market}.txt"
    if not p.exists():
        raise FileNotFoundError(f"no instrument file {p}")
    out: dict[str, list] = {}
    for ln in p.read_text().splitlines():
        parts = ln.strip().split("\t") if "\t" in ln else ln.split()
        if not parts:
            continue
        sym = parts[0]
        start = pd.to_datetime(parts[1]).normalize() if len(parts) > 1 else None
        end = pd.to_datetime(parts[2]).normalize() if len(parts) > 2 else None
        out.setdefault(sym, []).append((start, end))
    return out


def read_feature(symbol: str, field: str, calendar: pd.DatetimeIndex,
                 provider_uri: str | None = None) -> np.ndarray:
    """Return a float array aligned to `calendar` (NaN where no data)."""
    p = _uri(provider_uri) / "features" / symbol.lower() / f"{field}.day.bin"
    out = np.full(len(calendar), np.nan, dtype="float64")
    if not p.exists():
        return out
    arr = np.fromfile(p, dtype="<f4")
    if arr.size < 1:
        return out
    start_i = int(arr[0])
    vals = arr[1:].astype("float64")
    end_i = min(start_i + len(vals), len(calendar))
    out[start_i:end_i] = vals[: end_i - start_i]
    return out


def load_panel_bin(market: str = "csi300",
                   start: str = "2008-01-01", end: str = "2024-12-31",
                   provider_uri: str | None = None,
                   fields=("open", "high", "low", "close", "volume", "vwap"),
                   adjust: bool = True) -> pd.DataFrame:
    """Long panel [datetime, instrument, open, high, low, close, volume] for the
    market's point-in-time constituents over [start, end].

    If a `factor` field is present and adjust=True, prices are multiplied by it
    (forward-adjusted), matching Qlib's adjusted-price convention.
    """
    cal = read_calendar(provider_uri)
    mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
    members = read_instruments(market, provider_uri)

    frames = []
    for sym, windows in members.items():
        cols = {f: read_feature(sym, f, cal, provider_uri) for f in fields}
        if adjust:
            fac = read_feature(sym, "factor", cal, provider_uri)
            if np.isfinite(fac).any():
                for f in ("open", "high", "low", "close", "vwap"):
                    if f in cols:
                        cols[f] = cols[f] * fac
        df = pd.DataFrame(cols)
        df["datetime"] = cal
        df["instrument"] = sym
        # restrict to membership windows
        active = np.zeros(len(cal), dtype=bool)
        for (s, e) in windows:
            w = np.ones(len(cal), dtype=bool)
            if s is not None:
                w &= cal >= s
            if e is not None:
                w &= cal <= e
            active |= w
        df = df[active & mask & np.isfinite(df["close"])]
        if len(df):
            frames.append(df)

    if not frames:
        raise RuntimeError(f"no data read for market {market!r} at {_uri(provider_uri)}")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["instrument", "datetime"]).reset_index(drop=True)
