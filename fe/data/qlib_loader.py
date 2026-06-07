"""Optional real-data loader (Qlib A-share OHLCV).

The paper uses Qlib CSI300/CSI500.  Qlib is not installed here and its CN data
bundle is a large download that is region-gated, so this loader is best-effort:
it tries to import qlib and read the local data dir; on any failure it raises
`QlibUnavailable`, and callers fall back to the synthetic panel (documented in
REPRODUCTION_PLAN.md as the data substitution).
"""
from __future__ import annotations

import pandas as pd


class QlibUnavailable(RuntimeError):
    pass


def load_qlib_panel(market: str = "csi300",
                    start: str = "2008-01-01",
                    end: str = "2024-12-31",
                    provider_uri: str | None = None,
                    winsorize: bool = True) -> pd.DataFrame:
    """Return the same long-format schema as the synthetic panel, from Qlib.

    Columns: datetime, instrument, open, high, low, close, volume,
             ret_1, fwd_ret_{1,3,5,10}, label, split.

    Prefers the pure-NumPy .bin reader (no qlib package required); falls back to
    the qlib package if installed; raises QlibUnavailable if neither works.
    """
    uri = provider_uri or "~/.qlib/qlib_data/cn_data"
    df = None

    # 1) pure-NumPy reader (no qlib install needed)
    from . import qlib_bin
    if qlib_bin.available(uri):
        try:
            df = qlib_bin.load_panel_bin(market=market, start=start, end=end,
                                         provider_uri=uri)
        except Exception as e:  # noqa: BLE001
            df = None
            _bin_err = e

    # 2) fall back to the qlib package
    if df is None:
        try:
            import qlib
            from qlib.data import D
            qlib.init(provider_uri=uri, region="cn")
            instruments = D.instruments(market=market)
            fields = ["$open", "$high", "$low", "$close", "$volume"]
            qdf = D.features(instruments, fields, start_time=start, end_time=end, freq="day")
            if qdf is None or len(qdf) == 0:
                raise RuntimeError("qlib returned no data")
            qdf = qdf.rename(columns={"$open": "open", "$high": "high", "$low": "low",
                                      "$close": "close", "$volume": "volume"})
            df = qdf.reset_index().rename(columns={"instrument": "instrument",
                                                   "datetime": "datetime"})
        except Exception as e:  # noqa: BLE001
            raise QlibUnavailable(
                f"no readable Qlib data at {uri}: bin-reader unavailable and "
                f"qlib package path failed ({e})") from e

    df = df.sort_values(["instrument", "datetime"])

    # forward returns + same-day return
    g = df.groupby("instrument")["close"]
    df["ret_1"] = g.pct_change()
    for h in (1, 3, 5, 10):
        df[f"fwd_ret_{h}"] = g.shift(-h) / df["close"] - 1.0
    # Qlib Alpha158 LABEL0: Ref(close,-2)/Ref(close,-1) - 1 (1-day execution delay)
    df["label"] = g.shift(-2) / g.shift(-1) - 1.0

    # Cross-sectional winsorization of returns/label (the free community bundle
    # has thin-name / adjustment-factor jumps that inflate return std and
    # destabilize IC + the backtest). Clip each date to its [1%, 99%] band.
    ret_cols = ["ret_1", "label"] + [f"fwd_ret_{h}" for h in (1, 3, 5, 10)]
    if winsorize:
        df = _winsorize_xs(df, ret_cols, lo=0.01, hi=0.99)

    # chronological split mirroring the paper
    df["split"] = "train"
    df.loc[df["datetime"] >= "2015-01-01", "split"] = "valid"
    df.loc[df["datetime"] >= "2017-01-01", "split"] = "test"
    return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _winsorize_xs(df: pd.DataFrame, cols, lo=0.01, hi=0.99) -> pd.DataFrame:
    """Clip each column to its per-date [lo, hi] quantile band (cross-sectional)."""
    grp = df.groupby("datetime")
    for c in cols:
        if c not in df.columns:
            continue
        qlo = grp[c].transform(lambda s: s.quantile(lo))
        qhi = grp[c].transform(lambda s: s.quantile(hi))
        df[c] = df[c].clip(lower=qlo, upper=qhi)
    return df


# Index symbol per market, for the excess-return benchmark.
_BENCH_SYMBOL = {"csi300": "SH000300", "csi500": "SH000905"}


def load_benchmark(market: str = "csi300", start: str = "2008-01-01",
                   end: str = "2024-12-31", provider_uri: str | None = None) -> pd.Series:
    """Daily benchmark index return series indexed by datetime (paper-faithful
    excess returns).  Uses the pure-NumPy bin reader, else the qlib package."""
    uri = provider_uri or "~/.qlib/qlib_data/cn_data"
    sym = _BENCH_SYMBOL.get(market)
    if sym is None:
        raise QlibUnavailable(f"no benchmark symbol for market {market!r}")

    from . import qlib_bin
    if qlib_bin.available(uri):
        cal = qlib_bin.read_calendar(uri)
        close = qlib_bin.read_feature(sym, "close", cal, uri)
        s = pd.Series(close, index=cal).dropna()
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        if len(s) > 1:
            return s.pct_change().rename("bench_ret")

    try:
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=uri, region="cn")
        bench = D.features([sym], ["$close"], start_time=start, end_time=end, freq="day")
        s = bench["$close"].droplevel("instrument") if hasattr(bench["$close"], "droplevel") else bench["$close"]
        return s.sort_index().pct_change().rename("bench_ret")
    except Exception as e:  # noqa: BLE001
        raise QlibUnavailable(f"benchmark {sym} unavailable at {uri}: {e}") from e
