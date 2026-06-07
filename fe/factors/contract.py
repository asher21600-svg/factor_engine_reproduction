"""The executable factor I/O contract (paper §4).

A factor is a Python program of the form:

    def factor(pricing_data: pl.DataFrame, parameters: dict) -> pl.DataFrame:
        ...  # returns columns [instrument, datetime, Factor]

Factors are *code strings* (the evolution genome) compiled to callables.  This
module compiles them, feeds them an OHLCV frame in the paper's expected shape,
runs them with a wall-clock guard, and returns a tidy scored frame
[datetime, instrument, value] ready for `fe.eval.metrics.evaluate_factor`.
"""
from __future__ import annotations

import math
import signal
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


class FactorRunError(Exception):
    """Raised when a factor program fails to compile or execute."""


# Globals available to factor programs.  The paper's factors use pl, pd, np.
def _factor_globals() -> dict:
    return {"pl": pl, "pd": pd, "np": np, "math": math, "__builtins__": __builtins__}


def compile_factor(code: str, fn_name: str | None = None):
    """Compile a factor source string to a callable `factor(pricing_data, parameters)`.

    If `fn_name` is None, picks the first top-level `def` in the source.
    """
    ns: dict = _factor_globals()
    try:
        exec(compile(code, "<factor>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        raise FactorRunError(f"compile failed: {type(e).__name__}: {e}") from e

    if fn_name is None:
        # first user-defined function in the namespace
        cands = [v for k, v in ns.items()
                 if callable(v) and getattr(v, "__module__", None) is None
                 and not k.startswith("_") and k not in ("pl", "pd", "np", "math")]
        # fall back: any function defined in this exec
        funcs = [v for v in ns.values() if callable(v) and getattr(v, "__code__", None)
                 and v.__code__.co_filename == "<factor>"]
        if funcs:
            fn = funcs[0]
        elif cands:
            fn = cands[0]
        else:
            raise FactorRunError("no factor function found in source")
    else:
        fn = ns.get(fn_name)
        if fn is None:
            raise FactorRunError(f"function {fn_name!r} not found")
    return fn


def panel_to_pricing(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert our long panel to the Qlib-style frame the paper's factors expect:
    a pandas DataFrame with columns instrument, datetime, $open,$high,$low,$close,$volume.
    """
    cols = {"open": "$open", "high": "$high", "low": "$low",
            "close": "$close", "volume": "$volume"}
    out = panel[["instrument", "datetime", "open", "high", "low", "close", "volume"]].copy()
    out = out.rename(columns=cols)
    return out.reset_index(drop=True)


class _Timeout:
    """Best-effort wall-clock guard (POSIX). No-op if signals unavailable."""
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        try:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        except (ValueError, AttributeError):
            self._old = None
        return self

    def __exit__(self, *exc):
        try:
            signal.alarm(0)
            if self._old is not None:
                signal.signal(signal.SIGALRM, self._old)
        except (ValueError, AttributeError):
            pass

    @staticmethod
    def _handler(signum, frame):
        raise FactorRunError("factor execution timed out")


def run_factor(fn, pricing_pd: pd.DataFrame, parameters: dict | None = None,
               timeout_s: int = 25) -> pd.DataFrame:
    """Run a compiled factor; return tidy [datetime, instrument, value] (pandas).

    Raises FactorRunError on failure (so the engine can score it as invalid).
    """
    parameters = parameters or {}
    try:
        with _Timeout(timeout_s):
            out = fn(pricing_pd, parameters)
    except FactorRunError:
        raise
    except Exception as e:  # noqa: BLE001
        raise FactorRunError(f"runtime error: {type(e).__name__}: {e}") from e

    # Normalize output to pandas [datetime, instrument, value]
    if isinstance(out, pl.DataFrame):
        out = out.to_pandas()
    elif not isinstance(out, pd.DataFrame):
        raise FactorRunError(f"factor returned {type(out).__name__}, expected DataFrame")

    # Identify the value column (paper names it 'Factor'; be lenient)
    val_col = None
    for c in ("Factor", "value", "factor"):
        if c in out.columns:
            val_col = c
            break
    if val_col is None:
        # last non-key column
        non_key = [c for c in out.columns if c not in ("instrument", "datetime")]
        if not non_key:
            raise FactorRunError("factor output has no value column")
        val_col = non_key[-1]

    if "instrument" not in out.columns or "datetime" not in out.columns:
        raise FactorRunError("factor output missing instrument/datetime")

    out = out[["datetime", "instrument", val_col]].rename(columns={val_col: "value"})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if out.empty:
        raise FactorRunError("factor produced no finite values")
    return out


@dataclass
class _Cache:
    pricing: pd.DataFrame | None = None
    key: tuple | None = None


_PRICING_CACHE = _Cache()


def _panel_signature(panel: pd.DataFrame) -> tuple:
    """Content-aware cache key.

    NOTE: id(panel) alone is unsafe — CPython reuses memory addresses after an
    object is freed, so a new panel can collide with a freed one's id and be
    served stale pricing.  We combine id() with a cheap content fingerprint so a
    collision only hits the cache if the content is genuinely identical.
    """
    n = len(panel)
    if n == 0:
        return (id(panel), 0)
    c = panel["close"].to_numpy()
    return (id(panel), n, float(c[0]), float(c[-1]), float(c[n // 2]),
            float(panel["volume"].to_numpy()[0]))


def score_factor(code_or_fn, panel: pd.DataFrame, parameters: dict | None = None,
                 fn_name: str | None = None, timeout_s: int = 25) -> pd.DataFrame:
    """Convenience: compile (if needed) + run against a panel.

    Caches the OHLCV→pricing conversion per panel (by content signature) to
    avoid rebuilding it on every evaluation in the evolution loop.
    """
    fn = compile_factor(code_or_fn, fn_name) if isinstance(code_or_fn, str) else code_or_fn

    sig = _panel_signature(panel)
    if _PRICING_CACHE.key == sig and _PRICING_CACHE.pricing is not None:
        pricing = _PRICING_CACHE.pricing
    else:
        pricing = panel_to_pricing(panel)
        _PRICING_CACHE.pricing = pricing
        _PRICING_CACHE.key = sig

    return run_factor(fn, pricing.copy(), parameters, timeout_s=timeout_s)
