"""Offline LLM-reasoned novel factor proposals.

These programs are a deterministic fallback library for runs without a reachable
live macro-agent.  Each is a distinct, economically motivated alpha hypothesis
for short-horizon Chinese A-share prediction, written to the same Polars I/O
contract and injected into the evolution as macro "proposals" (fresh-idea
mutations) that the Bayesian micro-search then tunes.  When Kimi/Moonshot is
configured, the live agent in ``fe.evolution.macro`` can add fresh
dataset-conditioned edits on top of this library.

Reasoning per factor is in its docstring — this is the LLM "###IDEA" step.
"""
from __future__ import annotations

# Shared input boilerplate (matches the seed's I/O handling).
_PRELUDE = """
    eps = parameters.get("epsilon", 1e-9)
    if isinstance(pricing_data, pd.DataFrame):
        df = pl.from_pandas(pricing_data.reset_index()).rename({
            '$close': 'close', '$open': 'open', '$high': 'high',
            '$low': 'low', '$volume': 'volume'})
    else:
        df = pricing_data.rename({
            '$close': 'close', '$open': 'open', '$high': 'high',
            '$low': 'low', '$volume': 'volume'})
    df = df.select(['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume']).with_columns([
        pl.col('datetime').cast(pl.Date),
        pl.col(['open', 'high', 'low', 'close', 'volume']).cast(pl.Float64),
    ]).sort(['instrument', 'datetime'])
"""

_EPILOGUE = """
    df = df.with_columns(
        ((pl.col('raw') - pl.col('raw').mean().over('datetime')) /
         (pl.col('raw').std(ddof=0).over('datetime') + eps)).alias('Factor'))
    out = df.select(['instrument', 'datetime', 'Factor']).filter(
        pl.col('Factor').is_not_nan() & pl.col('Factor').is_finite())
    return out
"""


# IDEA: Amihud-illiquidity-amplified short-term reversal. A-share short-horizon
# returns mean-revert, and the reversal is strongest in illiquid names (high
# |ret|/turnover). Signal = (-k-day return) weighted by the cross-sectional rank
# of trailing Amihud illiquidity.
AMIHUD_REVERSAL = f'''
def factor(pricing_data, parameters):{_PRELUDE}
    k = parameters.get("k", 5)
    df = df.with_columns([
        (pl.col('close') / pl.col('close').shift(1).over('instrument') - 1.0).alias('ret1'),
        (pl.col('volume') * pl.col('close')).alias('turnover'),
    ])
    df = df.with_columns(
        (pl.col('ret1').abs() / (pl.col('turnover') + eps)).rolling_mean(window_size=k, min_periods=2).over('instrument').alias('illiq'))
    df = df.with_columns([
        (pl.col('close') / pl.col('close').shift(k).over('instrument') - 1.0).alias('mom_k'),
        (pl.col('illiq').rank().over('datetime') / (pl.col('illiq').count().over('datetime') + 1)).alias('illiq_r'),
    ])
    df = df.with_columns((-pl.col('mom_k') * pl.col('illiq_r')).alias('raw')){_EPILOGUE}
'''

# IDEA: Overnight/intraday decomposition. In A-shares the overnight gap
# (open/prev_close) carries momentum while the intraday move (close/open) tends
# to reverse. Signal = smoothed overnight return MINUS smoothed intraday return.
OVERNIGHT_INTRADAY = f'''
def factor(pricing_data, parameters):{_PRELUDE}
    k = parameters.get("k", 5)
    df = df.with_columns([
        (pl.col('open') / pl.col('close').shift(1).over('instrument') - 1.0).alias('overnight'),
        (pl.col('close') / pl.col('open') - 1.0).alias('intraday'),
    ])
    df = df.with_columns([
        pl.col('overnight').rolling_mean(window_size=k, min_periods=2).over('instrument').alias('on_k'),
        pl.col('intraday').rolling_mean(window_size=k, min_periods=2).over('instrument').alias('in_k'),
    ])
    df = df.with_columns((pl.col('on_k') - pl.col('in_k')).alias('raw')){_EPILOGUE}
'''

# IDEA: Volatility-scaled momentum (risk-adjusted). Raw k-day momentum divided by
# trailing realized volatility — rewards steady trends over noisy ones, a
# robust cross-sectional signal.
VOLSCALED_MOMENTUM = f'''
def factor(pricing_data, parameters):{_PRELUDE}
    k = parameters.get("k", 10)
    df = df.with_columns(
        (pl.col('close') / pl.col('close').shift(1).over('instrument') - 1.0).alias('ret1'))
    df = df.with_columns([
        (pl.col('close') / pl.col('close').shift(k).over('instrument') - 1.0).alias('mom_k'),
        pl.col('ret1').rolling_std(window_size=k, min_periods=2).over('instrument').alias('vol_k'),
    ])
    df = df.with_columns((pl.col('mom_k') / (pl.col('vol_k') + eps)).alias('raw')){_EPILOGUE}
'''

# IDEA: Price-volume divergence reversal. When price rises on shrinking volume
# (or falls on rising volume), the move is unconfirmed and tends to reverse.
# Signal = -(sign of k-day return) * (volume trend) — penalize unconfirmed moves.
PV_DIVERGENCE = f'''
def factor(pricing_data, parameters):{_PRELUDE}
    k = parameters.get("k", 5)
    df = df.with_columns([
        (pl.col('close') / pl.col('close').shift(k).over('instrument') - 1.0).alias('mom_k'),
        (pl.col('volume') / (pl.col('volume').rolling_mean(window_size=k, min_periods=2).over('instrument') + eps) - 1.0).alias('voltrend'),
    ])
    df = df.with_columns((-pl.col('mom_k') * pl.col('voltrend')).alias('raw')){_EPILOGUE}
'''

# IDEA: Garman-Klass low-volatility anomaly. Range-based (GK) volatility,
# trailing-averaged; low-vol names tend to outperform (defensive anomaly), so
# the signal is the negative cross-sectional rank of GK volatility.
GK_LOWVOL = f'''
def factor(pricing_data, parameters):{_PRELUDE}
    k = parameters.get("k", 10)
    df = df.with_columns(
        (0.5 * (pl.col('high') / (pl.col('low') + eps)).log() ** 2
         - (2.0 * 0.6931 - 1.0) * (pl.col('close') / (pl.col('open') + eps)).log() ** 2).alias('gk'))
    df = df.with_columns(
        pl.col('gk').rolling_mean(window_size=k, min_periods=2).over('instrument').alias('gk_k'))
    df = df.with_columns(
        (-(pl.col('gk_k').rank().over('datetime') / (pl.col('gk_k').count().over('datetime') + 1) - 0.5)).alias('raw')){_EPILOGUE}
'''


PROPOSALS = {
    "amihud_reversal": (AMIHUD_REVERSAL, {"k": {"type": "int", "low": 2, "high": 20}}),
    "overnight_intraday": (OVERNIGHT_INTRADAY, {"k": {"type": "int", "low": 2, "high": 20}}),
    "volscaled_momentum": (VOLSCALED_MOMENTUM, {"k": {"type": "int", "low": 5, "high": 40}}),
    "pv_divergence": (PV_DIVERGENCE, {"k": {"type": "int", "low": 2, "high": 20}}),
    "gk_lowvol": (GK_LOWVOL, {"k": {"type": "int", "low": 5, "high": 30}}),
}
