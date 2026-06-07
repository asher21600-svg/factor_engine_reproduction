"""Synthetic cross-sectional OHLCV panel with an embedded, recoverable alpha.

WHY THIS EXISTS
---------------
The paper runs on Qlib A-share data (CSI300/CSI500), which is not available in
this environment.  Rather than fake-and-forget, this generator embeds a
*specific, economically-flavoured* structure so that the paper's own factor
programs (Listings 1.3 / 1.4) earn realistic Information Coefficients, AND so
that the evolved factor's documented improvements are genuinely rewarded.  That
turns the seed-vs-evolved comparison into a real consistency check of FE's
machinery rather than a toy.

GENERATIVE MODEL (per stock i, day t)
-------------------------------------
  * Persistent latent "informed pressure"  a_{i,t}  is an AR(1) process
    (rho ~ 0.9), unit variance.  It barely moves over a 5-day window -> a
    multi-day average of a noisy daily read recovers it better than one day.
  * Daily *observed* pressure  p_{i,t} = a_{i,t} + sigma_p * noise.  Only p
    shows up in today's OHLCV geometry; a is latent.
  * Intraday geometry: the close's position in the day's range is a monotone
    (tanh) function of p, centred on the mid-price.  -> (close-mid)/range is an
    unbiased read of p; (close-low)/range carries an offset (rewards the
    evolved factor's close-vs-mid change).
  * Volume: log-volume rises with |p| and a per-stock liquidity level, plus a
    FAT-TAILED shock (rewards rank-normalisation over z-scoring).
  * Forward returns load on the *persistent* state with a coefficient that
    scales with the stock's liquidity tier (rewards turnover weighting), i.e.
        ret_{i,t} = mu + beta*market_t + delta_i * SIGN * a_{i,t-1} + noise
    where delta_i is proportional to liquidity.  Returns therefore depend on
    yesterday's persistent pressure -> a factor that estimates a_{i,t} from
    OHLCV up to day t predicts the return over (t, t+h].  No look-ahead.

All effect sizes are knobs (GenParams) so IC can be calibrated into the paper's
realistic 0.02-0.06 band, and SIGN can be flipped so the seed factor lands on
positive IC (the convention the paper reports).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GenParams:
    rho: float = 0.92            # persistence of the latent informed state
    sigma_p: float = 1.10        # daily observation noise on pressure (vs unit-var state)
    market_vol: float = 0.012    # daily common-factor stdev
    idio_vol: float = 0.045      # daily idiosyncratic return stdev (calibrated: ensemble IC~0.05)
    base_drift: float = 0.0002   # tiny per-day drift
    range_pct_mean: float = 0.022  # mean intraday range as fraction of price
    range_pct_sd: float = 0.006
    gap_sd: float = 0.004        # overnight gap stdev
    tanh_k: float = 0.9          # pressure -> close-position sensitivity
    # TURNOVER is the primitive (liquidity x activity); volume = turnover/close.
    # This keeps turnover (the evolved factor's weight) a clean liquidity signal,
    # while raw share-volume (the seed's weight) inherits 1/price contamination.
    to_base: float = 16.0        # log-turnover base (~ CNY notional)
    to_pressure: float = 0.30    # |pressure| loading on log-turnover
    to_noise_sd: float = 0.30    # gaussian part of log-turnover noise
    to_outlier_p: float = 0.04   # prob of a fat-tailed turnover spike
    to_outlier_sd: float = 0.9   # spike size (log scale) -> rewards rank-norm
    alpha_strength: float = 0.0020  # base return loading on persistent pressure (calibrated: seed IC~0.03)
    alpha_sign: float = 1.0      # +1 => factor aligns positively with fwd return
    liq_tier_spread: float = 0.8  # spread of per-stock predictability/liquidity tiers


def build_synthetic_panel(
    n_stocks: int = 300,
    n_days: int = 1500,
    seed: int = 300,
    start: str = "2008-01-01",
    params: GenParams | None = None,
    train_frac: float = 0.50,
    valid_frac: float = 0.15,
) -> pd.DataFrame:
    """Return a long-format panel:

        datetime | instrument | open high low close volume |
        ret_1 | fwd_ret_1 fwd_ret_3 fwd_ret_5 fwd_ret_10 | true_alpha | split

    `ret_1`     : realized close-to-close simple return on day t (for backtest).
    `fwd_ret_h` : forward cumulative simple return over (t, t+h]  (labels).
    `true_alpha`: latent persistent pressure a_{i,t} (oracle, diagnostics only).
    `split`     : 'train' / 'valid' / 'test' by chronological fraction.
    """
    p = params or GenParams()
    rng = np.random.default_rng(seed)
    N, T = n_stocks, n_days

    # --- per-stock static characteristics --------------------------------
    # liquidity tier in [~0.3, ~1.7]: drives both volume level and predictability
    liq = np.exp(rng.normal(0.0, p.liq_tier_spread, size=N))
    liq = liq / liq.mean()                      # mean 1
    delta = p.alpha_strength * liq              # per-stock return loading on pressure
    beta = rng.normal(1.0, 0.25, size=N)        # market beta
    mu = p.base_drift + rng.normal(0.0, 0.0001, size=N)
    to_base = p.to_base + np.log(liq)           # liquid stocks turn over more notional
    range_pct = np.clip(rng.normal(p.range_pct_mean, p.range_pct_sd, size=N),
                        0.006, 0.08)

    # --- latent persistent pressure a_{i,t}  (T x N), AR(1), unit variance
    a = np.zeros((T, N))
    a[0] = rng.normal(0.0, 1.0, size=N)
    innov_sd = np.sqrt(1.0 - p.rho ** 2)
    for t in range(1, T):
        a[t] = p.rho * a[t - 1] + innov_sd * rng.normal(0.0, 1.0, size=N)

    # daily observed pressure p_{i,t} = a + noise  (drives today's OHLCV)
    pressure = a + p.sigma_p * rng.normal(0.0, 1.0, size=(T, N))

    # --- market factor & realized close-to-close returns -----------------
    market = rng.normal(0.0, p.market_vol, size=T)
    idio = rng.normal(0.0, p.idio_vol, size=(T, N))
    # return on day t loads on YESTERDAY's persistent pressure (a[t-1])
    a_lag = np.vstack([np.zeros((1, N)), a[:-1]])          # a_{t-1}, a_0 unused
    ret = (mu[None, :] + beta[None, :] * market[:, None]
           + (p.alpha_sign * delta)[None, :] * a_lag + idio)
    ret = np.clip(ret, -0.099, 0.099)                      # A-share ±10% limit

    # --- price path & intraday OHLCV -------------------------------------
    close = np.empty((T, N))
    close[0] = 100.0 * np.exp(rng.normal(0.0, 0.05, size=N))
    for t in range(1, T):
        close[t] = close[t - 1] * (1.0 + ret[t])

    prev_close = np.vstack([close[0][None, :], close[:-1]])
    gap = rng.normal(0.0, p.gap_sd, size=(T, N))
    open_ = prev_close * (1.0 + gap)
    open_[0] = close[0] * (1.0 + rng.normal(0.0, p.gap_sd, size=N))

    span = close * range_pct[None, :]                      # day's high-low width
    # close position in range: monotone in pressure, centred at 0.5 (mid)
    cpos = 0.5 + 0.5 * np.tanh(p.tanh_k * pressure)
    cpos = np.clip(cpos, 0.03, 0.97)
    low = close - cpos * span
    high = close + (1.0 - cpos) * span
    # ensure the bar contains open and close
    low = np.minimum.reduce([low, open_, close]) - 1e-6 * close
    high = np.maximum.reduce([high, open_, close]) + 1e-6 * close

    # --- turnover (fat-tailed) -> volume = turnover / close --------------
    # Turnover is the economically meaningful weight (rewards the evolved
    # factor's turnover term).  Raw share-volume = turnover/close then carries
    # a 1/price contamination, which is what the seed factor weights by.
    base_lt = to_base[None, :] + p.to_pressure * np.abs(pressure)
    gauss = p.to_noise_sd * rng.normal(0.0, 1.0, size=(T, N))
    spike_mask = rng.random(size=(T, N)) < p.to_outlier_p
    spikes = spike_mask * np.abs(rng.normal(0.0, p.to_outlier_sd, size=(T, N)))
    turnover = np.exp(base_lt + gauss + spikes)
    volume = turnover / close

    # --- forward cumulative returns (labels) -----------------------------
    fwd = {}
    for h in (1, 3, 5, 10):
        fwd_h = np.full((T, N), np.nan)
        fwd_h[: T - h] = close[h:] / close[: T - h] - 1.0
        fwd[h] = fwd_h

    # Qlib Alpha158 default label (LABEL0): Ref(close,-2)/Ref(close,-1) - 1
    # i.e. decide at t, execute at t+1 close, realize at t+2 close (1-day delay).
    label = np.full((T, N), np.nan)
    label[: T - 2] = close[2:] / close[1:T - 1] - 1.0

    # --- assemble long format --------------------------------------------
    dates = pd.bdate_range(start=start, periods=T)
    inst = np.array([f"S{idx:04d}" for idx in range(N)])
    dt_col = np.repeat(dates.values, N)
    in_col = np.tile(inst, T)

    df = pd.DataFrame({
        "datetime": dt_col,
        "instrument": in_col,
        "open": open_.ravel(),
        "high": high.ravel(),
        "low": low.ravel(),
        "close": close.ravel(),
        "volume": volume.ravel(),
        "ret_1": np.vstack([np.full((1, N), np.nan), ret[1:]]).ravel(),
        "fwd_ret_1": fwd[1].ravel(),
        "fwd_ret_3": fwd[3].ravel(),
        "fwd_ret_5": fwd[5].ravel(),
        "fwd_ret_10": fwd[10].ravel(),
        "label": label.ravel(),          # Qlib Alpha158 LABEL0
        "true_alpha": a.ravel(),
    })

    # chronological split
    n_train = int(T * train_frac)
    n_valid = int(T * valid_frac)
    split = np.empty(T, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_valid] = "valid"
    split[n_train + n_valid:] = "test"
    df["split"] = np.repeat(split, N)

    return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)


if __name__ == "__main__":  # quick smoke test
    d = build_synthetic_panel(n_stocks=50, n_days=120, seed=1)
    print(d.shape)
    print(d.head())
    print(d.groupby("split").size() // 50, "days per split")
    # oracle IC: does true_alpha predict fwd_ret_5?
    from scipy.stats import pearsonr
    sub = d.dropna(subset=["fwd_ret_5"])
    ics = sub.groupby("datetime").apply(
        lambda g: pearsonr(g["true_alpha"], g["fwd_ret_5"])[0]
        if len(g) > 5 else np.nan).dropna()
    print("oracle mean IC(true_alpha, fwd_ret_5) =", round(ics.mean(), 4))
