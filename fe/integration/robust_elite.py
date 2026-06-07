"""OOS-robust, parsimony-penalized elite selection (report's V3 #1 + #2).

The live run overfit because elite factors were chosen by VALIDATION fitness
alone, then sign-flipped out-of-sample.  This module re-selects from a candidate
pool using only in-sample information (train + validation), preferring factors
that are:
  * sign-consistent and positive on BOTH the train and validation splits
    (a far better predictor of test transfer than validation alone), and
  * parsimonious — penalized for parameter count and, especially, for
    hardcoded `parameters.get()` defaults that never entered Bayesian tuning.

Test metrics are computed for REPORTING only — never used for selection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..factors import score_factor
from ..eval import evaluate_factor


def _complexity(code: str, params: dict) -> tuple[int, int]:
    """(#params the code reads, #undeclared hardcoded knobs)."""
    used = set(re.findall(r'parameters\.get\(\s*["\']([^"\']+)', code or ""))
    used.discard("epsilon")
    declared = set((params or {}).keys())
    undeclared = [u for u in used if u not in declared]
    return len(used), len(undeclared)


@dataclass
class Candidate:
    name: str
    code: str
    params: dict
    ic_train: float = float("nan")
    ic_valid: float = float("nan")
    fit_train: float = float("nan")
    fit_valid: float = float("nan")
    ic_test: float = float("nan")     # reporting only
    n_params: int = 0
    n_undeclared: int = 0

    @property
    def sign_consistent(self) -> bool:
        return (self.ic_train > 0) and (self.ic_valid > 0)

    @property
    def robust_score(self) -> float:
        # reward the WORSE of the two in-sample windows; penalize complexity & hidden knobs
        base = min(self.fit_train, self.fit_valid)
        return base - 0.03 * self.n_params - 0.15 * self.n_undeclared


def evaluate_candidates(candidates: list[tuple[str, str, dict]], panel: pd.DataFrame,
                        primary_lag: int = 5) -> list[Candidate]:
    """Score each (name, code, params) on train / valid / test splits + complexity."""
    out = []
    for name, code, params in candidates:
        params = params or {}
        try:
            scored = score_factor(code, panel, params)
        except Exception:  # noqa: BLE001
            continue
        mt = evaluate_factor(scored, panel, primary_lag=primary_lag, split="train")
        mv = evaluate_factor(scored, panel, primary_lag=primary_lag, split="valid")
        mte = evaluate_factor(scored, panel, primary_lag=primary_lag, split="test")
        npar, nund = _complexity(code, params)
        out.append(Candidate(
            name=name, code=code, params=params,
            ic_train=mt.headline()["IC"], ic_valid=mv.headline()["IC"],
            fit_train=mt.fitness, fit_valid=mv.fitness, ic_test=mte.headline()["IC"],
            n_params=npar, n_undeclared=nund))
    return out


def select_validation_only(cands: list[Candidate], k: int = 5) -> list[Candidate]:
    """Reproduce the overfit rule: top-k by validation fitness."""
    return sorted(cands, key=lambda c: c.fit_valid, reverse=True)[:k]


def select_robust(cands: list[Candidate], k: int = 5) -> list[Candidate]:
    """V3 rule: sign-consistent (positive on train AND valid), ranked by the
    parsimony-penalized worst-window score. Selective — may return < k."""
    eligible = [c for c in cands if c.sign_consistent]
    eligible.sort(key=lambda c: c.robust_score, reverse=True)
    return eligible[:k]


# ---------------------------------------------------------------------------
# V3 #3 — orthogonalize a factor vs the Alpha158 feature space (per date) and
# gate on the residual's marginal IC, i.e. signal NOT already in the baseline.
# ---------------------------------------------------------------------------
def orthogonalize_vs_features(scored: pd.DataFrame, base_long: pd.DataFrame,
                              base_cols: list) -> pd.DataFrame:
    """Cross-sectional residual of `scored['value']` on the Alpha158 features,
    per date: resid = factor - X·beta (OLS with intercept). Returns
    [datetime, instrument, value=resid]."""
    import numpy as np
    m = scored.merge(base_long, on=["datetime", "instrument"], how="inner")
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"] + list(base_cols))
    out = []
    for dt, g in m.groupby("datetime"):
        y = g["value"].to_numpy(dtype="float64")
        if len(g) < len(base_cols) + 5:           # under-determined -> just demean
            r = y - y.mean()
        else:
            X = np.column_stack([np.ones(len(g)), g[base_cols].to_numpy(dtype="float64")])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
        out.append(pd.DataFrame({"datetime": g["datetime"].to_numpy(),
                                 "instrument": g["instrument"].to_numpy(), "value": r}))
    return pd.concat(out, ignore_index=True) if out else scored[["datetime", "instrument", "value"]]


def marginal_metrics(resid: pd.DataFrame, panel: pd.DataFrame, primary_lag: int = 5):
    """(IC_train, IC_valid, IC_test) of an orthogonalized residual factor."""
    return tuple(evaluate_factor(resid, panel, primary_lag=primary_lag, split=s).headline()["IC"]
                 for s in ("train", "valid", "test"))


# ---------------------------------------------------------------------------
# V3 #4 — portfolio-aware objective: reward IC AND yearly stability, penalize
# turnover (factor-ranking churn ≈ trading cost), instead of IC/ICIR alone.
# ---------------------------------------------------------------------------
def portfolio_metrics(scored: pd.DataFrame, panel: pd.DataFrame, split: str = "valid",
                      primary_lag: int = 5,
                      w_stability: float = 0.5, w_turnover: float = 0.04) -> dict:
    """Return {ic, turnover, min_year_ic, score}. `turnover` ∈ [0,2] is 1 − mean
    consecutive-date cross-sectional rank autocorrelation (high churn ⇒ high cost).
    score = IC + w_stability·min_yearly_IC − w_turnover·(turnover·10)."""
    import numpy as np
    m = evaluate_factor(scored, panel, primary_lag=primary_lag, split=split)
    ic = m.headline()["IC"]
    keep = panel.loc[panel.get("split").eq(split), ["datetime", "instrument", f"fwd_ret_{primary_lag}"]]
    sub = scored.merge(keep, on=["datetime", "instrument"], how="inner")
    # turnover: 1 - mean rank-autocorrelation between consecutive dates
    wide = sub.pivot_table(index="datetime", columns="instrument", values="value").sort_index()
    ranks = wide.rank(axis=1)
    ac = ranks.corrwith(ranks.shift(1), axis=1)
    turnover = float(1.0 - ac.mean()) if ac.notna().any() else 1.0
    # yearly IC stability (worst year)
    rc = f"fwd_ret_{primary_lag}"
    yr = sub.assign(_y=sub["datetime"].astype("datetime64[ns]").dt.year)
    yearly = []
    for y, g in yr.groupby("_y"):
        gg = g[["value", rc]].dropna()
        if len(gg) > 50:
            yearly.append(gg["value"].corr(gg[rc]))
    min_year_ic = float(np.nanmin(yearly)) if yearly else float("nan")
    score = ic + w_stability * (min_year_ic if min_year_ic == min_year_ic else 0.0) \
        - w_turnover * (turnover * 10.0)
    return {"ic": ic, "turnover": turnover, "min_year_ic": min_year_ic, "score": score}
