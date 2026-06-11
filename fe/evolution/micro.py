"""Micro-level parameter optimization — Bayesian search (paper §4.2, "Implementation").

Optuna's TPE sampler is an Expected-Improvement method that splits trials at a
performance quantile; we set that quantile to the paper's top-25% threshold
(y* = top 25% of observed scores).  The default objective is now V3
portfolio-aware scoring on train/validation, while `ic_only` preserves the
original paper-faithful validation `combined_score` path.

The paper notes support for "TPE, Gaussian Process-based methods, and other
probabilistic optimization algorithms" — `sampler='tpe'` (default) is used;
`sampler='random'` is available as a control for the Bayesian-vs-none ablation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import optuna

from ..config import LAGS, BAYES_TOP_QUANTILE
from ..eval import evaluate_objective
from ..factors.contract import score_factor, compile_factor, FactorRunError

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


@dataclass
class MicroResult:
    best_params: dict
    best_score: float
    best_metrics: object
    n_trials: int
    history: list   # per-trial scores (for the Bayesian-vs-none ablation plot)
    best_components: dict | None = None


def _suggest(trial, name, spec):
    t = spec.get("type", "float")
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    log = bool(spec.get("log", False))
    return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=log)


def optimize_parameters(code: str, param_space: dict, panel,
                        split: str = "valid", lags=LAGS, primary_lag: int = 5,
                        n_trials: int = 20, sampler: str = "tpe",
                        seed: int = 0, fixed: dict | None = None,
                        objective: str = "portfolio_v3") -> MicroResult:
    """Bayesian search over `param_space`; maximize the configured objective."""
    fixed = fixed or {}
    fn = compile_factor(code)   # compile once; reuse across trials

    if not param_space:
        # nothing to tune — evaluate once with fixed params
        try:
            obj = evaluate_objective(score_factor(fn, panel, fixed), panel,
                                     code=code, params=fixed, objective=objective,
                                     lags=lags, primary_lag=primary_lag, split=split)
            return MicroResult(dict(fixed), obj.score, obj.metrics, 1, [obj.score],
                               obj.components)
        except FactorRunError:
            return MicroResult(dict(fixed), float("-inf"), None, 1, [float("-inf")])

    history: list = []

    def _trial_objective(trial):          # NB: must not shadow the `objective` str param
        params = dict(fixed)
        for name, spec in param_space.items():
            params[name] = _suggest(trial, name, spec)
        try:
            obj = evaluate_objective(score_factor(fn, panel, params), panel,
                                     code=code, params=params, objective=objective,
                                     lags=lags, primary_lag=primary_lag, split=split)
            score = obj.score
        except FactorRunError:
            score = -1e9
        if score != score:    # NaN
            score = -1e9
        trial.set_user_attr("score", score)
        history.append(score)
        return score

    if sampler == "random":
        smp = optuna.samplers.RandomSampler(seed=seed)
    else:
        # TPE with EI quantile = top 25% (paper's y* threshold)
        smp = optuna.samplers.TPESampler(
            seed=seed, gamma=lambda n: max(1, int(BAYES_TOP_QUANTILE * n)))

    study = optuna.create_study(direction="maximize", sampler=smp)
    study.optimize(_trial_objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(fixed)
    best_params.update(study.best_params)
    # recompute metrics at the best params (full bundle)
    try:
        best_obj = evaluate_objective(score_factor(fn, panel, best_params), panel,
                                      code=code, params=best_params, objective=objective,
                                      lags=lags, primary_lag=primary_lag, split=split)
        best_metrics = best_obj.metrics
        best_score = best_obj.score
        best_components = best_obj.components
    except FactorRunError:
        best_metrics, best_score, best_components = None, float("-inf"), None
    return MicroResult(best_params, best_score, best_metrics, n_trials, history,
                       best_components)
