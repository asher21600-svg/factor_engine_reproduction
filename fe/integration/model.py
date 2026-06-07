"""Multi-factor model (paper Integration module): LGBM on evolved factors +
Alpha-mini baseline features, trained point-in-time (no random splits).

`build_feature_matrix` runs each evolved factor program to a feature column and
joins the Alpha-mini set.  `train_predict` fits LightGBM on the train split
(early-stopped on validation) to predict forward returns, then scores the test
split — the composite signal `z_t` the paper feeds to the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..eval import evaluate_factor
from ..factors.contract import score_factor
from .baseline import alpha_mini_features


@dataclass
class ModelResult:
    preds: pd.DataFrame                 # [datetime, instrument, pred, split]
    feature_cols: list
    importance: dict = field(default_factory=dict)
    test_metrics: object = None         # FactorMetrics of the model score on test
    label: str = "fwd_ret_5"


def build_feature_matrix(panel: pd.DataFrame, factor_specs: list | None = None,
                         with_baseline: bool = True, baseline: str = "alpha_mini",
                         qlib_kwargs: dict | None = None,
                         extra: tuple | None = None) -> tuple[pd.DataFrame, list]:
    """factor_specs: list of (name, code, params). Returns (features_df, cols).

    baseline : 'alpha_mini' (portable, default) or 'alpha158' (exact Qlib
        Alpha158 — real-data path; falls back to alpha_mini if Qlib is absent).
    """
    parts = []
    cols: list[str] = []
    if with_baseline:
        base, base_cols = None, None
        if baseline == "alpha158":
            # pure-pandas Alpha158 (no qlib package needed) — the real-data baseline
            try:
                from .alpha158_pandas import alpha158_features
                base, base_cols = alpha158_features(panel)
                print(f"  [alpha158: {len(base_cols)} pandas features]")
            except Exception as e:  # noqa: BLE001
                print(f"  [alpha158_pandas failed: {e}] -> alpha_mini")
                base = None
        if base is None:
            base = alpha_mini_features(panel)
            base_cols = [c for c in base.columns if c not in ("datetime", "instrument")]
        parts.append(base.set_index(["datetime", "instrument"]))
        cols += base_cols

    for name, code, params in (factor_specs or []):
        sc = score_factor(code, panel, params).rename(columns={"value": name})
        parts.append(sc.set_index(["datetime", "instrument"]))
        cols.append(name)

    if extra is not None:
        edf, ecols = extra
        parts.append(edf.set_index(["datetime", "instrument"]))
        cols += list(ecols)

    feat = pd.concat(parts, axis=1).reset_index()
    return feat, cols


def train_predict(panel: pd.DataFrame, feat: pd.DataFrame, feature_cols: list,
                  label: str = "fwd_ret_5", primary_lag: int = 5,
                  seed: int = 0) -> ModelResult:
    import lightgbm as lgb

    keep = ["datetime", "instrument", label, "split"]
    data = feat.merge(panel[keep], on=["datetime", "instrument"], how="inner")
    data = data.replace([np.inf, -np.inf], np.nan)

    tr = data[(data["split"] == "train") & data[label].notna()].dropna(subset=feature_cols)
    va = data[(data["split"] == "valid") & data[label].notna()].dropna(subset=feature_cols)
    if len(tr) < 100:
        raise RuntimeError("not enough training rows")

    Xtr, ytr = tr[feature_cols], tr[label]
    Xva, yva = (va[feature_cols], va[label]) if len(va) else (Xtr, ytr)

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=50, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])

    # predict on the whole panel where features are present
    pred_rows = data.dropna(subset=feature_cols).copy()
    pred_rows["pred"] = model.predict(pred_rows[feature_cols])
    preds = pred_rows[["datetime", "instrument", "pred", "split"]]

    importance = dict(sorted(zip(feature_cols, model.feature_importances_),
                             key=lambda x: -x[1]))
    # model score IC on the test split
    scored = preds[preds["split"] == "test"][["datetime", "instrument", "pred"]] \
        .rename(columns={"pred": "value"})
    tm = evaluate_factor(scored, panel, primary_lag=primary_lag, split="test")
    return ModelResult(preds, feature_cols, importance, tm, label)
