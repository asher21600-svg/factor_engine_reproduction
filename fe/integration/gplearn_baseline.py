"""GPLearn baseline — genetic-programming symbolic factors (paper Table-1 baseline).

Fits gplearn's SymbolicTransformer on the train split (features = Alpha-mini,
target = forward return), optimizing rank correlation (Spearman) with the label,
to synthesize K symbolic alpha features.  These are merged with the baseline and
fed to the same LightGBM model — the analog of the paper's "GPLearn 50 factors +
Alpha158" arm, letting us compare FE-evolved vs GP-evolved factors on identical data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import alpha_mini_features


def gplearn_features(panel: pd.DataFrame, n_features: int = 10, label: str = "label",
                     generations: int = 10, population: int = 800,
                     seed: int = 0, max_train_rows: int = 150000) -> tuple[pd.DataFrame, list]:
    """Return (features_df [datetime,instrument,gp_0..], cols). Fit on train split only.

    Caps the GP fit to `max_train_rows` (random sample) so it stays tractable on
    million-row real panels; the learned programs are then applied to all rows.
    """
    from gplearn.genetic import SymbolicTransformer

    base = alpha_mini_features(panel)
    feat_cols = [c for c in base.columns if c not in ("datetime", "instrument")]
    data = base.merge(panel[["datetime", "instrument", label, "split"]],
                      on=["datetime", "instrument"], how="inner").replace([np.inf, -np.inf], np.nan)

    tr = data[(data["split"] == "train")].dropna(subset=feat_cols + [label])
    if len(tr) < 500:
        raise RuntimeError("not enough training rows for gplearn")
    if len(tr) > max_train_rows:
        tr = tr.sample(n=max_train_rows, random_state=seed)

    st = SymbolicTransformer(
        n_components=n_features, generations=generations, population_size=population,
        hall_of_fame=max(n_features * 2, 20), tournament_size=20,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"),
        metric="spearman", parsimony_coefficient=0.001,
        random_state=seed, n_jobs=-1, verbose=0)
    st.fit(tr[feat_cols].values, tr[label].values)

    rows = data.dropna(subset=feat_cols).copy()
    gp = st.transform(rows[feat_cols].values)
    cols = [f"gp_{i}" for i in range(gp.shape[1])]
    out = pd.DataFrame(gp, columns=cols, index=rows.index)
    out["datetime"] = rows["datetime"].values
    out["instrument"] = rows["instrument"].values
    out = out.replace([np.inf, -np.inf], np.nan)
    return out[["datetime", "instrument"] + cols], cols
