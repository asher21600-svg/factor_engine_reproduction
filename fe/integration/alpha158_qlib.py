"""Exact Alpha158 features via Qlib's handler (paper-faithful baseline set).

Used only on the real-data path: the paper merges evolved factors with the full
Qlib Alpha158 set.  When Qlib + the cn_data bundle are installed, this returns
all 158 features in the pipeline's [datetime, instrument, <cols>] schema.  When
Qlib is absent (e.g. this environment), callers fall back to the portable
`alpha_mini` set (`fe/integration/baseline.py`).

NOTE: This path is implemented to Qlib's documented API but is NOT exercised in
the synthetic environment (no qlib/data here) — verify it on the machine that
hosts ~/.qlib/qlib_data/cn_data.
"""
from __future__ import annotations

import pandas as pd


class Alpha158Unavailable(RuntimeError):
    pass


def qlib_alpha158(market: str = "csi300",
                  start: str = "2008-01-01", end: str = "2024-12-31",
                  fit_start: str = "2008-01-01", fit_end: str = "2014-12-31",
                  provider_uri: str | None = None) -> tuple[pd.DataFrame, list]:
    """Return (features_df, feature_cols) with the full Alpha158 feature set.

    features_df columns: datetime, instrument, <158 alpha158 features>.
    Normalization is fit on the train window (fit_start..fit_end) to avoid
    look-ahead, matching the paper's split.
    """
    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158
    except Exception as e:  # noqa: BLE001
        raise Alpha158Unavailable(f"qlib/Alpha158 not importable: {e}") from e

    try:
        qlib.init(provider_uri=provider_uri or "~/.qlib/qlib_data/cn_data", region="cn")
        handler = Alpha158(instruments=market, start_time=start, end_time=end,
                           fit_start_time=fit_start, fit_end_time=fit_end)
        feat = handler.fetch(col_set="feature")           # MultiIndex (datetime, instrument)
    except Exception as e:  # noqa: BLE001
        raise Alpha158Unavailable(f"Alpha158 fetch failed: {e}") from e

    feat = feat.copy()
    # flatten possible MultiIndex columns to flat strings
    if isinstance(feat.columns, pd.MultiIndex):
        feat.columns = ["_".join(str(x) for x in c if x != "") for c in feat.columns]
    feat = feat.reset_index().rename(columns={"datetime": "datetime", "instrument": "instrument"})
    cols = [c for c in feat.columns if c not in ("datetime", "instrument")]
    feat["datetime"] = pd.to_datetime(feat["datetime"]).dt.normalize()
    return feat, cols
