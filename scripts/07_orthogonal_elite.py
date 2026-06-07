#!/usr/bin/env python
"""V3 #3: orthogonalize the robust-selected elite factors vs the Alpha158-128
feature space (per-date residual) and gate on marginal (residual) IC — feeding
the model only the part of each factor that Alpha158 does NOT already capture.
A/B vs baseline and vs the raw robust factors. Uses the existing run (no LLM).

    python scripts/07_orthogonal_elite.py
"""
import _bootstrap  # noqa: F401

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from fe import config
from fe.factors import SEED_SRC, score_factor
from fe.factors.llm_proposals import PROPOSALS
from fe.integration import build_feature_matrix, train_predict, backtest
from fe.integration.robust_elite import (evaluate_candidates, select_robust,
                                         orthogonalize_vs_features, marginal_metrics)


def _candidates(evo):
    cands = [("seed", SEED_SRC, {})]
    for i, el in enumerate(evo.get("elite", [])):
        cands.append((f"kimi_elite_{i}", el["code"], el.get("params", {})))
    for name, (src, pspace) in PROPOSALS.items():
        defaults = {k: int((v["low"] + v["high"]) / 2) if v["type"] == "int"
                    else (v["low"] + v["high"]) / 2 for k, v in pspace.items()}
        cands.append((f"claude_{name}", src, defaults))
    return cands


def _model_test(panel, base_long, base_cols, extra_df=None, extra_cols=None, label="label"):
    feat, cols = base_long, list(base_cols)
    if extra_df is not None and extra_cols:
        feat = base_long.merge(extra_df, on=["datetime", "instrument"], how="left")
        cols = cols + list(extra_cols)
    mr = train_predict(panel, feat, cols, label=label)
    h = mr.test_metrics.headline()
    bt = backtest(mr.preds, panel, split="test")
    return {"IC": h["IC"], "RIC": h["RIC"], "AR": bt.metrics["AR"], "SR": bt.metrics["SR"],
            "n_features": len(cols)}


def main():
    evo = json.load(open(config.OUTPUTS / "evolution.json"))
    p300 = pd.read_parquet(config.OUTPUTS / "csi300_panel.parquet")
    p500 = pd.read_parquet(config.OUTPUTS / "csi500_panel.parquet")

    rb = select_robust(evaluate_candidates(_candidates(evo), p300), k=config.ELITE_TOP_NODES)
    rb_specs = [(c.name, c.code, c.params) for c in rb]
    print(f"robust-selected factors: {[c.name for c in rb]}")

    out = {"robust": [c.name for c in rb], "universes": {}}
    for uni, panel in [("csi300", p300), ("csi500", p500)]:
        print(f"\n== {uni}: building Alpha158-128 base (once) ==")
        base_long, base_cols = build_feature_matrix(
            panel, [], with_baseline=True, baseline="alpha158", qlib_kwargs={"market": uni})

        # raw robust factors
        raw_parts = []
        for name, code, params in rb_specs:
            s = score_factor(code, panel, params).rename(columns={"value": name})
            raw_parts.append(s.set_index(["datetime", "instrument"]))
        raw_wide = pd.concat(raw_parts, axis=1).reset_index()
        raw_cols = [n for n, _, _ in rb_specs]

        # orthogonalized residuals + marginal-IC gate
        ortho_parts, ortho_cols, gate = [], [], []
        for name, code, params in rb_specs:
            s = score_factor(code, panel, params)
            resid = orthogonalize_vs_features(s, base_long, base_cols)
            ic_tr, ic_va, ic_te = marginal_metrics(resid, panel)
            keep = (ic_tr > 0) and (ic_va > 0)
            gate.append({"name": name, "marg_ic_train": ic_tr, "marg_ic_valid": ic_va,
                         "marg_ic_test": ic_te, "kept": keep})
            print(f"  ortho {name:22s} marginal IC train/valid/test = "
                  f"{ic_tr:+.4f}/{ic_va:+.4f}/{ic_te:+.4f}  {'KEEP' if keep else 'drop'}")
            if keep:
                on = f"o_{name}"
                ortho_parts.append(resid.rename(columns={"value": on}).set_index(["datetime", "instrument"]))
                ortho_cols.append(on)
        ortho_wide = pd.concat(ortho_parts, axis=1).reset_index() if ortho_parts else None

        base = _model_test(panel, base_long, base_cols)
        rawm = _model_test(panel, base_long, base_cols, raw_wide, raw_cols)
        orthm = (_model_test(panel, base_long, base_cols, ortho_wide, ortho_cols)
                 if ortho_wide is not None else base)
        out["universes"][uni] = {"baseline": base, "robust_raw": rawm,
                                 "robust_orthogonal": orthm, "gate": gate,
                                 "n_orthogonal_kept": len(ortho_cols)}
        print(f"  TEST IC: baseline {base['IC']:+.4f} | robust-raw {rawm['IC']:+.4f} | "
              f"robust-orthogonal {orthm['IC']:+.4f}   (kept {len(ortho_cols)} orth. factors)")
        print(f"  TEST SR: baseline {base['SR']:+.2f} | robust-raw {rawm['SR']:+.2f} | "
              f"robust-orthogonal {orthm['SR']:+.2f}")

    with open(config.OUTPUTS / "orthogonal_elite.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'orthogonal_elite.json'}")


if __name__ == "__main__":
    main()
