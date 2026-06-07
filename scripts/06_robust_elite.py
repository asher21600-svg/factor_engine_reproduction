#!/usr/bin/env python
"""V3 #1+#2: re-select elite factors with an OOS-robust, parsimony-penalized rule
and A/B it against the validation-only rule that overfit — using the EXISTING
evolution.json (no LLM re-run).

Candidate pool (all have code): the 5 live-Kimi elite + the 5 Claude-reasoned
proposals + the seed. Selection uses train+validation only; test is hold-out.

    python scripts/06_robust_elite.py
"""
import _bootstrap  # noqa: F401

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from fe import config
from fe.factors import SEED_SRC
from fe.factors.llm_proposals import PROPOSALS
from fe.integration import build_feature_matrix, train_predict, backtest
from fe.integration.robust_elite import (evaluate_candidates, select_validation_only,
                                         select_robust)


def _candidates(evo):
    cands = [("seed", SEED_SRC, {})]
    for i, el in enumerate(evo.get("elite", [])):
        cands.append((f"kimi_elite_{i}", el["code"], el.get("params", {})))
    for name, (src, pspace) in PROPOSALS.items():
        defaults = {k: int((v["low"] + v["high"]) / 2) if v["type"] == "int"
                    else (v["low"] + v["high"]) / 2 for k, v in pspace.items()}
        cands.append((f"claude_{name}", src, defaults))
    return cands


def _augmented_test(panel, specs, label="label"):
    feat, cols = build_feature_matrix(panel, specs, with_baseline=True, baseline="alpha158",
                                      qlib_kwargs={"market": "csi300"})
    mr = train_predict(panel, feat, cols, label=label)
    h = mr.test_metrics.headline()
    bt = backtest(mr.preds, panel, split="test")
    return h["IC"], h["RIC"], bt.metrics["AR"], bt.metrics["SR"]


def main():
    evo = json.load(open(config.OUTPUTS / "evolution.json"))
    p300 = pd.read_parquet(config.OUTPUTS / "csi300_panel.parquet")
    p500 = pd.read_parquet(config.OUTPUTS / "csi500_panel.parquet")

    print("== scoring candidate pool on real CSI300 (train/valid/test) ==")
    cands = evaluate_candidates(_candidates(evo), p300)
    print(f"{'candidate':22s} {'IC_train':>9} {'IC_valid':>9} {'IC_test':>9} "
          f"{'fit_val':>8} {'turn':>6} {'score':>8} {'#par':>4} {'#hid':>4} {'signOK':>6}")
    for c in sorted(cands, key=lambda c: c.fit_valid, reverse=True):
        print(f"{c.name:22s} {c.ic_train:+9.4f} {c.ic_valid:+9.4f} {c.ic_test:+9.4f} "
              f"{c.fit_valid:+8.3f} {c.turnover:6.3f} {c.robust_score:+8.3f} "
              f"{c.n_params:4d} {c.n_undeclared:4d} {str(c.sign_consistent):>6}")

    vo = select_validation_only(cands, k=config.ELITE_TOP_NODES)
    rb = select_robust(cands, k=config.ELITE_TOP_NODES)
    print(f"\nvalidation-only picks ({len(vo)}): {[c.name for c in vo]}")
    print(f"robust picks         ({len(rb)}): {[c.name for c in rb]}")

    out = {"validation_only": [c.name for c in vo], "robust": [c.name for c in rb],
           "candidates": {c.name: {"ic_train": c.ic_train, "ic_valid": c.ic_valid,
                                   "ic_test": c.ic_test, "n_params": c.n_params,
                                   "n_undeclared": c.n_undeclared,
                                   "turnover": c.turnover,
                                   "min_year_ic": c.min_year_ic,
                                   "objective_score": c.objective_score,
                                   "sign_consistent": c.sign_consistent} for c in cands},
           "universes": {}}

    print("\n== A/B: augmented multi-factor model on the TEST hold-out ==")
    print(f"{'universe':8s} {'baseline':>10} {'valid-only':>11} {'robust':>9}  (IC; then SR)")
    for uni, panel in [("csi300", p300), ("csi500", p500)]:
        b_ic, b_ric, b_ar, b_sr = _augmented_test(panel, [])
        vo_specs = [(c.name, c.code, c.params) for c in vo]
        rb_specs = [(c.name, c.code, c.params) for c in rb]
        v_ic, v_ric, v_ar, v_sr = _augmented_test(panel, vo_specs)
        r_ic, r_ric, r_ar, r_sr = _augmented_test(panel, rb_specs)
        print(f"{uni:8s} {b_ic:+10.4f} {v_ic:+11.4f} {r_ic:+9.4f}   IC")
        print(f"{'':8s} {b_sr:+10.2f} {v_sr:+11.2f} {r_sr:+9.2f}   SR")
        out["universes"][uni] = {
            "baseline": {"IC": b_ic, "RIC": b_ric, "AR": b_ar, "SR": b_sr},
            "validation_only": {"IC": v_ic, "RIC": v_ric, "AR": v_ar, "SR": v_sr},
            "robust": {"IC": r_ic, "RIC": r_ric, "AR": r_ar, "SR": r_sr}}

    with open(config.OUTPUTS / "robust_elite.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'robust_elite.json'}")


if __name__ == "__main__":
    main()
