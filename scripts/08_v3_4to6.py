#!/usr/bin/env python
"""Demonstrate V3 #4–#6 on the EXISTING run (no LLM re-run):

  #6  Parameter-contract enforcement — re-tune the best Kimi factor's 4 hidden
      (undeclared) knobs via Optuna; before vs after on valid fitness + test IC.
  #4  Portfolio-aware objective — re-rank the candidate pool by (IC + yearly
      stability − turnover) vs IC-only; show what each objective would pick.
  #5  Plateau-aware compute — from the 300-iter history, what early-stopping
      would have saved at the same best factor.

    python scripts/08_v3_4to6.py
"""
import _bootstrap  # noqa: F401

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from fe import config
from fe.factors import SEED_SRC, score_factor
from fe.factors.llm_proposals import PROPOSALS
from fe.eval import evaluate_factor
from fe.evolution import macro
from fe.evolution.micro import optimize_parameters
from fe.evolution.engine import EvolutionConfig  # noqa: F401 (documents patience field)
from fe.report.build_report import _frontier_milestones
from fe.integration.robust_elite import portfolio_metrics


def _candidates(evo):
    cands = [("seed", SEED_SRC, {})]
    for i, el in enumerate(evo.get("elite", [])):
        cands.append((f"kimi_elite_{i}", el["code"], el.get("params", {})))
    for name, (src, pspace) in PROPOSALS.items():
        defaults = {k: int((v["low"] + v["high"]) / 2) if v["type"] == "int"
                    else (v["low"] + v["high"]) / 2 for k, v in pspace.items()}
        cands.append((f"claude_{name}", src, defaults))
    return cands


def main():
    evo = json.load(open(config.OUTPUTS / "evolution.json"))
    panel = pd.read_parquet(config.OUTPUTS / "csi300_panel.parquet")
    out = {}

    # ---- #6: enforce parameter contract -> re-tune the hidden knobs ----
    print("== #6  parameter-contract enforcement: re-tune hidden knobs ==")
    best = evo["elite"][0]
    code, declared = best["code"], dict(best["params"])
    hidden_space = {k: v for k, v in macro._auto_declare_params(code, {}).items()
                    if k not in declared}
    print(f"  declared/tuned originally: {sorted(declared)}")
    print(f"  hidden knobs now tunable : {sorted(hidden_space)}")
    before = evaluate_factor(score_factor(code, panel, declared), panel, split="valid")
    before_te = evaluate_factor(score_factor(code, panel, declared), panel, split="test")
    res = optimize_parameters(code, hidden_space, panel, split="valid",
                              n_trials=30, seed=1, fixed=declared)
    after_te = evaluate_factor(score_factor(code, panel, res.best_params), panel, split="test")
    print(f"  BEFORE (hidden=default): valid fit {before.fitness:+.3f}, test IC {before_te.headline()['IC']:+.4f}")
    print(f"  AFTER  (hidden tuned)  : valid fit {res.best_metrics.fitness:+.3f}, "
          f"test IC {after_te.headline()['IC']:+.4f}  tuned={ {k: round(v,3) for k,v in res.best_params.items() if k not in declared} }")
    out["param_contract"] = {
        "hidden_knobs": sorted(hidden_space),
        "before": {"valid_fit": before.fitness, "test_ic": before_te.headline()["IC"]},
        "after": {"valid_fit": res.best_metrics.fitness, "test_ic": after_te.headline()["IC"],
                  "tuned": {k: v for k, v in res.best_params.items() if k not in declared}}}

    # ---- #4: portfolio-aware objective re-ranking ----
    print("\n== #4  portfolio-aware objective (IC + yearly stability − turnover) ==")
    rows = []
    for name, c, p in _candidates(evo):
        try:
            pm = portfolio_metrics(score_factor(c, panel, p), panel, split="valid")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:22s} skipped: {e}"); continue
        rows.append((name, pm["ic"], pm["turnover"], pm["min_year_ic"], pm["score"]))
    by_ic = sorted(rows, key=lambda r: r[1], reverse=True)
    by_port = sorted(rows, key=lambda r: r[4], reverse=True)
    print(f"  {'candidate':22s} {'IC':>8} {'turnover':>9} {'minYrIC':>8} {'portScore':>10}")
    for r in by_port:
        print(f"  {r[0]:22s} {r[1]:+8.4f} {r[2]:9.3f} {r[3]:+8.4f} {r[4]:+10.4f}")
    print(f"  IC-only top-3       : {[r[0] for r in by_ic[:3]]}")
    print(f"  portfolio-aware top3: {[r[0] for r in by_port[:3]]}")
    out["portfolio"] = {"ic_only_top": [r[0] for r in by_ic[:3]],
                        "portfolio_top": [r[0] for r in by_port[:3]],
                        "candidates": {r[0]: {"ic": r[1], "turnover": r[2],
                                              "min_year_ic": r[3], "score": r[4]} for r in rows}}

    # ---- #5: plateau-aware compute (post-hoc from history) ----
    print("\n== #5  plateau-aware compute (post-hoc) ==")
    milestones, plateau_iter, n_iters = _frontier_milestones(evo)
    elapsed = evo.get("elapsed_s", 0)
    per_iter = elapsed / max(1, n_iters)
    out["plateau"] = {"plateau_iter": plateau_iter, "n_iters": n_iters,
                      "elapsed_s": elapsed, "scenarios": {}}
    for pat in (30, 50):
        stop = min(n_iters, plateau_iter + pat)
        saved = (n_iters - stop) * per_iter
        out["plateau"]["scenarios"][pat] = {"stop_iter": stop, "saved_s": saved,
                                            "saved_frac": saved / max(1, elapsed)}
        print(f"  patience={pat}: stop at iter {stop}/{n_iters} → save "
              f"{saved/3600:.1f}h ({100*saved/max(1,elapsed):.0f}%) at the SAME best factor")

    with open(config.OUTPUTS / "v3_4to6.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n-> saved {config.OUTPUTS / 'v3_4to6.json'}")


if __name__ == "__main__":
    main()
