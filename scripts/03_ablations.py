#!/usr/bin/env python
"""Phase 4c: ablations.

  * Bayesian micro-search ON vs OFF (paper Fig.5): same macro budget, compare the
    best-objective trajectory and final fitness.
  * Multi-island 1 vs 2 (paper Table 3): compare the discovered factor's quality.

(The CoE-vs-top-k ablation in Table 3 acts on the LLM macro context and is not
run here, since no API credential is available; noted in the report.)
"""
import _bootstrap  # noqa: F401
import json

import pandas as pd

from fe import config
from fe.factors import SEED_SRC, score_factor
from fe.eval import evaluate_factor
from fe.evolution import EvolutionEngine, EvolutionConfig


def run(panel, **kw):
    cfg = EvolutionConfig(verbose=False, **kw)
    eng = EvolutionEngine(SEED_SRC, panel, cfg)
    res = eng.run()
    return eng, res


def best_on(uni, code, params):
    panel = pd.read_parquet(config.OUTPUTS / f"{uni}_panel.parquet")
    m = evaluate_factor(score_factor(code, panel, params), panel, primary_lag=5, split="test")
    h = m.headline()
    return {"IC": round(h["IC"], 4), "RIC": round(h["RIC"], 4),
            "RICIR": round(h["RICIR"], 4), "fitness": round(m.fitness, 4)}


def main():
    panel = pd.read_parquet(config.OUTPUTS / "fast_panel.parquet")
    ITERS, TRIALS = 24, 14
    out = {}

    print("== Ablation 1: Bayesian micro-search ON vs OFF (Fig.5) ==")
    abl = {}
    for tag, use_bayes in [("with_bayes", True), ("without_bayes", False)]:
        eng, res = run(panel, iterations=ITERS, n_islands=2, micro_trials=TRIALS,
                       use_bayes=use_bayes, seed=2)
        abl[tag] = {"history": res.history, "best_fitness": round(res.best.fitness, 4),
                    "best_reward": round(res.best.reward, 4),
                    "best_transforms": sorted(t for t in res.best.transforms if t)}
        print(f"  {tag:14s}: best fitness={res.best.fitness:+.4f} reward={res.best.reward:+.4f} "
              f"transforms={abl[tag]['best_transforms']}")
    out["bayes_ablation"] = abl

    print("\n== Ablation 2: multi-island 1 vs 2 (Table 3) ==")
    isl = {}
    for n_isl in (1, 2):
        eng, res = run(panel, iterations=ITERS, n_islands=n_isl, micro_trials=TRIALS,
                       use_bayes=True, seed=3)
        csi = best_on("csi300", res.best.code, res.best.params)
        isl[str(n_isl)] = {"best_fitness_mining": round(res.best.fitness, 4),
                           "csi300_test": csi,
                           "best_transforms": sorted(t for t in res.best.transforms if t)}
        print(f"  islands={n_isl}: mining fitness={res.best.fitness:+.4f} | "
              f"csi300 test RIC={csi['RIC']:+.4f} fit={csi['fitness']:+.4f}")
    out["island_ablation"] = isl

    with open(config.OUTPUTS / "ablations.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> saved {config.OUTPUTS / 'ablations.json'}")


if __name__ == "__main__":
    main()
