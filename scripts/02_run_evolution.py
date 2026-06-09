#!/usr/bin/env python
"""Phase 4a: run the FE macro-micro co-evolution engine and persist the result.

Evolves from the paper's seed factor on the 'fast' panel (the mining universe),
using either the deterministic transform library or a configured live LLM
(Kimi/Moonshot, OpenAI-compatible, or Anthropic) for macro mutations, and
Optuna TPE for micro parameter search. Saves the discovered best factor
(code + params), the convergence history, and the full evolved pool.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
import os

import pandas as pd

from fe import config
from fe.factors import SEED_SRC
from fe.evolution import EvolutionEngine, EvolutionConfig
from fe.evolution.macro import describe_llm_config
from fe.integration.robust_elite import evaluate_candidates, select_robust


def extract_elite_validation(res, k=config.ELITE_TOP_NODES, thresh=config.FS_THRESHOLD):
    """Top-k distinct evolved programs with fitness > thresh (paper §4.3)."""
    nodes = [n for n in res.all_nodes()
             if n.valid and math.isfinite(n.fitness) and n.fitness > thresh]
    nodes.sort(key=lambda n: n.fitness, reverse=True)
    elite, seen = [], set()
    for n in nodes:
        if n.code in seen:
            continue
        seen.add(n.code)
        elite.append({
            "code": n.code,
            "params": {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                       for kk, vv in (n.params or {}).items()},
            "fitness": round(n.fitness, 5),
            "transforms": sorted(t for t in n.transforms if t),
        })
        if len(elite) >= k:
            break
    return elite


def _clean_json(x):
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_json(v) for v in x]
    try:
        import numpy as np
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
    except Exception:  # noqa: BLE001
        pass
    return x


def extract_elite_robust(res, panel, k=config.ELITE_TOP_NODES, max_candidates: int = 80):
    """Default V3 elite rule: train+valid sign consistency, low turnover,
    yearly stability, and parsimony. Test metrics are reporting only."""
    raw, seen = [], set()
    nodes = [n for n in res.all_nodes() if n.valid]
    nodes.sort(key=lambda n: (n.reward, n.fitness), reverse=True)
    for n in nodes[:max_candidates]:
        if not n.valid or n.code in seen:
            continue
        seen.add(n.code)
        raw.append((f"node_{n.node_id}", n.code, n.params or {}))
    cands = evaluate_candidates(raw, panel)
    picked = select_robust(cands, k=k)
    if not picked:
        return extract_elite_validation(res, k=k), {
            "rule": "robust_fallback_validation",
            "reason": "no sign-consistent train+validation candidates",
            "candidates": len(cands),
        }
    elite = []
    for c in picked:
        elite.append({
            "name": c.name,
            "code": c.code,
            "params": {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                       for kk, vv in (c.params or {}).items()},
            "fitness": round(c.fit_valid, 5),
            "robust_score": round(c.robust_score, 5),
            "ic_train": round(c.ic_train, 5),
            "ic_valid": round(c.ic_valid, 5),
            "ic_test_report_only": round(c.ic_test, 5),
            "turnover": round(c.turnover, 5) if c.turnover == c.turnover else None,
            "min_year_ic": round(c.min_year_ic, 5) if c.min_year_ic == c.min_year_ic else None,
            "n_params": c.n_params,
            "n_undeclared": c.n_undeclared,
        })
    return elite, {
        "rule": "robust_v3",
        "candidates": len(cands),
        "selected": [c.name for c in picked],
    }


def serialize_pool(res):
    nodes = []
    for tr in res.islands:
        for n in tr.nodes:
            nodes.append({
                "id": n.node_id, "parent": (n.parent.node_id if n.parent else None),
                "island": n.island, "depth": n.depth,
                "transforms": sorted(t for t in n.transforms if t),
                "reward": None if n.reward != n.reward else round(n.reward, 5),
                "fitness": None if n.fitness != n.fitness else round(n.fitness, 5),
                "score_components": _clean_json(n.score_components),
                "idea": n.idea, "change": n.change_summary,
                "params": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in (n.params or {}).items()},
                "valid": bool(n.valid),
            })
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="fast")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--islands", type=int, default=config.N_ISLANDS)
    ap.add_argument("--trials", type=int, default=18)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--llm-model", default=None,
                    help="override FE_LLM_MODEL for this run")
    ap.add_argument("--require-llm", action="store_true",
                    help="require at least one accepted live LLM mutation before fallback is allowed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objective", choices=["portfolio_v3", "portfolio_v4", "ic_only"], default="portfolio_v3",
                    help="portfolio_v3 (IC-robust default), portfolio_v4 (return-aware excess-return), "
                         "or ic_only (paper-faithful validation IC)")
    ap.add_argument("--elite-rule", choices=["robust", "validation"], default="robust",
                    help="default V3 robust elite selection or old validation-only top-k")
    ap.add_argument("--patience", type=int, default=50,
                    help="early-stop after this many non-improving iterations (0 disables)")
    ap.add_argument("--out", default=str(config.OUTPUTS / "evolution.json"))
    args = ap.parse_args()

    if args.require_llm:
        os.environ["FE_REQUIRE_LLM"] = "1"
    if args.llm_model:
        os.environ["FE_LLM_MODEL"] = args.llm_model

    panel = pd.read_parquet(config.OUTPUTS / f"{args.panel}_panel.parquet")
    cfg = EvolutionConfig(iterations=args.iterations, n_islands=args.islands,
                          micro_trials=args.trials, use_bayes=True,
                          use_llm=args.use_llm, llm_model=args.llm_model,
                          seed=args.seed, verbose=True, objective=args.objective,
                          patience=args.patience)
    llm_cfg = describe_llm_config() if args.use_llm else {}
    print(f"== Evolution on '{args.panel}' panel: {args.iterations} iters, "
          f"{args.islands} islands, {args.trials} Bayesian trials/step, "
          f"macro={'LLM+det' if args.use_llm else 'deterministic'}, "
          f"objective={args.objective}, elite={args.elite_rule}, patience={args.patience} ==")
    if args.use_llm:
        bases = ", ".join(llm_cfg.get("base_urls", [])) or "(Anthropic SDK)"
        print(f"   LLM provider={llm_cfg.get('provider')} model={llm_cfg.get('model')} "
              f"base={bases} key={'set' if llm_cfg.get('api_key_present') or llm_cfg.get('anthropic_key_present') else 'missing'}")
    eng = EvolutionEngine(SEED_SRC, panel, cfg)
    res = eng.run()

    best = res.best
    root = res.islands[0].root
    if args.elite_rule == "robust":
        elite, elite_meta = extract_elite_robust(res, panel)
    else:
        elite, elite_meta = extract_elite_validation(res), {"rule": "validation_only"}
    out = {
        "panel": args.panel,
        "config": {"iterations": args.iterations, "islands": args.islands,
                   "trials": args.trials, "use_llm": args.use_llm,
                   "llm_model": args.llm_model,
                   "require_llm": os.environ.get("FE_REQUIRE_LLM") in ("1", "true", "True"),
                   "objective": args.objective, "elite_rule": args.elite_rule,
                   "patience": args.patience},
        "elapsed_s": round(res.elapsed, 1), "n_evals": res.n_evals, "n_llm": res.n_llm,
        "llm": describe_llm_config() if args.use_llm else {},
        "seed_reward": round(root.reward, 5), "seed_fitness": round(root.fitness, 5),
        "seed_score_components": _clean_json(root.score_components),
        "best_reward": round(best.reward, 5), "best_fitness": round(best.fitness, 5),
        "best_score_components": _clean_json(best.score_components),
        "best_transforms": sorted(t for t in best.transforms if t),
        "best_params": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in best.params.items()},
        "best_idea": best.idea,
        "best_code": best.code,
        "elite": elite,
        "elite_meta": elite_meta,
        "history": res.history,
        "pool": serialize_pool(res),
        "n_nodes": len(res.all_nodes()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n== DONE in {res.elapsed:.1f}s, {res.n_evals} evals, "
          f"{len(res.all_nodes())} nodes ==")
    print(f"seed     : reward={root.reward:+.4f} fitness={root.fitness:+.4f}")
    print(f"best     : reward={best.reward:+.4f} fitness={best.fitness:+.4f} "
          f"transforms={out['best_transforms']}")
    print(f"best params: {out['best_params']}")
    print(f"-> saved {args.out}")


if __name__ == "__main__":
    main()
