#!/usr/bin/env python
"""Phase 4b: evaluate the discovered factor and reproduce the paper's tables.

Consumes outputs/evolution.json (the engine's best factor) and produces
outputs/results.json with, for each universe (csi300/csi500):
  * single-factor TEST metrics: seed vs paper's evolved artifact (Listing 1.4)
    vs the FE-engine-evolved factor;
  * multi-factor model TEST IC: baseline (Alpha-mini) vs augmented (+FE factor);
  * backtest metrics + equity curves for baseline vs augmented;
  * yearly IC (alpha-decay analog, paper Fig.4).
Plus a multi-seed robustness pass with error bars (single-factor seed vs evolved).
"""
import _bootstrap  # noqa: F401
import json
import os

import numpy as np
import pandas as pd


def _native(o):
    """JSON serializer for numpy scalar/array types."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")

from fe import config
from fe.data import build_synthetic_panel, GenParams
from fe.factors import SEED_SRC, EVOLVED_SRC, score_factor
from fe.eval import evaluate_factor, xs_corr_by_date
from fe.integration import build_feature_matrix, train_predict, backtest

# Paper Table 1 reference rows. Our reproduction skips the report-bootstrapping
# module (no report corpus), so it is the **FE-alpha** variant — FE-alpha-2 is the
# fair comparison; FE-report-2 is the aspirational ceiling (needs the corpus).
# GPLearn (genetic-programming symbolic factors) is the other Table-1 baseline.
PAPER = {
    "csi300": {"Alpha158": {"IC": 0.0299, "ICIR": 0.2008, "RIC": 0.0331, "RICIR": 0.2164,
                            "AR": 0.0840, "MDD": 0.1749, "IR": 0.7440, "SR": 0.4196},
               "GPLearn": {"IC": 0.0292, "ICIR": 0.1971, "RIC": 0.0321, "RICIR": 0.2120,
                           "AR": 0.0814, "MDD": 0.1599, "IR": 0.7337, "SR": 0.4152},
               "FE-alpha-2": {"IC": 0.0315, "ICIR": 0.2211, "RIC": 0.0344, "RICIR": 0.2360,
                              "AR": 0.0943, "MDD": 0.1507, "IR": 0.8241, "SR": 0.4762},
               "FE-report-2": {"IC": 0.0474, "ICIR": 0.3185, "RIC": 0.0475, "RICIR": 0.3146,
                               "AR": 0.1899, "MDD": 0.1261, "IR": 1.6001, "SR": 1.0093}},
    "csi500": {"Alpha158": {"IC": 0.0403, "ICIR": 0.3100, "RIC": 0.0416, "RICIR": 0.3172,
                            "AR": 0.0197, "MDD": 0.2517, "IR": 0.2152, "SR": 0.0089},
               "GPLearn": {"IC": 0.0409, "ICIR": 0.3190, "RIC": 0.0427, "RICIR": 0.3113,
                           "AR": 0.0272, "MDD": 0.2279, "IR": 0.2751, "SR": 0.0451},
               "FE-alpha-2": {"IC": 0.0417, "ICIR": 0.3183, "RIC": 0.0434, "RICIR": 0.3293,
                              "AR": 0.0399, "MDD": 0.2384, "IR": 0.3770, "SR": 0.1064},
               "FE-report-2": {"IC": 0.0536, "ICIR": 0.4140, "RIC": 0.0487, "RICIR": 0.3744,
                               "AR": 0.0836, "MDD": 0.2151, "IR": 0.6719, "SR": 0.2945}},
}


def single_factor_row(code, params, panel):
    m = evaluate_factor(score_factor(code, panel, params), panel, primary_lag=5, split="test")
    h = m.headline()
    return {"IC": h["IC"], "ICIR": h["ICIR"], "RIC": h["RIC"], "RICIR": h["RICIR"],
            "fitness": m.fitness, "combined_score": m.combined_score}


def yearly_ic(preds, panel):
    sc = preds[preds["split"] == "test"][["datetime", "instrument", "pred"]].rename(columns={"pred": "value"})
    merged = sc.merge(panel[["datetime", "instrument", "fwd_ret_5"]], on=["datetime", "instrument"], how="inner")
    ic = xs_corr_by_date(merged, "value", "fwd_ret_5", "pearson")
    ric = xs_corr_by_date(merged, "value", "fwd_ret_5", "spearman")
    df = pd.DataFrame({"ic": ic, "ric": ric})
    df["year"] = pd.to_datetime(df.index).year
    out = df.groupby("year").mean()
    return {int(y): {"ic": round(r.ic, 4), "ric": round(r.ric, 4)} for y, r in out.iterrows()}


def evaluate_universe(uni, evolved_code, evolved_params, elite_specs=None,
                      baseline="alpha_mini", label="fwd_ret_5", use_benchmark=False,
                      gplearn=False, factor_mode="orthogonal",
                      label_mode="date_demeaned"):
    panel = pd.read_parquet(config.OUTPUTS / f"{uni}_panel.parquet")
    # provenance: real Qlib panels span 2008-2024 (~4133 dates); synthetic ~1500
    data_source = "real" if panel["datetime"].nunique() > 3000 else "synthetic"
    res = {"single_factor": {}, "model": {}, "backtest": {}, "yearly_ic": {},
           "paper": PAPER.get(uni, {}),
           "config": {"baseline": baseline, "label": label, "data_source": data_source,
                      "benchmark": "index" if use_benchmark else "equal_weight",
                      "factor_mode": factor_mode, "label_mode": label_mode}}

    # real index benchmark (paper-faithful) if requested & available
    bench = None
    if use_benchmark:
        try:
            from fe.data.qlib_loader import load_benchmark
            bench = load_benchmark(uni)
        except Exception as e:  # noqa: BLE001
            print(f"  [benchmark unavailable: {e}] -> equal-weight market")

    # --- single-factor (test) ---
    res["single_factor"]["seed"] = single_factor_row(SEED_SRC, {}, panel)
    res["single_factor"]["paper_evolved_artifact"] = single_factor_row(EVOLVED_SRC, {}, panel)
    res["single_factor"]["fe_engine_evolved"] = single_factor_row(evolved_code, evolved_params, panel)

    # --- model: baseline vs augmented (augmented = top-k elite evolved factors, §4.3) ---
    qkw = {"market": uni, "start": "2008-01-01", "end": "2024-12-31",
           "fit_start": "2008-01-01", "fit_end": "2014-12-31"}
    aug = elite_specs or [("fe_evolved", evolved_code, evolved_params)]
    res["config"]["n_elite_factors"] = len(aug)
    specs = {"baseline": [], "augmented": aug}
    for name, sp in specs.items():
        feat, cols = build_feature_matrix(panel, sp, with_baseline=True,
                                          baseline=baseline, qlib_kwargs=qkw,
                                          orthogonalize_factors=(name == "augmented"
                                                                 and factor_mode == "orthogonal"))
        mr = train_predict(panel, feat, cols, label=label, label_mode=label_mode)
        h = mr.test_metrics.headline()
        res["model"][name] = {"IC": h["IC"], "ICIR": h["ICIR"], "RIC": h["RIC"],
                              "RICIR": h["RICIR"], "n_features": len(cols)}
        bt = backtest(mr.preds, panel, split="test", benchmark=bench)
        res["backtest"][name] = {k: round(v, 5) for k, v in bt.metrics.items()}
        bt.equity.to_csv(config.OUTPUTS / f"equity_{uni}_{name}.csv", index=False)
        res["yearly_ic"][name] = yearly_ic(mr.preds, panel)
        res["model"][name]["top_features"] = list(mr.importance.items())[:6]

    # --- optional GPLearn baseline arm (symbolic GP factors + baseline) ---
    if gplearn:
        try:
            from fe.integration.gplearn_baseline import gplearn_features
            gpf = gplearn_features(panel, n_features=8, label=label,
                                   generations=6, population=400)
            feat, cols = build_feature_matrix(panel, [], with_baseline=True,
                                              baseline=baseline, qlib_kwargs=qkw, extra=gpf)
            mr = train_predict(panel, feat, cols, label=label, label_mode=label_mode)
            h = mr.test_metrics.headline()
            res["model"]["gplearn"] = {"IC": h["IC"], "ICIR": h["ICIR"], "RIC": h["RIC"],
                                       "RICIR": h["RICIR"], "n_features": len(cols)}
            bt = backtest(mr.preds, panel, split="test", benchmark=bench)
            res["backtest"]["gplearn"] = {k: round(v, 5) for k, v in bt.metrics.items()}
            bt.equity.to_csv(config.OUTPUTS / f"equity_{uni}_gplearn.csv", index=False)
            res["yearly_ic"]["gplearn"] = yearly_ic(mr.preds, panel)
            print(f"  gplearn arm: model IC={h['IC']:+.4f} (vs augmented {res['model']['augmented']['IC']:+.4f})")
        except Exception as e:  # noqa: BLE001
            print(f"  [gplearn arm skipped: {e}]")
    return res


def multiseed_robustness(evolved_code, evolved_params, seeds=(101, 202, 303, 404, 505)):
    gp = GenParams()
    rows = {"seed": [], "fe_engine_evolved": []}
    for ds in seeds:
        panel = build_synthetic_panel(n_stocks=300, n_days=1500, seed=ds, params=gp)
        rows["seed"].append(single_factor_row(SEED_SRC, {}, panel))
        rows["fe_engine_evolved"].append(single_factor_row(evolved_code, evolved_params, panel))
    summ = {}
    for k, lst in rows.items():
        summ[k] = {met: {"mean": round(float(np.mean([r[met] for r in lst])), 4),
                         "std": round(float(np.std([r[met] for r in lst])), 4)}
                   for met in ("IC", "RIC", "fitness")}
    summ["seeds"] = list(seeds)
    return summ


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["alpha_mini", "alpha158"], default="alpha_mini",
                    help="baseline feature set ('alpha158' = exact Qlib set, real-data path)")
    ap.add_argument("--label", default="fwd_ret_5",
                    help="model label column ('label' = Qlib Alpha158 LABEL0)")
    ap.add_argument("--use-benchmark", action="store_true",
                    help="use the real CSI index as backtest benchmark (needs Qlib)")
    ap.add_argument("--gplearn", action="store_true",
                    help="also run the GPLearn (genetic-programming) baseline arm")
    ap.add_argument("--factor-mode", choices=["orthogonal", "raw"], default="orthogonal",
                    help="integration mode for FE factors; default residualizes vs the baseline features")
    ap.add_argument("--label-mode", choices=["date_demeaned", "raw"], default="date_demeaned",
                    help="model target; date_demeaned is the excess/market-neutral default")
    ap.add_argument("--universes", nargs="*", default=["csi300", "csi500"])
    args = ap.parse_args()

    with open(config.OUTPUTS / "evolution.json") as f:
        evo = json.load(f)
    evolved_code, evolved_params = evo["best_code"], evo["best_params"]
    elite = evo.get("elite") or []
    elite_specs = [(f"fe_elite_{i}", e["code"], e["params"]) for i, e in enumerate(elite)] \
        or [("fe_evolved", evolved_code, evolved_params)]
    print("Discovered factor transforms:", evo["best_transforms"], "params:", evolved_params)
    print(f"elite factors (FS>{config.FS_THRESHOLD}): {len(elite_specs)} "
          f"-> augmented multi-factor model")
    print(f"config: baseline={args.baseline} label={args.label} "
          f"label_mode={args.label_mode} factor_mode={args.factor_mode} "
          f"benchmark={'index' if args.use_benchmark else 'equal_weight'}")

    out = {"evolution_summary": {k: evo[k] for k in
                                 ("panel", "best_transforms", "best_params", "seed_fitness",
                                  "best_fitness", "seed_reward", "best_reward", "elapsed_s",
                                  "n_evals", "n_nodes")},
           "run_config": {"baseline": args.baseline, "label": args.label,
                          "label_mode": args.label_mode, "factor_mode": args.factor_mode,
                          "benchmark": "index" if args.use_benchmark else "equal_weight",
                          "orthogonal_max_features": os.environ.get("FE_ORTHOGONAL_MAX_FEATURES")},
           "universes": {}}
    out["evolution_summary"]["objective"] = evo.get("config", {}).get("objective", "ic_only")
    out["evolution_summary"]["elite_rule"] = evo.get("config", {}).get("elite_rule", "validation")
    for uni in args.universes:
        print(f"\n== evaluating {uni} ==")
        out["universes"][uni] = evaluate_universe(
            uni, evolved_code, evolved_params, elite_specs=elite_specs,
            baseline=args.baseline, label=args.label, use_benchmark=args.use_benchmark,
            gplearn=args.gplearn, factor_mode=args.factor_mode,
            label_mode=args.label_mode)
        out["run_config"]["data_source"] = out["universes"][uni]["config"]["data_source"]
        sf = out["universes"][uni]["single_factor"]
        md = out["universes"][uni]["model"]
        bt = out["universes"][uni]["backtest"]
        print(f"  single-factor IC: seed={sf['seed']['IC']:+.4f} "
              f"paper-artifact={sf['paper_evolved_artifact']['IC']:+.4f} "
              f"FE-evolved={sf['fe_engine_evolved']['IC']:+.4f}")
        print(f"  model IC: baseline={md['baseline']['IC']:+.4f} augmented={md['augmented']['IC']:+.4f}")
        print(f"  backtest AR: baseline={bt['baseline']['AR']:+.3f} augmented={bt['augmented']['AR']:+.3f} "
              f"| SR {bt['baseline']['SR']:.2f}->{bt['augmented']['SR']:.2f}")

    print("\n== multi-seed robustness (single-factor, 5 data seeds) ==")
    out["robustness"] = multiseed_robustness(evolved_code, evolved_params)
    r = out["robustness"]
    print(f"  seed     IC {r['seed']['IC']['mean']:+.4f}±{r['seed']['IC']['std']:.4f} "
          f"fit {r['seed']['fitness']['mean']:+.3f}±{r['seed']['fitness']['std']:.3f}")
    print(f"  FE-evolv IC {r['fe_engine_evolved']['IC']['mean']:+.4f}±{r['fe_engine_evolved']['IC']['std']:.4f} "
          f"fit {r['fe_engine_evolved']['fitness']['mean']:+.3f}±{r['fe_engine_evolved']['fitness']['std']:.3f}")

    with open(config.OUTPUTS / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=_native)
    print(f"\n-> saved {config.OUTPUTS / 'results.json'}")


if __name__ == "__main__":
    main()
