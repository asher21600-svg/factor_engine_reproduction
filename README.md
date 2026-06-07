# FactorEngine (FE) — Reproduction

A faithful, runnable reproduction of the machinery in **FactorEngine: A Program-level
Knowledge-Infused Factor Mining Framework for Quantitative Investment** (arXiv:2603.16365),
built with the `quant-paper-reproduction` skill.

> **TL;DR of scope.** The paper's *engine* (program-level macro–micro co-evolution + Bayesian
> micro-search + UCT tree + multi-island) is re-implemented to spec and validated. Data (Qlib
> A-shares), the LLM backbone (Gemini-2.5-Pro), the full Alpha158 set, and the 200/400-iteration
> compute budget are **substituted and documented** — see [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md).
> We report *relative* effects (which reproduce) and are explicit about absolute gaps (which don't).

## What's implemented

| Paper component | Module |
|---|---|
| Programmatic factor I/O contract; seed (Listing 1.3) & evolved (Listing 1.4) factors | `fe/factors/` |
| IC / ICIR / RankIC / RankICIR, lags {1,3,5,10}, combined_score, fitness (Eq.5) | `fe/eval/metrics.py` |
| UCT program tree (Eq.1), Q/N backprop | `fe/evolution/tree.py` |
| Chain-of-Experience path scoring (Eq.2–4) | `fe/evolution/coe.py` |
| Bayesian micro-search (Optuna TPE, top-25% EI) | `fe/evolution/micro.py` |
| Macro mutation: deterministic transform library + live LLM backend (Kimi/Moonshot China, OpenAI-compatible, or Anthropic; Listing 1.1/1.2 prompts) | `fe/evolution/macro.py` |
| 4-stage loop, multi-island migration | `fe/evolution/engine.py` |
| Alpha158-style baseline, LightGBM multi-factor model | `fe/integration/{baseline,model}.py` |
| Trading strategy + A-share cost model + portfolio metrics (Appendix A.3/A.4) | `fe/integration/backtest.py` |
| Synthetic OHLCV panel (embedded recoverable alpha) + optional Qlib loader | `fe/data/` |
| Self-contained HTML/PDF report | `fe/report/` |

## Run it

```bash
PY=/Users/difeisu/miniconda3/bin/python   # interpreter with the deps (see requirements.txt)

$PY scripts/01_build_data.py                 # build synthetic panels -> outputs/*.parquet
$PY scripts/02_run_evolution.py              # run the FE engine      -> outputs/evolution.json
$PY scripts/03_ablations.py                  # Bayes on/off, islands  -> outputs/ablations.json
$PY scripts/04_evaluate.py                   # tables + backtest       -> outputs/results.json
$PY scripts/05_build_report.py               # report -> outputs/reproduction_report.{html,pdf}
```

To use the real Kimi/Moonshot macro path for dataset evolution:
```bash
export FE_LLM_PROVIDER=kimi-cn
export FE_LLM_BASE_URL=https://api.moonshot.cn/v1
export FE_LLM_API_KEY=sk-...          # or export MOONSHOT_API_KEY
export FE_LLM_MODEL=kimi-k2.6
export FE_LLM_TEMPERATURE=1           # Moonshot may enforce a model-specific value
unset FE_LLM_PROXY                    # direct HTTPS is preferred in China
python scripts/check_llm.py           # real call -> parse diff -> compile
FE_REQUIRE_LLM=1 python scripts/02_run_evolution.py --panel csi300 --iterations 200 --use-llm
```
Use `FE_LLM_BASE_URL=https://api.moonshot.ai/v1` for an international Moonshot key, or
`FE_LLM_PROXY=https://...` only when your network requires an explicit proxy. `FE_LLM_PROXY`
is used only by the LLM call path; set `HTTPS_PROXY` separately only if you also want `pip`
or data downloads to use a proxy. If your Python package index is misconfigured or you need a
mirror, set `FE_PIP_INDEX_URL=https://pypi.org/simple` or your preferred mirror. The run records the provider/model/base in
`outputs/evolution.json` without storing the API key.

By default, `scripts/02_run_evolution.py` now uses the V3 production path:

- `--objective portfolio_v3`: train/validation robust score with stability, turnover, and complexity penalties.
- `--elite-rule robust`: sign-consistent, parsimony-penalized elite selection instead of validation-only top-k.
- `--patience 50`: plateau-aware early stopping.

For a paper-faithful IC-only ablation, run:
```bash
python scripts/02_run_evolution.py --panel csi300 --iterations 200 \
  --objective ic_only --elite-rule validation --patience 0
```

**Real Qlib A-share data (paper-faithful)** — no `qlib` package needed (a pure-NumPy `.bin`
reader handles it); full guide in [REAL_DATA.md](REAL_DATA.md):
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python data/setup_qlib.py        # download prebuilt CN bundle (~1.5 GB) -> ~/.qlib/qlib_data/cn_data
python tests/test_data.py        # verify it loaded
python scripts/01_build_data.py --market csi300 --profiles csi300     # + csi500
python scripts/02_run_evolution.py --panel csi300 --iterations 200
python scripts/04_evaluate.py --baseline alpha158 --label label --use-benchmark
python scripts/03_ablations.py && python scripts/05_build_report.py
```
Everything falls back to the synthetic panel if no bundle is present.

## Key results (synthetic CSI300/500, ~40-iteration budget)

- The engine evolves the seed factor to **~5–7× single-factor fitness** on the mining set and higher
  out-of-sample IC, rediscovering the paper's own refinements (turnover weighting, mid-price centering,
  EWM smoothing).
- Adding the FE factor to the LightGBM model **raises test IC and every backtest metric** vs the
  Alpha-mini baseline — same direction as the paper's headline.
- Bayesian micro-search and a second island both help (ablations), matching Fig. 5 / Table 3.

See `outputs/reproduction_report.html` for the full write-up.
