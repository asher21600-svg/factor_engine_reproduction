# Reproduction Plan — FactorEngine (FE)

**Paper:** *FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment* — Lin, Feng, Feng, Huang, Chen, Yang, Zhou, Fei, Liu, Li. arXiv:2603.16365v2 [cs.AI], 9 Apr 2026.

**Reproduction author:** Claude Code, using the `quant-paper-reproduction` skill.
**Date:** 2026-06-02.

---

## Phase 0 — The six questions

### 1. What is the dependent variable?
Future cross-sectional stock returns. Formally (paper §3), a factor `f` maps a lookback window of OHLCV features `X_{t-L+1:t} ∈ R^{N×L×M}` to an `l`-step-ahead per-stock predictive signal `r_{t+l} ∈ R^N`. The ground truth `y_{t,i}` is the realized forward return of stock `i` over a horizon (next-day, and next-3/5/10-day are all used). Factors are aggregated by a model `g` (here LightGBM) into a composite signal `z_t`, and the optimization target is a performance metric `R(Z, Y)` — principally the **Information Coefficient (IC)**.

### 2. What are the predictors / features?
**Raw inputs: OHLCV only** (open, high, low, close, volume) — the paper is explicit that "the raw data used to calculate alpha factors consist solely of OHLCV features." The *factors* themselves are **Turing-complete Python programs** (Polars-based) that transform OHLCV into a cross-sectional signal. Two factor sources:
- **FE-alpha**: seeded from hand-crafted Qlib Alpha factors (5- or 10-factor subsets, see below).
- **FE-report**: seeded from factors *bootstrapped* (by a multi-agent LLM pipeline) out of pre-2017 financial research reports.
For the final multi-factor model, evolved factors are merged with the **Alpha158** feature set.

### 3. What is the universe?
Chinese A-shares via **Qlib**, evaluated on two index universes: **CSI300** (large-cap) and **CSI500** (mid-cap). Trained/mined on the "full-market" dataset.

### 4. What is the period?
- **Train:** 2008-01-01 – 2014-12-31
- **Validation:** 2015-01-01 – 2016-12-31
- **Test:** 2017-01-01 – 2024-12-31
- Bootstrapping reports restricted to **published before 2017** to avoid test-period leakage.
- (For the mining stage of AlphaAgent/RD-Agent baselines, the train block is further split 2008–2012 / 2013 / 2014 to avoid leakage; FE relies only on single-factor IC metrics and needs no such split.)

### 5. What is the evaluation?
**Predictive metrics:** IC, ICIR, Rank IC (RIC), Rank ICIR (RICIR).
- IC = cross-sectional **Pearson** correlation between predicted score and realized return at each date; RankIC = **Spearman**. ICIR = mean(IC)/std(IC); RICIR = mean(RIC)/std(RIC) (Eqs. 6–7).
- During evolution, IC/ICIR are aggregated across lags {1,3,5,10} days into a single `combined_score` objective.
- Elite filtering uses fitness score **Eq. 5**: `FS = (IC*10 + ICIR + RIC*10 + RICIR)/4`.

**Portfolio metrics:** Annualized Return (AR, Eq. 8), Information Ratio (IR), Maximum Drawdown (MDD, Eq. 10), Sharpe Ratio (annualized, Eq. 14). All computed on the **excess return** series (portfolio − benchmark).

**Trading strategy (Appendix A.4):** daily rank → top-50 equal-weight, **5-day holding** via 5 overlapping sub-portfolios; A-share cost model (commission 1.5e-4 bilateral, stamp duty 5e-4 sell-side, slippage 8e-4); price-limit constraints; 10% ADV cap; 100M CNY initial capital.

### 6. What is the headline claim?
**Table 1, row `FE-report-2` (400 iterations), CSI300:**
- IC = **0.0474**, ICIR = 0.3185, RIC = 0.0475, RICIR = 0.3146
- AR = **0.1899** (18.99%), |MDD| = 12.61%, IR = 1.6001, SR = 1.0093

Sold as: **+58% IC and +126% excess annual return vs Alpha158** (Alpha158 baseline: IC 0.0299, AR 0.0840). Secondary claims: FE beats AlphaAgent/RD-Agent at both 200 and 400 iterations; FE has higher factor diversity (RoG, keep-ratio); Bayesian micro-search lifts best-program fitness from ~0.25 → ~0.38 (Fig. 5 ablation); slower alpha decay (Fig. 4).

---

## Phase 1 — Paper type

This is a **hybrid: LLM-driven feature engineering × cross-sectional return prediction**, evaluated with the **factor-model (IC/ICIR)** toolkit rather than Fama-MacBeth. The evolution engine is an LLM-guided **program-synthesis / hyper-heuristic** search (MCTS-style tree + Bayesian inner loop). Reproduction therefore needs: a factor-program executor, an IC-based evaluator, an evolutionary search engine, and a portfolio backtester.

---

## Scope & feasibility — what reproduces vs what is substituted

A *fully faithful* reproduction would require: (a) Qlib full-market A-share daily data 2008–2024, (b) a corpus of pre-2017 Chinese research reports, (c) Gemini-2.5-Pro API at the scale of 200–400 LLM-driven iterations × Bayesian inner loops on a 56-core server, and (d) the proprietary baselines (TRA, RD-Agent, AlphaAgent). That is out of reach in this environment. Following the skill's **"substitute and acknowledge"** doctrine, the reproduction is built as a faithful re-implementation of FE's *machinery* and *evaluation*, validated on anchored artifacts the paper provides verbatim:

| Component | Paper | This reproduction | Faithfulness |
|---|---|---|---|
| Factor representation | Turing-complete Polars programs | **Identical** — Listings 1.3 (seed) & 1.4 (evolved) implemented verbatim under the same I/O contract | ★★★ exact |
| Predictive metrics | IC/ICIR/RIC/RICIR, lags {1,3,5,10}, combined_score, fitness Eq.5 | **Identical formulas** | ★★★ exact |
| Evolution engine | UCT tree (Eq.1, c=√2), CoE path scoring (Eq.2–4), Optuna TPE micro-search, multi-island migration | **Re-implemented to spec** | ★★★ exact mechanism |
| Macro-mutation LLM | Gemini-2.5-Pro | LLM path implemented for **Kimi/Moonshot China (`api.moonshot.cn`), Moonshot international (`api.moonshot.ai`), Anthropic, and other OpenAI-compatible endpoints**; a deterministic/offline proposal library remains as fallback. Live runs should use `scripts/check_llm.py` and `FE_REQUIRE_LLM=1` so the output cannot masquerade as LLM-driven when the endpoint is unavailable. | ★★☆ substituted LLM (acknowledged; also a Phase-6 extension) |
| Data | Qlib A-share OHLCV, CSI300/500 | **Real Qlib if downloadable; else a synthetic OHLCV panel** with realistic price–volume dynamics + embedded cross-sectional predictability | ★☆☆ substituted data (acknowledged) |
| Multi-factor model | LGBM on evolved + Alpha158 | **LGBM** on evolved + Alpha158-style baseline factors | ★★☆ |
| Backtest | Top-50, 5-day hold, A-share costs | **Identical strategy & cost model** | ★★★ exact |
| Iteration budget | 200 / 400 | **Reduced** (≈40, matching the paper's *ablation* budget) — scale gap acknowledged | ★☆☆ reduced scale |
| Proprietary baselines (TRA, RD-Agent, AlphaAgent) | reported | **Not reproduced**; compared against paper's printed numbers + our own Alpha158/GPlearn-style baseline | — |

**What we can therefore legitimately claim to reproduce:**
1. That FE's *evolved* factor (Listing 1.4) improves predictive IC over its *seed* (Listing 1.3) — the core "evolution works" micro-claim — on identical data.
2. That Bayesian micro-search lifts evolved-program fitness vs no-Bayes (Fig. 5 ablation), under our reduced budget.
3. The full end-to-end pipeline runs and produces the paper's metric suite and a backtest with the exact trading/cost model.

**What we explicitly cannot claim:** the absolute Table-1 numbers (IC 0.0474, AR 18.99%), because they depend on the full A-share dataset and 400-iteration LLM evolution at scale. We report our analogous numbers and treat any gap as a documented limitation, not a contradiction of the paper.

---

## Build map (modules → phases)

- `fe/config.py` — universe sizes, dates, lags {1,3,5,10}, cost model, fitness params (α=β=γ=1, c=√2, FS threshold 0.4).
- `fe/data/` — `synthetic.py` (OHLCV panel + embedded alpha), `qlib_loader.py` (optional real data). → **Phase 2**
- `fe/factors/` — `contract.py` (executor), `seed.py` (Listing 1.3), `evolved.py` (Listing 1.4), `baseline.py` (Alpha158-style). → **Phase 3a**
- `fe/eval/metrics.py` — IC/RIC/ICIR/RICIR, lags, combined_score, fitness Eq.5. → **Phase 3b**
- `fe/evolution/` — `tree.py` (UCT), `coe.py` (Eq.2–4), `micro.py` (Optuna), `macro.py` (Kimi/Moonshot/OpenAI-compatible live backend + fallback), `engine.py` (4-stage loop, islands). → **Phase 3c**
- `fe/integration/` — `model.py` (LGBM), `backtest.py` (strategy + costs + portfolio metrics). → **Phase 3d**
- `fe/report/` — HTML + PDF. → **Phase 5**
- `scripts/01..06` — runnable entry points. → **Phases 4–5**

## Output checklist (from the skill)
- [x] `REPRODUCTION_PLAN.md` answers the six questions
- [ ] `outputs/<universe>_panel.parquet` exists per universe
- [ ] In-window evaluation compared to the paper's headline claim (gap investigated)
- [ ] ≥1 out-of-window / sensitivity extension run (Bayes on/off; LLM substitution; period split)
- [ ] `outputs/reproduction_report.html` self-contained
- [ ] `outputs/reproduction_report.pdf`
- [ ] Honest discussion of substitutions in the report
