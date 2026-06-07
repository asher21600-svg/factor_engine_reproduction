# Reproduction Plan — FactorEngine (FE)

**Paper:** *FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment* — Lin, Feng, Feng, Huang, Chen, Yang, Zhou, Fei, Liu, Li. arXiv:2603.16365v2 [cs.AI], 9 Apr 2026.

**Reproduction author:** Claude Code, using the `quant-paper-reproduction` skill.
**Initial plan date:** 2026-06-02.
**Current status date:** 2026-06-07.

**Status summary:** Phases 0-5 are complete on real Qlib China A-share data, and the follow-up
V3 protocol is implemented/demonstrated on the existing 300-iteration live Kimi/Moonshot run.
The current report is `outputs/reproduction_report.html` / `.pdf`.

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

## Phase status — current execution

| Phase | Purpose | Current status |
|---|---|---|
| Phase 0 | Answer the six reproduction questions | Complete |
| Phase 1 | Classify the paper and define scope | Complete |
| Phase 2 | Build/load CSI300 and CSI500 panels | Complete on real Qlib CN data; local parquet panels are generated but intentionally ignored by git due size |
| Phase 3 | Re-implement FE machinery | Complete: factor contract, metrics, UCT/CoE, Bayesian micro-search, multi-island engine, live LLM macro path, model, backtest |
| Phase 4 | Run experiments and ablations | Complete: 300-iteration live Kimi/Moonshot CSI300 run, CSI300/CSI500 evaluation, Bayes/island ablations |
| Phase 5 | Produce report artifacts | Complete: self-contained HTML and PDF report |
| Phase 6 / V3 | Harden selection/integration after overfit diagnosis | Complete/demonstrated: robust elite selection, parsimony penalty, orthogonalization, portfolio-aware scoring, prompt/parser contract, plateau-aware stopping |

## Scope & feasibility — what reproduces vs what is substituted

A bit-exact reproduction would require: (a) the paper's exact Qlib data snapshot and full-market
membership files, (b) the proprietary/pre-2017 Chinese research-report corpus used for FE-report,
(c) the same Gemini-2.5-Pro snapshot at the paper's scale on comparable hardware, and (d) the
proprietary baselines (TRA, RD-Agent, AlphaAgent). This reproduction therefore separates
**faithful machinery/data evaluation** from **non-reproducible paper dependencies**.

| Component | Paper | This reproduction | Faithfulness |
|---|---|---|---|
| Factor representation | Turing-complete Polars programs | **Identical** — Listings 1.3 (seed) & 1.4 (evolved) implemented verbatim under the same I/O contract | ★★★ exact |
| Predictive metrics | IC/ICIR/RIC/RICIR, lags {1,3,5,10}, combined_score, fitness Eq.5 | **Identical formulas** | ★★★ exact |
| Evolution engine | UCT tree (Eq.1, c=√2), CoE path scoring (Eq.2–4), Optuna TPE micro-search, multi-island migration | **Re-implemented to spec** | ★★★ exact mechanism |
| Macro-mutation LLM | Gemini-2.5-Pro | **Live Kimi/Moonshot China** (`api.moonshot.cn`, `kimi-k2.6`) plus Moonshot international, Anthropic, and OpenAI-compatible backends. The live gate uses the same call path as evolution; deterministic fallback is explicitly labeled when transient/unparseable calls occur. | ★★☆ substituted LLM, live and audited |
| Data | Qlib A-share OHLCV, CSI300/500 | **Real Qlib CN bundle** loaded through the pure NumPy `.bin` reader; CSI300/CSI500 panels generated point-in-time and evaluated on the paper window. Synthetic remains only as fallback/dev mode. | ★★☆ real data, not paper's exact snapshot |
| Multi-factor model | LGBM on evolved + Alpha158 | **LGBM** on evolved factors + Alpha158-128 features, Qlib label, and real index benchmark | ★★☆ |
| Backtest | Top-50, 5-day hold, A-share costs | **Identical strategy & cost model** | ★★★ exact |
| Iteration budget | 200 / 400 | **300 live-macro iterations**, 2 islands, 12 Bayesian trials/step, 3602 evaluations. This is between the paper's 200/400 budgets but not the exact reported run. | ★★☆ close but not bit-exact |
| Proprietary baselines (TRA, RD-Agent, AlphaAgent) | reported | **Not reproduced**; compared against paper's printed numbers + our own Alpha158/GPlearn-style baseline | — |

**What we can legitimately claim:**
1. The FE machinery is reproduced end-to-end on real Qlib A-share data with the paper's metric suite and trading/cost model.
2. A live Kimi/Moonshot macro-agent can drive the FE mutation path: the 300-iteration CSI300 run lifts validation fitness from `0.07233` to `0.63626` and reward from `0.1017` to `0.57647`.
3. The reproduction surfaces a key practical failure mode: validation-only elite selection overfits the 2015-2016 window. CSI300 augmented model IC falls from `0.01585` to `0.01166`, while the evolved single factor transfers better to CSI500 (`IC=0.01672`) than CSI300.
4. V3 closes the main diagnosis loop: robust/parsimony selection removes the CSI300 IC degradation (`0.01585` baseline to `0.01589` robust), and orthogonalized robust factors improve CSI500 model IC (`0.00939` baseline to `0.01352` orthogonalized).
5. Plateau analysis shows the best validation frontier was last improved at iteration 99/300; a patience-50 stop would preserve the best factor while saving about 50% of runtime.

**What we explicitly cannot claim:** the absolute Table-1 numbers (for example, CSI300 FE-report-2
IC 0.0474 and AR 18.99%), because they depend on the paper's exact Gemini snapshot, report corpus,
data snapshot, baseline implementations, and 400-iteration trajectory. The report treats these gaps
as documented limits rather than contradictions of the paper.

---

## Build map (modules → phases)

- `fe/config.py` — universe sizes, dates, lags {1,3,5,10}, cost model, fitness params (α=β=γ=1, c=√2, FS threshold 0.4).
- `fe/data/` — `qlib_bin.py` / `qlib_loader.py` (real Qlib `.bin` reader), `synthetic.py` (fallback/dev panel). → **Phase 2**
- `fe/factors/` — `contract.py` (executor), `seed.py` (Listing 1.3), `evolved.py` (Listing 1.4), `baseline.py` (Alpha158-style). → **Phase 3a**
- `fe/eval/metrics.py` — IC/RIC/ICIR/RICIR, lags, combined_score, fitness Eq.5. → **Phase 3b**
- `fe/evolution/` — `tree.py` (UCT), `coe.py` (Eq.2–4), `micro.py` (Optuna), `macro.py` (Kimi/Moonshot/OpenAI-compatible live backend + V3 parser/parameter contract), `engine.py` (4-stage loop, islands, plateau stop). → **Phase 3c / V3**
- `fe/integration/` — `model.py` (LGBM), `backtest.py` (strategy + costs + portfolio metrics), `robust_elite.py` (V3 robust selection, orthogonalization, portfolio-aware scoring). → **Phase 3d / V3**
- `fe/report/` — HTML + PDF. → **Phase 5**
- `scripts/01_build_data.py` — build/load CSI300/CSI500 panels. → **Phase 2**
- `scripts/02_run_evolution.py` — run the FE macro-micro evolution engine. → **Phase 4a**
- `scripts/03_ablations.py` — Bayes/island ablations. → **Phase 4c**
- `scripts/04_evaluate.py` — Alpha158/label/benchmark evaluation. → **Phase 4b**
- `scripts/05_build_report.py` — final HTML/PDF report. → **Phase 5**
- `scripts/06_robust_elite.py`, `scripts/07_orthogonal_elite.py`, `scripts/08_v3_4to6.py` — V3 protocol demonstrations. → **Phase 6**

## Output checklist (from the skill)
- [x] `REPRODUCTION_PLAN.md` answers the six questions
- [x] `outputs/<universe>_panel.parquet` exists per universe locally; intentionally git-ignored because the files are large generated data
- [x] In-window and out-of-window evaluation compared to the paper's headline claim; gap investigated in the report
- [x] Sensitivity/extension runs: Bayes on/off, 1 vs 2 islands, live LLM substitution, robust elite selection, orthogonalization, portfolio objective, parser/parameter contract, plateau stopping
- [x] `outputs/reproduction_report.html` self-contained
- [x] `outputs/reproduction_report.pdf`
- [x] Honest discussion of substitutions, overfit, and V3 fixes in the report

## Next phases, if continuing beyond this reproduction

These are extensions, not blockers for the current reproduction:

1. **Phase 7 — scale replication:** repeat the same protocol at 400+ iterations and multiple random seeds, then report confidence intervals for elite selection and portfolio metrics.
2. **Phase 8 — FE-report corpus:** add a documented, pre-2017 public research-report corpus or a licensed private corpus to reproduce the paper's FE-report branch more directly.
3. **Phase 9 — model/backtest stress tests:** evaluate alternative labels/horizons, market regimes, cost assumptions, and turnover constraints to separate factor alpha from execution sensitivity.
4. **Phase 10 — external baselines:** add open implementations or careful approximations of RD-Agent / AlphaAgent / TRA if reproducible references become available.
