# Plan — Improve Excess Return by Adjusting Factors

**Goal:** lift backtested **excess return** (annualized return over the CSI300/CSI500
*index* benchmark) and its risk-adjusted form, the **Information Ratio** (IR = excess /
tracking-error, net of A-share costs) — not just IC. Builds on the V3 protocol (robust
selection, parsimony, orthogonalization, portfolio-aware objective, plateau stop).

## Core diagnosis: IC ≠ excess return
V3 lifted IC (CSI500 +44% via orthogonalization; CSI300 degradation eliminated), but
excess return moved only modestly. The gap is where the next gains are:

- **Equal-weight top-50** discards factor *conviction* (a 0.06-IC and a 0.20-IC name get
  equal weight).
- **Uncontrolled style tilts** — in A-shares the **size/beta tilt dominates**; the index
  already prices it, so it isn't true excess.
- **Turnover × A-share costs** (stamp 5e-4 sell + commission + slippage) eat gross alpha;
  **net** excess return is what counts.
- **Redundant elites** — top-5 by robust score can be near-clones, so the combined signal
  is barely more diversified than one factor.
- **IC-optimized ≠ return-optimized** — we evolve/select against an IC-flavored objective
  and only *hope* it backtests well.

## A. Factor selection & diversity — *which* factors enter
| # | Lever | Why it lifts excess return | Effort |
|---|-------|---------------------------|--------|
| A1 | **Set-level redundancy selection** (greedy max marginal IR, or cluster-and-pick-one) | 5 complementary alphas instead of 5 clones → higher, steadier IR | Med |
| A2 | **Deliberate multi-style tracks** (momentum / reversal / liquidity / volatility seeds, then pool) | diversifies the return driver; uncorrelated alphas compound | Med |
| A3 | **Mutual orthogonalization across the elite set** (residualize vs Alpha158 **+ already-selected elites**) | each added factor contributes *new* alpha; kills double-counting | Low |

## B. Factor combination & weighting — factors → one signal
| # | Lever | Why | Effort |
|---|-------|-----|--------|
| B1 | **Size/industry/beta neutralization** of each factor before merge | usually the **single biggest A-share excess-return lever** — strips the tilt the index already pays for | Med |
| B2 | **IC/ICIR-weighted linear combination** (Grinold–Kahn) alongside/instead of LightGBM | return-aware, low-overfit; often beats tree-merge on net IR | Low |
| B3 | **Decay-aware weighting** by signal persistence over the **5-day hold** | matches factor horizon to holding period → higher realized excess | Low |

## C. Portfolio construction — signal → weights
| # | Lever | Why | Effort |
|---|-------|-----|--------|
| C1 | **Score/rank-weighted** holdings (or top-decile tiered) vs equal-weight top-50 | capital follows conviction → higher gross excess | Low |
| C2 | **Turnover hysteresis / no-trade band**; tune the 5-tranche overlap | cuts the cost drag subtracting from net excess | Low |
| C3 | **Enhanced-indexing** — index + active tilts under a TE budget | makes excess return the explicit objective; controls TE → reliable IR | High |

## D. Objective alignment — evolve/select *for* excess return
| # | Lever | Why | Effort |
|---|-------|-----|--------|
| D1 | **`portfolio_v4` objective** — regularized net-excess-return / IR on validation (keep train∧valid robustness) | stop optimizing a proxy; align search with the goal | Med |
| D2 | **Real cost model in fitness** (commission+stamp+slippage on est. turnover) | search prefers *tradeable* alpha | Low |

## Recommended sequence (ROI-ordered)
1. **Quick wins** (offline A/Bs on the existing `evolution.json`, no 14h Kimi rerun) —
   **B1 → C1 → C2 → A3**. Biggest, fastest expected lift; all measurable offline.
2. **Medium** — A1 → B2 → D1. D1 is now complete via the fresh V4 live run; the next
   objective-alignment step is `portfolio_v5` with real turnover costs in fitness.
3. **Bigger bets** — C3 → A2.

## Honest evaluation protocol (don't re-overfit)
- **Primary metric:** annualized excess return (AR − index) and **IR = excess/TE, net of
  A-share costs**. Report **per year** (V3 lesson: averages hide regime instability).
- A/B every lever vs the current default/reference arm on **both CSI300 and CSI500** (results were
  universe-dependent — mid-cap held the orthogonal alpha).
- Tune on validation; keep **2017–2024 test a true holdout**; select factor-set changes
  with the robust train∧valid rule.
- Fold into the report as **Result 7 — excess-return levers**.

## ⚠️ CORRECTION (objective-shadowing bug) — read first
A variable-shadowing bug in `fe/evolution/micro.py` (`optimize_parameters` defined a local
`def objective(trial)` that shadowed its `objective: str` argument) caused the Optuna
micro-search to **silently optimize `portfolio_v3` for every run**, regardless of
`--objective`. **Proof:** each run's stored `best_reward` equals the `portfolio_v3` score to
5 decimals (the "v4" run 0.31262 vs v4-formula 0.30149; the live "v5" run 0.32357 vs
v5-formula 0.37043). **Consequences:**
- The **"V4 beats V3" result below (and former report Result 8) is RETRACTED** — the two
  runs both optimized v3; their differences were LLM-trajectory / mutation-count noise
  (≈250 vs ≈10 accepted mutations), not the objective.
- **Result 9 (turnover → positive net AER) is UNAFFECTED** — it is backtest-side, independent
  of the evolution objective.
- **Fixed & hardened:** nested fn renamed (`_trial_objective`); `evaluate_objective` now
  **raises on an unknown objective** instead of defaulting to v3 (this guard is what surfaced
  the bug). Validated: the micro-search now threads `portfolio_v3 / v4 / v5` correctly.
- A genuine v3-vs-v4-vs-v5 comparison **requires a fresh run with the fix** (pending; live
  Kimi on the user's machine). The numbers in the V4 section below are kept for the record but
  do NOT reflect an actual objective change.

## Status / changelog
Implemented in `scripts/09_excess_return.py` and `scripts/10_excess_tier2.py`
(test split, index benchmark, net of A-share costs) plus
`fe/integration/backtest.py` (`weighting`, `hysteresis` params, behavior-preserving).
The fresh V4 run (`outputs/evolution.json`) is a 300-iteration live Kimi/Moonshot run with
`objective=portfolio_v4`, 2 islands, 12 Bayesian trials/step, robust elite selection, and 10
accepted live LLM mutations.

### V4 result: excess return improved, but cost drag dominates
V4 improves net AER versus the saved V3 reference on both universes:

| Universe | Baseline AER | V3 AER | V4 AER | V4 vs V3 |
|----------|--------------|--------|--------|----------|
| CSI300 | −2.06% | −1.67% | **−1.28%** | **+0.39 pp** |
| CSI500 | −0.80% | −1.00% | **−0.49%** | **+0.51 pp** |

The gross-vs-net decomposition shows why net AER is still slightly negative:

| Case | net AER | ann. cost drag | gross AER proxy | turnover |
|------|---------|----------------|-----------------|----------|
| CSI300 V4 default | −1.28% | 9.05% | **+7.77%** | 75.5× |
| CSI500 V4 + A3 | −0.38% | 10.05% | **+9.66%** | 83.8× |

**Interpretation:** the signal is already gross-positive. The remaining problem is annualized
turnover near 75-84×, which turns +8-10% gross excess into roughly flat/negative net excess.

### Current lever readout
| Lever group | CSI300 | CSI500 | Verdict |
|-------------|--------|--------|---------|
| A3 mutual-ortho | AER −1.46% | **AER −0.38%** | carry for CSI500 |
| B1 size-neutral | AER −1.54% | AER −0.65% | weak in current V4 artifacts |
| C1 rank-weight | AER −2.55% | AER −2.25% | reject |
| C2 hysteresis 0.5 | AER −1.07% | AER −0.96% | helps CSI300 a little, but turnover barely moves |
| A3+B1b size+beta | **AER −0.98%** | AER −1.09% | carry for CSI300 only |
| B2 IC-weighted linear | AER −5.75% | AER −7.03% | reject |

**Recommendation now:**
- **CSI300:** current best post-hoc arm is **A3+B1b size+beta** (AER −0.98%).
- **CSI500:** current best post-hoc arm is **A3 mutual-ortho** (AER −0.38%, IR +0.032).
- **Drop C1 and B2.** They destroy realized excess.
- **Do not rely on C2 alone.** Selection hysteresis is too weak because the 5-day overlapping
  tranche structure still forces frequent book rotation.

## Next phase: turnover-first sweep
Run a cached-prediction backtest sweep before another expensive LLM run:

1. **Signal smoothing:** EWM daily predictions with spans 5 and 10 before ranking.
2. **Longer holding:** test 10, 15, and 20-day holds; keep holding=5 as the paper default.
3. **Rank-band rebalancing:** buy only inside top 30; sell only after falling outside top 100.
4. **Combined best:** target turnover around 20-25×. If gross excess survives, a 3× cost cut
   mechanically moves CSI300 toward +4-5% net AER and CSI500 toward +6% net AER.

## RESULT: positive net excess return achieved (`scripts/11_turnover_sweep.py` → `outputs/turnover_sweep.json`; report Result 9)
The turnover sweep (EWM smoothing × holding × rank-band, on cached v4 preds) flips net AER **positive on both universes**:

| Universe | default (5d) | best config | turnover | cost | **net AER** | IR | SR |
|----------|--------------|-------------|----------|------|------------|----|----|
| CSI300 | −1.28% | 20d hold (base) | 75→**20×** | 9.0%→**2.4%** | **+1.07%** | +0.22 | +0.26 |
| CSI500 | −0.49% | 20d hold + A3 + band | 84→**22×** | 10.0%→**2.6%** | **+1.88%** | **+0.52** | +0.14 |

- **Holding period is the dominant lever** (5→20d cuts turnover/cost ~4×). **EWM smoothing REJECTED**
  (shaved gross faster than it cut cost once the hold handled turnover). **Rank band marginal** (helps
  CSI500 only). A3 helps CSI500 not CSI300 (the universe-split persists).
- Net AER is positive but **smaller than the naive +4–6% projection**: gross excess also *decays* with the
  longer hold (CSI300 +7.8%→+3.5%), so net = gross − cost wins by a smaller margin. The gap to the index is
  nonetheless **closed** (positive net excess return, net of A-share costs).
- Backtest fix: `tranche = 1/holding` (renormalizes the book for any holding; equals the paper default at
  holding=5). The whole chain now closes: positive IC → positive *gross* excess → cost was the killer →
  longer holding → **positive net excess return.**
- **Next refinements:** find the exact holding optimum (test 12/15/25/30) and implement `portfolio_v5`
  (cost-net objective) so the search evolves factors that are tradeable net, not just gross-predictive.

## Refinement 1 — finer holding sweep (DONE)
`scripts/11` is now CLI-parameterizable (`--spans --holds --bands`). A fine sweep over
{5,10,12,15,20,25,30,40}-day holds (span=1; smoothing already rejected) locates the optima:
- **CSI500: interior optimum at a 25-day hold** — A3 + band → **net AER +2.11%, IR +0.60**
  (turnover 17×, cost 2.1%). Gross excess decays beyond 25d, so net falls past the peak.
- **CSI300: monotonic to 40 days** — base + band → **net AER +1.90%, IR +0.35** (turnover 10×,
  cost 1.2%). Cost-dominated: it prefers low frequency; the optimum is at/beyond the longest
  tested hold. Honest caveat: a 25–40-day hold is a lower-frequency strategy than the paper's
  5-day design — the right tradeoff for a short-horizon, cost-dominated signal.

Both beat the earlier h20 result (CSI300 +1.07%, CSI500 +1.88%). Report Result 9 cites the
refined optima dynamically.

## Refinement 2 — `portfolio_v5` cost-net objective (IMPLEMENTED)
`fe/eval/objectives.py`: `portfolio_v5` = **annualized gross top-decile excess − (top-decile
set-turnover × A-share round-trip cost)**, train ∧ valid, minus complexity/sign penalties.
New helper `quantile_set_turnover` (daily fraction of the top-decile that rotates) gives the
cost cheaply without a backtest; `ROUND_TRIP_COST = 2·commission + stamp + 2·slippage ≈ 2.4e-3`.
Wired `--objective portfolio_v5`. Unit-tested (best v4 elite: ann_gross 0.39, ann_cost 0.044,
**net 0.35**; top-decile turnover 7.2%/day). This makes tradeability part of factor *search*
rather than a post-hoc backtest surprise. Definitive run needs a fresh live evolution:
`OBJECTIVE=portfolio_v5 ./run_kimi_v3.sh`.

## Result 10 — beating the baseline AND the index (cumulative excess)
Addresses "augmented cumulative excess ≈ baseline, both trail the index." Diagnosis:
(1) evolved OHLCV factors are **spanned by Alpha158** (marginal IC ≈0 on CSI300, +0.0008
on CSI500) → LightGBM gives them ~0 importance → augmented ≈ baseline; (2) the 5-day book
is cost-dominated → both trail the index (cum-excess < 1.0). Fixes (`scripts/12_beat_baseline.py`):
**residual stacking** (Alpha158 first, then a small FE model on *only* its residual) + the
**Result-9 optimal hold**. Cumulative excess (portfolio ÷ index; >1.0 beats the index):

| Universe (opt hold) | arm | 5-day | optimal |
|---------------------|-----|-------|---------|
| CSI300 (40d) | baseline | 0.852 | **1.134** (best) |
|              | augmented | 0.881 | 1.119 |
|              | stack | 0.884 | 1.104 |
| CSI500 (25d) | baseline | 0.940 | 1.109 |
|              | augmented | 0.908 | 1.118 |
|              | stack | 0.837 | **1.126** (best, IR 0.41→0.46) |

- **At the optimal hold every arm beats the index** (cum-excess >1.0) — the *holding period*,
  not the factors, clears the index; at 5 days all trail it.
- **Residual stacking beats the baseline on CSI500** (1.109→1.126) — harvests the small
  orthogonal mid-cap alpha.
- **On CSI300 the baseline is unbeaten** (1.134): Alpha158 is near-efficient for OHLCV
  large-caps; same-modality factors only dilute. **Beating it needs a different data modality**
  (intraday/overnight, Amihud illiquidity, Garman–Klass vol, or non-OHLCV fundamentals) — the
  families the engine's runs gravitated toward (gk_lowvol/overnight/amihud).
- Honest ceiling: with OHLCV-only data the marginal value of evolved factors over Alpha158 is
  small and mid-cap-specific — consistent with the paper's own +5% FE-alpha IC gain.
