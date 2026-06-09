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

## V5 objective
`portfolio_v4` optimizes a gross top-decile excess-return proxy with turnover/complexity
penalties. The next evolution objective should be **`portfolio_v5` = gross excess − real
turnover cost**, using the report's commission/stamp/slippage model. That makes tradeability
part of factor search instead of a post-hoc backtest surprise.
