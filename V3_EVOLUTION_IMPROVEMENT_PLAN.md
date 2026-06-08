# V3 Evolution Improvement Plan

**Context:** fresh Kimi/Moonshot V3 run in `outputs/evolution.json`.

## 1. Double-check summary

The fresh run is a valid V3 run:

- Panel: `csi300`
- Objective: `portfolio_v3`
- Elite rule: `robust`
- Requested budget: 300 iterations, 2 islands, 12 Bayesian trials per step
- Actual budget: early-stopped at iteration 65 because `patience=50`
- Best frontier reached: iteration 15
- Runtime: 22,318.4 seconds
- Evaluations: 782
- Accepted live LLM mutations: 11
- Valid nodes: 108
- Best reward: `+0.3260`
- Best validation fitness: `+0.6046`
- Best transforms: `turnover + volscale`

Best-factor V3 components:

- Train IC: `+0.0228`
- Validation IC: `+0.0514`
- Validation RankIC: `+0.0742`
- Worst validation-year IC: `+0.0382`
- Turnover proxy: `0.2029`
- Hidden knobs: `0`

This is a cleaner optimization profile than the previous validation-only run: the selected factor is positive on train and validation, has no hidden parameters, and pays an explicit turnover/complexity penalty.

## 2. Main findings

### Finding A — early stop worked, but the search plateaued very early

The objective improved at iterations 1, 2, 5, and 15, then no new global best appeared through iteration 65. `patience=50` saved roughly 78% of the originally requested 300-iteration budget.

Interpretation: plateau stopping is doing its job, but the macro search is still narrow. The best solution was found quickly and then repeatedly rediscovered through migration/retuning.

### Finding B — Kimi participated, but the best factor was not primarily LLM-tagged

The run accepted 11 live LLM mutations. Several strong nodes contain `llm`, including an LLM + GK-low-vol + turnover candidate with reward `+0.3236`, very close to the best `+0.3260`.

However, the top saved best is `turnover + volscale` without the `llm` tag. This means the default deterministic library still dominates the search frontier. Kimi is useful, but not yet consistently steering the winning branch.

### Finding C — robust elites transfer much better to CSI500 than CSI300

Single-factor test check for the five selected robust elites:

| elite | CSI300 IC | CSI300 RankIC | CSI500 IC | CSI500 RankIC |
|---|---:|---:|---:|---:|
| node_27 | -0.00085 | +0.00567 | +0.01505 | +0.03535 |
| node_48 | +0.01348 | +0.04920 | +0.02171 | +0.06313 |
| node_56 | +0.01348 | +0.04920 | +0.02171 | +0.06313 |
| node_92 | +0.00110 | +0.00595 | +0.01736 | +0.03548 |
| node_34 | +0.00061 | +0.00082 | +0.01792 | +0.02956 |

Interpretation: the V3 objective found real mid-cap transfer signal. CSI500 is now the stronger target. CSI300 still needs either a different objective, different universe weighting, or a separate CSI300-specific branch.

### Finding D — elite selection still allows near-duplicate behavior

`node_48` and `node_56` have different code hashes/params but identical CSI300/CSI500 test metrics in the quick transfer check. This suggests functionally similar factor output, not necessarily exact code duplication.

Interpretation: robust selection needs an output-correlation or rank-correlation diversity penalty, not only code-level deduplication.

### Finding E — full V3 evaluation is now a compute bottleneck

The full evaluation command with Alpha158, orthogonalized FE factors, and date-demeaned target was stopped after it ran for more than 12 minutes without completing. The process was active, not hung. The likely bottleneck is per-date orthogonalization against 128 Alpha158 features across large real panels.

Interpretation: before another full report rebuild, evaluation should be optimized/cached. Otherwise every experiment becomes too slow to iterate.

## 3. Improvement plan

### Step 1 — add a fast V3 diagnostics script

Create `scripts/09_diagnose_v3_evolution.py` that reads `outputs/evolution.json` and writes `outputs/v3_evolution_diagnostics.json`.

It should report:

- improvement milestones and plateau iteration
- best objective components
- LLM node count and best LLM-node reward
- top transform families
- elite train/valid/test metrics
- CSI300 and CSI500 single-factor test transfer
- pairwise elite output rank correlations
- duplicate/near-duplicate elite warnings

This gives a cheap check before running LightGBM/backtest.

### Step 2 — add output-correlation diversity to robust elite selection

Update `select_robust()` so it greedily selects candidates by robust score but rejects a candidate if its factor output is too correlated with already selected elites.

Recommended rule:

- compute validation split cross-sectional rank signal per date
- reject if mean pairwise rank correlation with selected elites is above `0.90`
- if fewer than `k` survive, relax to `0.95`

Goal: avoid selecting multiple versions of the same turnover/volscale factor.

### Step 3 — make the objective multi-universe aware

The current evolution objective scores only the mining panel (`csi300`). The quick check shows the factors transfer much better to `csi500`.

Add an optional secondary panel objective:

```text
score = 0.65 * csi300_portfolio_v3 + 0.35 * csi500_sample_portfolio_v3
```

Use a sampled or downsampled CSI500 panel to keep runtime manageable. This should reduce CSI300 overfit while preserving the mid-cap signal.

### Step 4 — optimize orthogonalized evaluation

The current orthogonalization is too slow for rapid iteration. Improve it before relying on full report rebuilds.

Options:

- cache Alpha158 feature matrices to `outputs/cache/alpha158_<universe>.parquet`
- cache raw FE factor outputs by code hash and params hash
- orthogonalize against top Alpha158 principal components instead of all 128 raw columns
- use ridge regression with precomputed per-date matrices
- parallelize per-date residualization
- add `--fast-orthogonal` for diagnostics and keep exact orthogonalization for final report

Target: full `scripts/04_evaluate.py` should finish in minutes, not tens of minutes.

### Step 5 — split CSI300 and CSI500 conclusions

Do not force one conclusion across both universes.

Current evidence:

- CSI500: V3 factors look materially useful at the single-factor level.
- CSI300: only one elite has meaningful test IC; others are near zero.

The report should present this as:

```text
V3 finds stronger transferable OHLCV alpha in CSI500 than CSI300.
CSI300 needs stricter selection or a separate large-cap-specific objective.
```

### Step 6 — improve Kimi macro guidance

The prompt already asks for low-turnover/stable/orthogonal factors. Next, feed it the new diagnostics:

- current best turnover
- current worst-year IC
- whether the branch is redundant with selected elites
- whether the last LLM mutation improved robust score
- CSI300 vs CSI500 transfer gap

Ask Kimi to propose mutations that specifically address the weakest diagnostic, instead of general factor improvement.

### Step 7 — rerun only after the diagnostics/evaluation loop is faster

Recommended next experiment sequence:

1. Implement diagnostics + elite diversity penalty.
2. Optimize evaluation caching/orthogonalization.
3. Run a short live Kimi V3 test:

```bash
ITERS=80 PATIENCE=35 ./run_kimi_v3.sh
```

4. Run fast diagnostics.
5. If elite diversity and CSI300/CSI500 transfer improve, run the full budget:

```bash
ITERS=300 PATIENCE=50 ./run_kimi_v3.sh
```

6. Run full evaluation/report only after the fast diagnostics pass.

## 4. Success criteria for the next version

The next evolution run should be considered better only if it satisfies all of:

- no duplicate/near-duplicate elites
- at least 3/5 elites have positive CSI300 test IC
- at least 4/5 elites have positive CSI500 test IC
- augmented model IC is not below the Alpha158 baseline on CSI300
- augmented model IC improves over Alpha158 on CSI500
- annualized excess return or Sharpe improves after costs on at least one universe
- full evaluation finishes within an acceptable runtime budget

## 5. Immediate recommendation

Do not rerun another 300-iteration Kimi job yet. The current fresh run already showed the next bottlenecks:

1. elite diversity,
2. CSI300 transfer,
3. slow orthogonalized evaluation,
4. weak diagnostic feedback to Kimi.

Fix those first, then rerun.
