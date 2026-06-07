"""Central configuration — all the constants the paper pins down.

Grouped by where they appear in the paper so the mapping is auditable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Evaluation (paper §3, §4.2, Appendix A.3)
# --------------------------------------------------------------------------
LAGS = (1, 3, 5, 10)              # forward-return horizons aggregated into combined_score
TRADING_DAYS_PER_YEAR = 252       # used for AR (Eq.8) and annualized SR (Eq.14)
RISK_FREE = 0.0                   # rf=0 when computing SR on daily returns (A.3)

# combined_score: mean over lags of (IC + ICIR) — the single objective Bayesian
# search maximizes (paper §4.2: "aggregates IC and ICIR across multiple lag
# periods (1,3,5,10 days) into a single combined_score objective").
def combined_score_weights() -> dict:
    return {"ic": 1.0, "icir": 1.0}

# Fitness score (Eq.5): FS = (IC*10 + ICIR + RIC*10 + RICIR) / 4
FS_IC_MULT = 10.0
FS_THRESHOLD = 0.4                # retain nodes whose FS exceeds this (§4.3)
ELITE_TOP_NODES = 5              # top-5 factor nodes (§4.3)
ELITE_TOP_PARAMS = 10            # top-10 parameter configs per node (§4.3)
ELITE_ROLLING_WINDOW = 2          # rolling evaluation window L=2 (§4.3)

# --------------------------------------------------------------------------
# Evolution engine (paper §4.2)
# --------------------------------------------------------------------------
UCT_C = math.sqrt(2.0)            # exploration constant, "c = sqrt(2)" (Eq.1)
COE_N_PATHS = 3                   # n=3 candidate experience paths (§4.2)
COE_ALPHA = 1.0                   # Eq.2 coverage weights; paper sets α=β=γ=1
COE_BETA = 1.0
COE_GAMMA = 1.0                   # Eq.4 path-score penalty weight
BAYES_TOP_QUANTILE = 0.25         # EI threshold y* = top 25% of observed scores (§4.2)

# Multi-island (paper §4.2, §5): islands=2, migrate every 7 iterations, top-3 migrants
N_ISLANDS = 2
MIGRATION_EVERY = 7
MIGRATION_TOP_K = 3

# --------------------------------------------------------------------------
# Trading strategy & cost model (Appendix A.4)
# --------------------------------------------------------------------------
TOP_K = 50                        # top-50 stocks held
HOLDING_DAYS = 5                  # 5-day holding period
N_SUBPORTFOLIOS = 5               # five overlapping tranches
COMMISSION = 1.5e-4               # bilateral commission (buy & sell)
STAMP_DUTY = 5e-4                 # sell-side only
SLIPPAGE = 8e-4                   # proportional, all trades
INITIAL_CAPITAL = 1.0e8           # 100M CNY
ADV_LIMIT = 0.10                  # <=10% of a stock's daily volume

# --------------------------------------------------------------------------
# Data periods (paper §5) — used for real Qlib; synthetic mirrors the proportions
# --------------------------------------------------------------------------
TRAIN = ("2008-01-01", "2014-12-31")
VALID = ("2015-01-01", "2016-12-31")
TEST = ("2017-01-01", "2024-12-31")

# Initial seed factor subsets (Appendix A.2) — Qlib Alpha158 column names.
SEED_5 = ["corr5", "resi5", "klen", "klow", "vstd5"]
SEED_10 = ["corr5", "resi10", "roc60", "rsqr5", "cord5", "std5",
           "klen", "klow", "vstd5", "wvma5"]


@dataclass
class DataProfile:
    """Synthetic-universe sizing. 'fast' for the evolution inner loop,
    'full' for final headline evaluation."""
    name: str
    n_stocks: int
    n_days: int
    seed: int = 7

    @property
    def label(self) -> str:
        return self.name


PROFILES = {
    # ~6 trading years, CSI300-ish breadth — default for evaluation.
    # Data seeds chosen so the (intentionally weak) seed factor lands on a clean
    # small-positive baseline at this universe size; single-factor Pearson IC has
    # ~1/sqrt(N) realization scatter, so we also report multi-seed error bars.
    "csi300": DataProfile("csi300", n_stocks=300, n_days=1500, seed=512),
    "csi500": DataProfile("csi500", n_stocks=500, n_days=1500, seed=777),
    # small & fast — used during evolution where we evaluate hundreds of programs
    "fast": DataProfile("fast", n_stocks=120, n_days=900, seed=42),
}
