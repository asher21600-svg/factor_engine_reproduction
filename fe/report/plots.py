"""Figure generation for the reproduction report (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_SEED = "#888888"
C_FE = "#1f77b4"
C_FE2 = "#d62728"
C_BENCH = "#bbbbbb"


def convergence_plot(history, out: Path, title="Evolution: best objective vs iteration"):
    if not history:
        return None
    it = [h[0] for h in history]
    rw = [h[1] for h in history]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(it, rw, color=C_FE, lw=2)
    ax.set_xlabel("evolution iteration")
    ax.set_ylabel("best objective reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def bayes_ablation_plot(abl, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for tag, color, label in [("with_bayes", C_FE, "with Bayesian micro-search"),
                              ("without_bayes", C_SEED, "without (fixed params)")]:
        h = abl.get(tag, {}).get("history", [])
        if h:
            ax.plot([x[0] for x in h], [x[1] for x in h], color=color, lw=2, label=label)
    ax.set_xlabel("evolution iteration")
    ax.set_ylabel("best objective reward")
    ax.set_title("Ablation: Bayesian micro-search (paper Fig. 5)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def equity_plot(eq_base: pd.DataFrame, eq_aug: pd.DataFrame, out: Path, uni="CSI300"):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for eq, color, label in [(eq_base, C_SEED, "baseline (Alpha-mini)"),
                             (eq_aug, C_FE, "augmented (+FE factor)")]:
        if eq is None or len(eq) == 0:
            continue
        d = pd.to_datetime(eq["datetime"])
        ax.plot(d, eq["excess"], color=color, lw=2, label=label)
    ax.axhline(1.0, color=C_BENCH, ls="--", lw=1)
    ax.set_ylabel("cumulative excess (port / benchmark)")
    ax.set_title(f"Cumulative excess return — {uni} (paper Fig. 3 analog)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def yearly_ic_plot(yic_base: dict, yic_aug: dict, out: Path, uni="CSI300"):
    years = sorted(set(int(y) for y in yic_base) | set(int(y) for y in yic_aug))
    if not years:
        return None
    x = np.arange(len(years))
    b = [yic_base.get(str(y), yic_base.get(y, {})).get("ic", np.nan) for y in years]
    a = [yic_aug.get(str(y), yic_aug.get(y, {})).get("ic", np.nan) for y in years]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x - 0.2, b, 0.4, color=C_SEED, label="baseline")
    ax.bar(x + 0.2, a, 0.4, color=C_FE, label="augmented (+FE)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=0, fontsize=8)
    ax.set_ylabel("model IC"); ax.set_title(f"Yearly IC (alpha decay) — {uni} (Fig. 4 analog)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def headline_ic_plot(results, out: Path):
    """Grouped bars: single-factor seed vs FE-evolved, per universe."""
    unis = list(results["universes"].keys())
    seed = [results["universes"][u]["single_factor"]["seed"]["IC"] for u in unis]
    eng = [results["universes"][u]["single_factor"]["fe_engine_evolved"]["IC"] for u in unis]
    x = np.arange(len(unis))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - 0.2, seed, 0.4, color=C_SEED, label="seed factor")
    ax.bar(x + 0.2, eng, 0.4, color=C_FE2, label="FE-engine evolved")
    ax.set_xticks(x); ax.set_xticklabels([u.upper() for u in unis])
    ax.set_ylabel("single-factor test IC")
    ax.set_title("Seed vs FE-evolved single-factor IC")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def diversity_plot(pool, out: Path):
    """Fitness vs evolution depth, colored by #transforms (pool exploration)."""
    pts = [(n["depth"], n["fitness"], len(n["transforms"]))
           for n in pool if n.get("valid") and n.get("fitness") is not None]
    if not pts:
        return None
    d, f, t = zip(*pts)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    sc = ax.scatter(d, f, c=t, cmap="viridis", s=28, alpha=0.8)
    ax.set_xlabel("tree depth"); ax.set_ylabel("node fitness")
    ax.set_title("Evolved pool: fitness vs depth")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("# transforms")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out
