"""The macro-micro co-evolution engine (paper §4.2).

Each iteration runs the four-stage pipeline on one island:
  1. Program Selection   — UCT over the island tree (Eq.1)
  2. Idea Generation     — macro mutation (deterministic library or live LLM),
                           conditioned on a chain-of-experience context
  3. Implementation      — micro Bayesian (Optuna TPE) parameter search,
                           maximizing combined_score on the validation split
  4. Analysis/Feedback   — instantiate child, backpropagate reward (Q,N)

Multiple islands evolve concurrently; every `migration_every` iterations each
island's top-k programs migrate to the others (paper's multi-island design).
Selection uses ONLY train/validation reward — test metrics are never consulted
during search (recorded separately for honest out-of-sample reporting).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import (UCT_C, N_ISLANDS, MIGRATION_EVERY, MIGRATION_TOP_K, LAGS)
from ..eval import evaluate_objective
from ..factors.contract import score_factor, FactorRunError
from .tree import ProgramNode, ProgramTree
from .micro import optimize_parameters
from .macro import (deterministic_mutation, llm_mutation, BASE_PARAM_SPACE)
from .coe import select_experience_paths, build_coe_context


@dataclass
class EvolutionConfig:
    iterations: int = 40
    n_islands: int = N_ISLANDS
    migration_every: int = MIGRATION_EVERY
    migration_top_k: int = MIGRATION_TOP_K
    micro_trials: int = 18
    use_bayes: bool = True            # False => fixed/default params (ablation)
    use_llm: bool = False             # True => configured LLM macro (deterministic fallback)
    llm_model: str | None = None
    coe_mode: str = "coe"             # 'coe' or 'top_k' (ablation)
    split: str = "valid"
    lags: tuple = LAGS
    primary_lag: int = 5
    seed: int = 0
    verbose: bool = True
    patience: int = 0                 # V3 #5: early-stop after N stagnant iters (0=off)
    objective: str = "portfolio_v3"   # V3 default; use 'ic_only' for paper-faithful ablations


@dataclass
class EvolutionResult:
    islands: list
    best: ProgramNode
    history: list = field(default_factory=list)   # (iter, best_valid_reward)
    elapsed: float = 0.0
    n_evals: int = 0
    n_llm: int = 0

    def all_nodes(self):
        return [n for tree in self.islands for n in tree.nodes]


class EvolutionEngine:
    def __init__(self, seed_code: str, panel, cfg: EvolutionConfig,
                 seed_param_space: dict | None = None):
        self.cfg = cfg
        self.panel = panel
        self.rng = np.random.default_rng(cfg.seed)
        self.seed_code = seed_code
        self.seed_param_space = dict(seed_param_space or BASE_PARAM_SPACE)
        self._panel_desc = self._describe_panel(panel)
        self.islands: list[ProgramTree] = []
        self._n_evals = 0
        self._n_llm = 0
        self._llm_fails = 0          # consecutive LLM failures
        self._llm_disabled = False   # auto-disabled after repeated failures

    # -- reward = configured train/validation objective --------------------
    def _evaluate(self, code, params):
        try:
            obj = evaluate_objective(score_factor(code, self.panel, params), self.panel,
                                     code=code, params=params, objective=self.cfg.objective,
                                     lags=self.cfg.lags, primary_lag=self.cfg.primary_lag,
                                     split=self.cfg.split)
            self._n_evals += 1
            return obj
        except FactorRunError:
            self._n_evals += 1
            return None

    @staticmethod
    def _describe_panel(panel) -> dict:
        """Small, serializable dataset context for the macro agent."""
        desc = {"rows": int(getattr(panel, "shape", [0])[0])}
        for col in ("instrument", "datetime", "split"):
            if col in getattr(panel, "columns", []):
                try:
                    desc[f"n_{col}"] = int(panel[col].nunique())
                except Exception:  # noqa: BLE001
                    pass
        if "datetime" in getattr(panel, "columns", []):
            try:
                dt = panel["datetime"]
                desc["date_start"] = str(dt.min())[:10]
                desc["date_end"] = str(dt.max())[:10]
            except Exception:  # noqa: BLE001
                pass
        if "split" in getattr(panel, "columns", []):
            try:
                desc["splits"] = {str(k): int(v) for k, v in
                                  panel.groupby("split")["datetime"].nunique().items()}
            except Exception:  # noqa: BLE001
                pass
        return desc

    def _make_root(self, island: int) -> ProgramNode:
        # honest baseline: the seed at DEFAULT parameters
        m = self._evaluate(self.seed_code, {})
        root = ProgramNode(code=self.seed_code, params={}, param_space=self.seed_param_space,
                           island=island, idea="initial seed program")
        if m is not None:
            root.metrics = m.metrics
            root.score_components = m.components
            root.reward = m.score
            root.fitness = m.metrics.fitness
            root.valid = np.isfinite(m.score)
        root.visit_count = 1
        root.sum_reward = root.reward if np.isfinite(root.reward) else 0.0
        return root

    def _micro(self, code, param_space):
        if self.cfg.use_bayes and param_space:
            return optimize_parameters(
                code, param_space, self.panel, split=self.cfg.split,
                lags=self.cfg.lags, primary_lag=self.cfg.primary_lag,
                n_trials=self.cfg.micro_trials, seed=int(self.rng.integers(1 << 30)),
                objective=self.cfg.objective)
        # ablation w/o Bayes: evaluate at default params only
        obj = self._evaluate(code, {})
        from .micro import MicroResult
        score = obj.score if obj else float("-inf")
        metrics = obj.metrics if obj else None
        components = obj.components if obj else None
        return MicroResult({}, score, metrics, 1, [score], components)

    def _propose_llm(self, tree: ProgramTree, node: ProgramNode):
        """One LLM idea-generation attempt. Tracks failures and auto-disables the
        LLM backend after repeated misses so an unreachable endpoint can't stall
        the whole run (each step then proceeds on the deterministic library)."""
        if not self.cfg.use_llm or self._llm_disabled:
            return None
        paths = select_experience_paths(tree, node, mode=self.cfg.coe_mode)
        coe = build_coe_context(node, paths)
        mdesc = {
            "headline_metrics": node.metrics.headline() if node.metrics else {},
            "dataset": self._panel_desc,
        }
        mut = llm_mutation(node, tree.root.code, coe,
                           metrics_desc=str(mdesc),
                           model=self.cfg.llm_model)
        if mut is None or getattr(mut, "tag", "") == "llm_error":
            endpoint_error = mut is not None and getattr(mut, "tag", "") == "llm_error"
            if endpoint_error:
                self._llm_fails += 1
                print(f"  [LLM error: {mut.idea}]")
            # fail-loud mode: require the live backend to drive at least one
            # accepted mutation. After that, transient endpoint drops fall back
            # for the current step instead of discarding a long Kimi-driven run.
            require_llm = os.environ.get("FE_REQUIRE_LLM") in ("1", "true", "True")
            require_every = os.environ.get("FE_REQUIRE_LLM_EVERY_CALL") in ("1", "true", "True")
            gate_passed = os.environ.get("FE_LLM_GATE_PASSED") in ("1", "true", "True")
            if require_llm and (((self._n_llm == 0) and not gate_passed) or require_every):
                raise RuntimeError(
                    "FE_REQUIRE_LLM is set but the LLM call failed/returned nothing "
                    f"({getattr(mut, 'idea', 'no response')}). Check FE_LLM_PROVIDER / "
                    "FE_LLM_BASE_URL / FE_LLM_API_KEY or MOONSHOT_API_KEY / "
                    "FE_LLM_MODEL / FE_LLM_PROXY and reachability.")
            if require_llm and (self._n_llm > 0 or gate_passed):
                print("  [LLM transient/unparseable response after live gate -> deterministic fallback for this step]")
            disable_after_failures = os.environ.get("FE_LLM_DISABLE_AFTER_FAILURES", "")
            try:
                disable_after = int(disable_after_failures) if disable_after_failures else 0
            except ValueError:
                disable_after = 0
            if endpoint_error and disable_after and self._llm_fails >= disable_after and not self._llm_disabled:
                self._llm_disabled = True
                print("  [LLM endpoint failed repeatedly -> using the deterministic "
                      "transform library for the rest of this run]")
            return None
        if getattr(mut, "tag", "") == "llm":
            print(f"  [LLM idea] {getattr(mut, 'idea', '')[:100]}")
        self._n_llm += 1
        self._llm_fails = 0
        return mut

    def _step(self, tree: ProgramTree, it: int):
        # 1. Program Selection (UCT)
        cands = sorted(tree.nodes, key=lambda n: n.uct(UCT_C), reverse=True)
        node, mut = None, None
        # at most ONE LLM call per step, on the most promising node
        m = self._propose_llm(tree, cands[0])
        if m is not None and m.code != cands[0].code:
            node, mut = cands[0], m
        else:
            # deterministic macro over the top candidates with an applicable edit
            for cand in cands[:6]:
                dm = deterministic_mutation(cand, self.rng)
                if dm is not None and dm.code != cand.code:
                    node, mut = cand, dm
                    break
        if mut is None:
            # deterministic transforms exhausted: re-tune the island best
            node = tree.best("reward")
            mut = type("M", (), {"code": node.code, "param_space": {}, "idea": "re-tune params",
                                 "summary": "micro re-tune", "tag": "retune"})()

        # 2/3. Implementation: micro Bayesian search over combined param space
        pspace = dict(node.param_space)
        pspace.update(getattr(mut, "param_space", {}) or {})
        res = self._micro(mut.code, pspace)
        self._n_evals += getattr(res, "n_trials", 1)

        # 4. Analysis / feedback
        child = ProgramNode(
            code=mut.code, params=res.best_params, param_space=pspace,
            idea=getattr(mut, "idea", ""), change_summary=getattr(mut, "summary", ""),
            transforms=set(node.transforms) | {getattr(mut, "tag", "")},
        )
        child.metrics = res.best_metrics
        child.score_components = res.best_components or {}
        child.reward = res.best_score
        child.fitness = res.best_metrics.fitness if res.best_metrics else float("-inf")
        child.valid = np.isfinite(child.reward)
        child.feature_coords = tuple(sorted(t for t in child.transforms if t))
        tree.add_child(node, child)
        tree.backpropagate(child, child.reward if child.valid else node.reward)
        return child

    def run(self) -> EvolutionResult:
        cfg = self.cfg
        t0 = time.time()
        self.islands = [ProgramTree(self._make_root(i)) for i in range(cfg.n_islands)]
        history = []
        best = max((tr.root for tr in self.islands), key=lambda n: n.reward)
        last_improve_it = 0          # V3 #5: track plateau for early stopping

        for it in range(1, cfg.iterations + 1):
            isl = self.islands[(it - 1) % cfg.n_islands]
            child = self._step(isl, it)
            # track global best by validation reward
            cur_best = max((n for tr in self.islands for n in tr.nodes if n.valid),
                           key=lambda n: n.reward, default=best)
            if cur_best.reward > best.reward + 1e-9:
                best = cur_best
                last_improve_it = it
            history.append((it, best.reward))

            if cfg.verbose and (it % max(1, cfg.iterations // 10) == 0 or it == 1):
                ch_fit = child.fitness if np.isfinite(child.fitness) else float('nan')
                print(f"  iter {it:3d}/{cfg.iterations} | island {isl.root.island} | "
                      f"child[{','.join(sorted(child.transforms - {''})) or 'seed'}] "
                      f"reward={child.reward:+.3f} fit={ch_fit:+.3f} | "
                      f"GLOBAL best reward={best.reward:+.3f}")

            # migration between islands
            if cfg.n_islands > 1 and it % cfg.migration_every == 0:
                self._migrate()

            # V3 #5: plateau-aware early stop — quit once the frontier stalls
            if cfg.patience and (it - last_improve_it) >= cfg.patience:
                if cfg.verbose:
                    print(f"  [early stop] no global-best improvement for {cfg.patience} "
                          f"iters (frontier last improved at iter {last_improve_it}); "
                          f"stopping at {it}/{cfg.iterations}.")
                break

        return EvolutionResult(self.islands, best, history,
                               elapsed=time.time() - t0,
                               n_evals=self._n_evals, n_llm=self._n_llm)

    def _migrate(self):
        """Each island sends its top-k to every other island as root children."""
        snapshots = [tr.top_k(self.cfg.migration_top_k, key="reward") for tr in self.islands]
        for j, tree in enumerate(self.islands):
            for k, src in enumerate(self.islands):
                if k == j:
                    continue
                for mig in snapshots[k]:
                    clone = ProgramNode(
                        code=mig.code, params=dict(mig.params), param_space=dict(mig.param_space),
                        reward=mig.reward, fitness=mig.fitness, metrics=mig.metrics,
                        score_components=dict(mig.score_components),
                        valid=mig.valid, idea=f"[migrant from island {k}] {mig.idea}",
                        change_summary="migration", transforms=set(mig.transforms),
                        feature_coords=mig.feature_coords,
                    )
                    clone.visit_count = 1
                    clone.sum_reward = mig.reward if mig.valid else 0.0
                    tree.add_child(tree.root, clone)
