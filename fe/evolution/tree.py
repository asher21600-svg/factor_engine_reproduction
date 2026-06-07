"""Program-evolution tree with UCT selection (paper §4.2, Eq.1).

Each node is a fully-specified, executable factor program (not a partial state),
so any node can be re-evaluated and any node can be selected as the parent for
the next mutation.  Node value Q(v) is the mean reward over the subtree rooted
at v (incl. itself); UCT adds an exploration bonus.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from ..config import UCT_C


_ids = itertools.count()


@dataclass
class ProgramNode:
    code: str                                   # the factor program (genome)
    params: dict = field(default_factory=dict)  # best parameters from micro-search
    param_space: dict = field(default_factory=dict)  # declared Bayesian search ranges
    parent: "ProgramNode | None" = None
    children: list = field(default_factory=list)
    node_id: int = field(default_factory=lambda: next(_ids))
    island: int = 0
    depth: int = 0

    # evaluation
    reward: float = float("-inf")   # combined_score on the validation split
    fitness: float = float("-inf")  # Eq.5 fitness at the primary lag
    metrics: object = None          # FactorMetrics
    score_components: dict = field(default_factory=dict)
    valid: bool = False

    # UCT bookkeeping
    visit_count: int = 0
    sum_reward: float = 0.0

    # provenance (for chain-of-experience)
    idea: str = ""                  # the macro idea text that produced this node
    change_summary: str = ""        # short description of the diff vs parent
    transforms: set = field(default_factory=set)  # deterministic transforms applied (lineage)
    feature_coords: tuple = ()      # diversity descriptor (e.g. uses_smoothing, code-size bucket)

    @property
    def Q(self) -> float:
        return self.sum_reward / self.visit_count if self.visit_count else float("-inf")

    def uct(self, c: float = UCT_C) -> float:
        """Eq.1: Q(v) + c*sqrt(ln N_parent / N_v)."""
        if self.visit_count == 0:
            return float("inf")  # always explore an unvisited node first
        n_parent = self.parent.visit_count if self.parent is not None else self.visit_count
        n_parent = max(n_parent, 1)
        return self.Q + c * math.sqrt(math.log(n_parent) / self.visit_count)

    def lineage(self) -> list:
        """Root→self path (list of ProgramNode)."""
        path, n = [], self
        while n is not None:
            path.append(n)
            n = n.parent
        return list(reversed(path))


class ProgramTree:
    """Holds nodes for one island; supports UCT selection and Q/N backprop."""

    def __init__(self, root: ProgramNode):
        self.root = root
        self.nodes: list[ProgramNode] = [root]

    def add_child(self, parent: ProgramNode, child: ProgramNode) -> ProgramNode:
        child.parent = parent
        child.depth = parent.depth + 1
        child.island = parent.island
        parent.children.append(child)
        self.nodes.append(child)
        return child

    def backpropagate(self, node: ProgramNode, reward: float) -> None:
        """Update Q and N along the ancestor path up to the root (paper §4.2)."""
        for a in node.lineage():
            a.visit_count += 1
            a.sum_reward += reward

    def select(self, c: float = UCT_C) -> ProgramNode:
        """Select the most promising node (argmax UCT) among valid nodes."""
        cands = [n for n in self.nodes if n.valid] or self.nodes
        return max(cands, key=lambda n: n.uct(c))

    def best(self, key="fitness") -> ProgramNode:
        valid = [n for n in self.nodes if n.valid]
        if not valid:
            return self.root
        return max(valid, key=lambda n: getattr(n, key))

    def top_k(self, k: int, key="fitness") -> list[ProgramNode]:
        valid = [n for n in self.nodes if n.valid]
        return sorted(valid, key=lambda n: getattr(n, key), reverse=True)[:k]
