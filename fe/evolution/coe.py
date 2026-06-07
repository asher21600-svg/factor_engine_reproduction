"""Chain of Experience (CoE) — experience-path selection (paper §4.2, Eq.2-4).

Given the current node's chain C (its root→node lineage), select n candidate
paths from the tree that jointly maximise empirical effectiveness (Eq.3) and
minimise overlap with C (coverage, Eq.2), scored by Eq.4:

    Scov(pi)  = alpha * |Φ|/|pi| + beta * |Φ|/|C|          (Φ = pi ∩ C)
    Seff(pi)  = mean_m Score(pi_m)
    Stotal(pi)= Seff(pi) - gamma * Scov(pi)

The selected paths are rendered into the structured "Program Evolution History"
context the macro agent consumes (Listing 1.2).  A `top_k` mode (pick highest
Seff, ignore overlap) is provided for the CoE-vs-top-k ablation (Table 3).
"""
from __future__ import annotations

from ..config import COE_ALPHA, COE_BETA, COE_GAMMA, COE_N_PATHS


def _node_score(n) -> float:
    s = getattr(n, "fitness", None)
    if s is None or s != s or s == float("-inf"):   # NaN / -inf guard
        return 0.0
    return float(s)


def _candidate_paths(tree, current) -> list:
    """Candidate paths = root→leaf lineages of valid leaves, excluding C itself."""
    cur_ids = {n.node_id for n in current.lineage()}
    leaves = [n for n in tree.nodes if n.valid and not any(c.valid for c in n.children)]
    paths = []
    for lf in leaves:
        lin = lf.lineage()
        ids = {n.node_id for n in lin}
        if ids == cur_ids:
            continue                      # skip the current chain itself
        paths.append(lin)
    # de-dup by id-set
    seen, uniq = set(), []
    for p in paths:
        key = frozenset(n.node_id for n in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def coverage(path, C, alpha=COE_ALPHA, beta=COE_BETA) -> float:
    cur_ids = {n.node_id for n in C}
    p_ids = {n.node_id for n in path}
    phi = len(cur_ids & p_ids)
    if not path or not C:
        return 0.0
    return alpha * (phi / len(path)) + beta * (phi / len(C))


def effectiveness(path) -> float:
    if not path:
        return 0.0
    return sum(_node_score(n) for n in path) / len(path)


def select_experience_paths(tree, current, n=COE_N_PATHS, mode="coe",
                            alpha=COE_ALPHA, beta=COE_BETA, gamma=COE_GAMMA) -> list:
    """Return up to `n` paths (each a list of ProgramNode)."""
    C = current.lineage()
    cands = _candidate_paths(tree, current)
    if not cands:
        return []
    if mode == "top_k":
        scored = [(effectiveness(p), p) for p in cands]
    else:  # 'coe'
        scored = [(effectiveness(p) - gamma * coverage(p, C, alpha, beta), p)
                  for p in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:n]]


def build_coe_context(current, paths, max_steps=4) -> str:
    """Render the selected experience paths as the 'Program Evolution History'."""
    lines = []
    for i, path in enumerate(paths, 1):
        lines.append(f"## Experience path {i} (effectiveness={effectiveness(path):+.3f})")
        steps = path[-max_steps:]
        for n in steps:
            fit = getattr(n, "fitness", float("nan"))
            idea = (n.idea or n.change_summary or "initial seed").strip()
            if len(idea) > 220:
                idea = idea[:217] + "..."
            lines.append(f"  - node#{n.node_id} fit={fit:+.3f} :: {idea}")
    if not lines:
        return "(no prior experience paths yet)"
    return "\n".join(lines)
