"""Assemble a self-contained HTML reproduction report from the saved JSON results.

Figures are embedded as base64 so the .html is shareable as a single file.
"""
from __future__ import annotations

import base64
import html as html_lib
import json
import re
from pathlib import Path

import pandas as pd

from .. import config
from . import plots


def _b64(path: Path) -> str:
    if path is None or not Path(path).exists():
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;height:auto;"/>'


def _pct(x):
    try:
        return f"{100*float(x):.2f}%"
    except Exception:
        return "—"


def _num(x, n=4):
    try:
        return f"{float(x):+.{n}f}"
    except Exception:
        return "—"


def _table(headers, rows, cls="t"):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def _inline_md(text: str) -> str:
    text = html_lib.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _is_table_sep(line: str) -> bool:
    chars = set(line.strip().replace("|", "").replace(" ", ""))
    return bool(chars) and chars <= {"-", ":"}


def _md_table(rows):
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed.append([_inline_md(c) for c in cells])
    if len(parsed) > 1 and _is_table_sep(rows[1]):
        return _table(parsed[0], parsed[2:], cls="t plan-table")
    return _table(parsed[0], parsed[1:], cls="t plan-table")


def _basic_markdown(md: str) -> str:
    """Small Markdown renderer for embedding REPRODUCTION_PLAN.md without adding
    a runtime dependency. It supports the subset used by the plan file."""
    out, para, list_type = [], [], None
    lines = md.splitlines()

    def flush_para():
        if para:
            out.append("<p>" + _inline_md(" ".join(p.strip() for p in para)) + "</p>")
            para.clear()

    def flush_list():
        nonlocal list_type
        if list_type:
            out.append(f"</{list_type}>")
            list_type = None

    def start_list(kind: str):
        nonlocal list_type
        if list_type != kind:
            flush_para()
            flush_list()
            out.append(f"<{kind}>")
            list_type = kind

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            flush_list()
            i += 1
            continue
        if stripped == "---":
            flush_para()
            flush_list()
            i += 1
            continue
        if stripped.startswith("```"):
            flush_para()
            flush_list()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines):
                i += 1
            out.append("<pre><code>" + html_lib.escape("\n".join(code_lines)) + "</code></pre>")
            continue
        if stripped.startswith("|"):
            flush_para()
            flush_list()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            out.append(_md_table(table_lines))
            continue
        if stripped.startswith("# "):
            flush_para()
            flush_list()
            out.append(f"<h3>{_inline_md(stripped[2:])}</h3>")
        elif stripped.startswith("## "):
            flush_para()
            flush_list()
            out.append(f"<h3>{_inline_md(stripped[3:])}</h3>")
        elif stripped.startswith("### "):
            flush_para()
            flush_list()
            out.append(f"<h4>{_inline_md(stripped[4:])}</h4>")
        elif stripped.startswith("- [x] "):
            start_list("ul")
            out.append(f"<li><b>[x]</b> {_inline_md(stripped[6:])}</li>")
        elif stripped.startswith("- [ ] "):
            start_list("ul")
            out.append(f"<li>[ ] {_inline_md(stripped[6:])}</li>")
        elif stripped.startswith("- "):
            start_list("ul")
            out.append(f"<li>{_inline_md(stripped[2:])}</li>")
        else:
            m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            if m:
                start_list("ol")
                out.append(f"<li>{_inline_md(m.group(2))}</li>")
            else:
                flush_list()
                para.append(line)
        i += 1
    flush_para()
    flush_list()
    return "\n".join(out)


def _plan_appendix() -> str:
    plan_path = Path(__file__).resolve().parents[2] / "REPRODUCTION_PLAN.md"
    if not plan_path.exists():
        return '<p class="small">REPRODUCTION_PLAN.md was not found when this report was built.</p>'
    return _basic_markdown(plan_path.read_text())


def _param_contract(evo):
    """Scan the evolved factor's code for parameters.get() keys and compare to the
    declared/tuned set — undeclared keys are hardcoded defaults that never entered
    the Bayesian micro-search (an under-optimization the LLM introduced)."""
    import re
    code = evo.get("best_code", "") or ""
    used = sorted(set(re.findall(r'parameters\.get\(\s*["\']([^"\']+)', code)))
    declared = sorted((evo.get("best_params") or {}).keys())
    undeclared = [u for u in used if u not in declared and u != "epsilon"]
    return used, declared, undeclared


def _frontier_milestones(evo):
    """From history [[iter, best_reward], ...] return (milestones, plateau_iter, n_iters):
    milestones = iterations where the global-best reward improved."""
    h = evo.get("history") or []
    best = float("-inf")
    milestones = []
    for rec in h:
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue
        it, rw = rec[0], float(rec[1])
        if rw > best + 1e-9:
            best = rw
            milestones.append((it, rw))
    plateau_iter = milestones[-1][0] if milestones else None
    return milestones, plateau_iter, (h[-1][0] if h else 0)


def _yearly_overfit(U):
    """Per-universe augmented-vs-baseline yearly IC: (better, worse, total, worst_year, worst_delta)."""
    yb = (U.get("yearly_ic") or {}).get("baseline", {})
    ya = (U.get("yearly_ic") or {}).get("augmented", {})

    def _ic(v):
        return v.get("ic") if isinstance(v, dict) else v

    years = sorted(set(yb) & set(ya))
    better = worse = 0
    worst_year, worst_delta = None, 0.0
    for y in years:
        a, b = _ic(ya[y]), _ic(yb[y])
        if a is None or b is None:
            continue
        d = a - b
        better += d > 0
        worse += d < 0
        if d < worst_delta:
            worst_delta, worst_year = d, y
    return better, worse, len(years), worst_year, worst_delta


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;
margin:0 auto;padding:28px;color:#1a1a1a;line-height:1.5;}
h1{font-size:26px;margin-bottom:2px}h2{font-size:20px;margin-top:34px;border-bottom:2px solid #eee;padding-bottom:5px}
h3{font-size:16px;margin-top:22px;color:#333}
.sub{color:#666;font-size:14px;margin-top:0}
.hero{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
.card{flex:1;min-width:150px;background:#f7f9fc;border:1px solid #e3e8f0;border-radius:10px;padding:14px}
.card .v{font-size:24px;font-weight:700;color:#1f77b4}.card .l{font-size:12px;color:#667;margin-top:3px}
.box{background:#fff8e6;border:1px solid #f0e0a8;border-radius:8px;padding:12px 16px;margin:14px 0;font-size:14px}
.box.warn{background:#fdeeee;border-color:#f0b8b8}
.box.ok{background:#eef8ee;border-color:#b8e0b8}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #e2e2e2;padding:6px 9px;text-align:right}
th{background:#f2f4f8;text-align:center}td:first-child,th:first-child{text-align:left}
.best{background:#eef6ff;font-weight:600}
.fig{margin:14px 0;text-align:center}
code{background:#f3f3f3;padding:1px 5px;border-radius:4px;font-size:12px}
.small{font-size:12px;color:#777}
.plan-appendix{background:#fbfcff;border:1px solid #e3e8f0;border-radius:8px;padding:14px 16px;margin-top:12px}
.plan-appendix h3{font-size:17px;margin-top:22px}.plan-appendix h4{font-size:14px;margin:18px 0 6px;color:#444}
.plan-appendix p,.plan-appendix li{font-size:13px}.plan-appendix ul,.plan-appendix ol{padding-left:22px}
.plan-table{font-size:12px}.plan-table td,.plan-table th{text-align:left;vertical-align:top}
pre{background:#f3f3f3;border-radius:6px;padding:10px;overflow:auto;font-size:12px}
pre code{background:transparent;padding:0}
"""


def build_html(outputs: Path = None) -> Path:
    outputs = Path(outputs or config.OUTPUTS)
    figs = outputs / "figures"
    figs.mkdir(exist_ok=True)
    results = json.loads((outputs / "results.json").read_text())
    evo = json.loads((outputs / "evolution.json").read_text())
    abl = {}
    if (outputs / "ablations.json").exists():
        abl = json.loads((outputs / "ablations.json").read_text())
    robust = {}
    if (outputs / "robust_elite.json").exists():
        robust = json.loads((outputs / "robust_elite.json").read_text())
    ortho = {}
    if (outputs / "orthogonal_elite.json").exists():
        ortho = json.loads((outputs / "orthogonal_elite.json").read_text())
    v3_456 = {}
    if (outputs / "v3_4to6.json").exists():
        v3_456 = json.loads((outputs / "v3_4to6.json").read_text())
    ex = {}
    if (outputs / "excess_return.json").exists():
        ex = json.loads((outputs / "excess_return.json").read_text())
    ext2 = {}
    if (outputs / "excess_tier2.json").exists():
        ext2 = json.loads((outputs / "excess_tier2.json").read_text())
    results_v3 = {}            # portfolio_v3-evolved reference (backed up before the v4 refresh)
    if (outputs / "results_v3.json").exists():
        results_v3 = json.loads((outputs / "results_v3.json").read_text())
    sweep = {}
    if (outputs / "turnover_sweep.json").exists():
        sweep = json.loads((outputs / "turnover_sweep.json").read_text())
    beat = {}
    if (outputs / "beat_baseline.json").exists():
        beat = json.loads((outputs / "beat_baseline.json").read_text())
    plan_appendix = _plan_appendix()

    # ---- figures ----
    f_conv = plots.convergence_plot(evo.get("history", []), figs / "convergence.png")
    f_div = plots.diversity_plot(evo.get("pool", []), figs / "diversity.png")
    f_head = plots.headline_ic_plot(results, figs / "headline_ic.png")
    f_bayes = plots.bayes_ablation_plot(abl["bayes_ablation"], figs / "bayes.png") \
        if abl.get("bayes_ablation") else None

    eq_figs, yic_figs = {}, {}
    for uni in results["universes"]:
        eb = pd.read_csv(outputs / f"equity_{uni}_baseline.csv") if (outputs / f"equity_{uni}_baseline.csv").exists() else None
        ea = pd.read_csv(outputs / f"equity_{uni}_augmented.csv") if (outputs / f"equity_{uni}_augmented.csv").exists() else None
        eq_figs[uni] = plots.equity_plot(eb, ea, figs / f"equity_{uni}.png", uni.upper())
        yb = results["universes"][uni]["yearly_ic"].get("baseline", {})
        ya = results["universes"][uni]["yearly_ic"].get("augmented", {})
        yic_figs[uni] = plots.yearly_ic_plot(yb, ya, figs / f"yic_{uni}.png", uni.upper())

    # ---- data-source-aware framing ----
    rc = results.get("run_config", {})
    real = rc.get("data_source") == "real"
    DATASRC = ("real CSI300/CSI500 A-share data (Qlib <code>.bin</code> bundle, 2008–2024)"
               if real else "a synthetic OHLCV panel (no Qlib A-share access)")

    # ---- hero stats (csi300) ----
    u3 = results["universes"]["csi300"]
    u5 = results["universes"].get("csi500", u3)
    sf = u3["single_factor"]
    seed_fit = results["evolution_summary"]["seed_fitness"]
    best_fit = results["evolution_summary"]["best_fitness"]
    md = u3["model"]
    md5 = u5["model"]
    paper3 = u3.get("paper", {}).get("FE-alpha-2", {})

    def _ratio(a, b):
        try:
            return f"{abs(a)/abs(b):.1f}×" if b else "—"
        except Exception:
            return "—"

    # ---- live-LLM + overfitting detection (data-driven narrative) ----
    try:
        _evo = json.load(open(config.OUTPUTS / "evolution.json"))
        n_llm = int(_evo.get("n_llm") or 0)
    except Exception:  # noqa: BLE001
        n_llm = 0
    transforms_list = results["evolution_summary"].get("best_transforms", [])
    is_llm_run = n_llm > 0 or "llm" in transforms_list
    fe_d3 = md["augmented"]["IC"] - md["baseline"]["IC"]      # OOS model-IC change, CSI300
    fe_d5 = md5["augmented"]["IC"] - md5["baseline"]["IC"]    # OOS model-IC change, CSI500
    fe_helps = fe_d3 > 0 and fe_d5 > 0
    # high in-sample lift but it does not transfer -> overfitting
    overfit = (best_fit / seed_fit > 2.0) and not fe_helps
    sf5_oos = u5["single_factor"]["fe_engine_evolved"]["IC"]
    agent = "live Kimi/Moonshot" if is_llm_run else "the deterministic + Claude-reasoned"

    def _delta(x):
        return f"{x:+.4f}"

    hero = f"""
    <div class="hero">
      <div class="card"><div class="v">{best_fit/seed_fit:.1f}×</div><div class="l">in-sample (validation) fitness lift, seed→evolved</div></div>
      <div class="card"><div class="v">{n_llm if is_llm_run else len(transforms_list)}</div><div class="l">{'live Kimi macro-mutations applied (n_llm)' if is_llm_run else 'macro transforms in best factor'}</div></div>
      <div class="card"><div class="v">{_delta(fe_d3)}</div><div class="l">CSI300 <b>out-of-sample</b> model-IC change from FE elite (baseline {_num(md['baseline']['IC'])})</div></div>
      <div class="card"><div class="v">{_num(sf5_oos)}</div><div class="l">CSI500 FE-evolved single-factor OOS IC (standalone signal)</div></div>
    </div>"""

    # ---- single-factor table ----
    sft_rows = []
    labels = {"seed": "Seed (Listing 1.3)", "paper_evolved_artifact": "Paper evolved artifact (Listing 1.4)",
              "fe_engine_evolved": "FE-engine evolved (this repro)"}
    for uni in results["universes"]:
        s = results["universes"][uni]["single_factor"]
        for k, lab in labels.items():
            r = s[k]
            cls = ' class="best"' if k == "fe_engine_evolved" else ""
            sft_rows.append((f'<span{cls}>{uni.upper()} · {lab}</span>', _num(r["IC"]), _num(r["ICIR"], 3),
                             _num(r["RIC"]), _num(r["RICIR"], 3), _num(r["fitness"], 3)))
    sf_table = _table(["universe · factor", "IC", "ICIR", "RankIC", "RankICIR", "fitness"], sft_rows)

    # ---- model + backtest vs paper ----
    mb_rows = []
    for uni in results["universes"]:
        U = results["universes"][uni]
        paper = U.get("paper", {})
        arm_labels = [("baseline", "Baseline"), ("gplearn", "GPLearn arm (+GP)"),
                      ("augmented", "Augmented (+FE)")]
        for name, lab in arm_labels:
            if name not in U["model"]:
                continue
            m, b = U["model"][name], U["backtest"][name]
            cls = ' class="best"' if name == "augmented" else ""
            mb_rows.append((f'<span{cls}>{uni.upper()} · {lab}</span>', _num(m["IC"]), _num(m["RIC"]),
                            _pct(b["AR"]), _pct(b["AER"]), f'{b["IR"]:.2f}', f'{b["SR"]:.2f}', _pct(b["MDD"])))
        for pname in ("Alpha158", "GPLearn", "FE-alpha-2", "FE-report-2"):
            if pname in paper:
                pr = paper[pname]
                tag = "best" if pname == "FE-alpha-2" else ""   # FE-alpha is our fair target
                mb_rows.append((f'<i class="{tag}">{uni.upper()} · paper {pname}</i>', _num(pr["IC"]),
                                _num(pr["RIC"]), _pct(pr["AR"]), "—",
                                f'{pr["IR"]:.2f}', f'{pr["SR"]:.2f}', _pct(pr["MDD"])))
    mb_table = _table(["universe · model", "IC", "RankIC", "AR", "AER", "IR", "SR", "|MDD|"], mb_rows)

    # ---- robustness ----
    rob = results.get("robustness", {})
    rob_html = ""
    if rob:
        rr = []
        for k, lab in [("seed", "Seed"), ("fe_engine_evolved", "FE-evolved")]:
            d = rob[k]
            rr.append((lab, f'{d["IC"]["mean"]:+.4f} ± {d["IC"]["std"]:.4f}',
                       f'{d["RIC"]["mean"]:+.4f} ± {d["RIC"]["std"]:.4f}',
                       f'{d["fitness"]["mean"]:+.3f} ± {d["fitness"]["std"]:.3f}'))
        rob_note = ("" if not real else
                    " <b>Caveat for the real-data run:</b> the FE factor here was evolved on <i>real</i> "
                    "CSI300 (parameters tuned to it), so applying it to these synthetic panels is an "
                    "out-of-distribution transfer — the negative FE row reflects that mismatch, not the "
                    "engine's real-data quality (see the real CSI300/500 tables above). It is a parameter-"
                    "transfer stress test, not a performance claim.")
        rob_html = ("<p>Single-factor test metrics across "
                    f"{len(rob['seeds'])} independent synthetic data realizations (mean ± std), "
                    f"isolating the ~1/√N cross-sectional IC scatter.{rob_note}</p>"
                    + _table(["factor", "IC", "RankIC", "fitness"], rr))

    # ---- ablation tables ----
    abl_html = ""
    if abl.get("bayes_ablation"):
        ba = abl["bayes_ablation"]
        abl_html += ("<h3>Bayesian micro-search (Fig. 5)</h3>"
                     f"<p>Best objective with Bayesian search = <b>{ba['with_bayes']['best_reward']:+.3f}</b> "
                     f"vs without = <b>{ba['without_bayes']['best_reward']:+.3f}</b> "
                     "under the same macro budget.</p>")
        abl_html += f'<div class="fig">{_b64(f_bayes)}</div>'
    if abl.get("island_ablation"):
        ia = abl["island_ablation"]
        rows = [(f"{k} island(s)", f'{v["best_fitness_mining"]:+.3f}', _num(v["csi300_test"]["RIC"]),
                 _num(v["csi300_test"]["fitness"], 3)) for k, v in ia.items()]
        abl_html += ("<h3>Multi-island (Table 3)</h3>"
                     + _table(["islands", "mining fitness", "CSI300 test RankIC", "CSI300 test fitness"], rows))

    es = results["evolution_summary"]
    cur_obj = (evo.get("config", {}) or {}).get("objective") or es.get("objective") or "portfolio_v3"
    transforms = ", ".join(f"<code>{t}</code>" for t in es["best_transforms"]) or "—"
    params = ", ".join(f"<code>{k}={v}</code>" for k, v in es["best_params"].items())
    llm_meta = evo.get("llm", {}) or {}
    llm_last = llm_meta.get("last_success", {}) or {}
    llm_model = llm_last.get("model") or llm_meta.get("model") or "configured model"
    llm_base = (llm_last.get("base_url")
                or ", ".join(llm_meta.get("base_urls", []) or [])
                or "configured endpoint")
    llm_provider = llm_last.get("provider") or llm_meta.get("provider") or "configured provider"
    n_llm = int(evo.get("n_llm", 0) or 0)
    use_llm = bool(evo.get("config", {}).get("use_llm"))
    if use_llm and n_llm > 0:
        llm_gap = (f"a <b>live LLM macro-agent</b> accepted {n_llm} mutations "
                   f"(<code>{llm_provider}</code>, <code>{llm_model}</code>, "
                   f"<code>{llm_base}</code>), while deterministic fallback still covers "
                   "unusable transient outputs")
        llm_sub = (f"live configured LLM (<code>{llm_provider}</code>, "
                   f"<code>{llm_model}</code>, <code>{llm_base}</code>, "
                   f"<code>n_llm={n_llm}</code>) vs paper Gemini-2.5-Pro")
    elif use_llm:
        llm_gap = ("a live LLM endpoint was requested but accepted no valid mutations "
                   "(<code>n_llm=0</code>), so the engine used the deterministic/offline "
                   "proposal fallback for this saved run")
        llm_sub = ("configured LLM endpoint accepted no mutations in this run "
                   "(<code>n_llm=0</code>); deterministic/offline fallback used")
    else:
        llm_gap = ("no live LLM was requested for this saved run; the deterministic/offline "
                   "proposal macro path was used")
        llm_sub = ("deterministic/offline proposal macro path vs paper Gemini-2.5-Pro "
                   "(enable Kimi/Moonshot with <code>--use-llm</code>)")

    # --- data-driven verdict: did the in-sample gains transfer out-of-sample? ---
    _d3, _d5 = f"{fe_d3:+.4f}", f"{fe_d5:+.4f}"
    _b3, _a3 = _num(md["baseline"]["IC"]), _num(md["augmented"]["IC"])
    _b5, _a5 = _num(md5["baseline"]["IC"]), _num(md5["augmented"]["IC"])
    if fe_helps:
        transfer = (f" And the gains <b>transfer out-of-sample</b>: adding the elite factors raises test "
                    f"model IC on both universes (CSI300 {_b3}→{_a3}, CSI500 {_b5}→{_a5}).")
        overfit_para = ("With the strong Alpha158-128 baseline, FE's marginal out-of-sample lift is modest; "
                        "the absolute gap to FE-alpha-2 is mainly the 128/158-feature subset and reduced budget.")
    else:
        transfer = (" Whether these in-sample gains <b>transfer out-of-sample</b> is the key finding — see the "
                    "next box and Result&nbsp;2.")
        overfit_para = (
            f"<b>The elite factors overfit the 2015–16 validation window.</b> Out-of-sample (2017–2024), "
            f"validation-only elite selection is mixed rather than robust: CSI300 model IC falls "
            f"{_b3}→{_a3} (<b>{_d3}</b>), while CSI500 model IC rises slightly {_b5}→{_a5} "
            f"(<b>{_d5}</b>) but portfolio quality worsens (AR {_pct(u5['backtest']['baseline']['AR'])}→"
            f"{_pct(u5['backtest']['augmented']['AR'])}, SR {u5['backtest']['baseline']['SR']:.2f}→"
            f"{u5['backtest']['augmented']['SR']:.2f}). GPLearn beats the augmented model on CSI300 IC; "
            f"the augmented arm is the CSI500 IC winner, but not the portfolio winner. This is a faithful "
            f"reproduction of the <b>alpha-decay / regime-shift</b> failure mode FactorEngine's "
            f"diversity/regularization design is built to prevent — amplified here because {agent} optimized "
            f"a single-objective <code>combined_score</code> on a short, single-regime (488-day) validation set "
            f"with many free parameters. Bright spot: the FE-evolved <i>single</i> factor carries standalone "
            f"OOS signal on CSI500 (test IC {_num(sf5_oos)}).")

    if fe_helps:
        result2_note = (f"On real A-shares the elite factors add out-of-sample value over the Alpha158-128 "
                        f"baseline (CSI300 {_d3}, CSI500 {_d5}). <i>GPLearn</i> is the genetic-programming "
                        f"baseline run on the same data.")
        findings_lead = (f"<li><b>The evolved factors add out-of-sample value.</b> The elite factors lift model "
                         f"IC over Alpha158-128 (CSI300 {_d3}, CSI500 {_d5}); pooling them is more robust than any "
                         f"single program, echoing the paper's Integration module.</li>")
        conclusion_verdict = ("the elite evolved factors add IC and backtest value over the Alpha158-128 baseline "
                              "on both universes; it does not reach the paper's absolute Table-1 levels (mainly the "
                              "128/158-feature subset and reduced budget).")
    else:
        result2_note = (f"<b>Headline finding:</b> the validation-only elite factors <b>overfit</b>. "
                        f"They lower CSI300 model IC ({_d3}); on CSI500 they add a small IC lift ({_d5}) but "
                        f"still worsen AR/SR versus the baseline. Selecting elite factors by validation fitness "
                        f"alone chases a single 2015–16 regime; see Findings and the V3 protocol for the fix.")
        findings_lead = (
            f"<li><b>A live LLM factor-miner can overfit — and this run caught it.</b> {agent} lifted in-sample "
            f"validation fitness <b>{best_fit/seed_fit:.1f}×</b> ({seed_fit:+.3f}→{best_fit:+.3f}), but the "
            f"top-k elite factors (chosen by validation fitness) <b>do not transfer cleanly out-of-sample</b>: "
            f"CSI300 model IC falls ({_d3}), and CSI500's small IC gain ({_d5}) does not survive the portfolio "
            f"layer. That's the alpha-decay / regime-shift risk FactorEngine's diversity/regularization design "
            f"targets. <b>Implemented V3 fix:</b> select elite factors by OOS-robust, parsimony-penalized criteria "
            f"(sign-consistency across sub-windows + a parameter-count penalty), orthogonalize vs Alpha158, "
            f"and combine conservatively.</li>"
            f"<li><b>The idea isn't worthless — the selection/integration is.</b> The FE-evolved <i>single</i> "
            f"factor's CSI500 OOS IC is {_num(sf5_oos)} (real, standalone); the damage is in the validation-only "
            f"elite bundle and model-combination layer, not in Kimi connectivity or factor generation itself.</li>")
        conclusion_verdict = (
            f"the in-sample lift does <b>not</b> transfer robustly: the elite factors overfit the 2015–16 "
            f"validation window, reduce CSI300 model IC ({_d3}), and only add a fragile CSI500 IC gain "
            f"({_d5}) that fails in AR/SR — a faithful reproduction of the alpha-decay failure the paper's "
            f"design targets. The V3 protocol implements the response: OOS-robust, parsimony-penalized "
            f"selection; orthogonalization; portfolio-aware scoring; prompt/parser tightening; and "
            f"plateau-aware compute.")

    # ===== v2 sections: run diagnostics, search frontier, parameter contract,
    #       year-by-year overfit, and the V3 protocol (all data-driven) =====
    runtime_s = es.get("elapsed_s") or evo.get("elapsed_s") or 0
    n_iters_cfg = (evo.get("config", {}) or {}).get("iterations", es.get("n_evals"))
    diag_rows = [
        ("Panel evolved", es.get("panel", "—"), "Evolution ran on this universe; transferred to the other."),
        ("Iterations", n_iters_cfg, "Requested macro-agent budget."),
        ("Accepted live LLM mutations", n_llm,
         f"{(100*n_llm/n_iters_cfg):.0f}% of steps Kimi-driven; fallback covered transient/unparseable calls."
         if (is_llm_run and n_iters_cfg) else "Deterministic/offline macro path."),
        ("Provider / model", f"{llm_provider} / {llm_model}", "Live macro-agent backbone."),
        ("Endpoint", llm_base, "Route used by the evolution call path."),
        ("Runtime", f"{runtime_s/3600:.1f}h ({runtime_s:.0f}s)",
         f"≈{runtime_s/max(1,(n_iters_cfg or 1)):.0f}s per iteration." if n_iters_cfg else ""),
        ("Evaluations / nodes", f"{es.get('n_evals','—')} / {es.get('n_nodes','—')}", "Micro-search calls and tree size."),
        ("Best transforms", ", ".join(es.get("best_transforms", [])) or "—", "Macro edits in the best factor."),
        ("Validation fitness", f"{seed_fit:+.3f} → {best_fit:+.3f} ({best_fit/seed_fit:.1f}×)",
         "Eq.5 FS, seed → best (in-sample)."),
    ]
    diag_table = _table(["Item", "Value", "Interpretation"], diag_rows)

    # --- search frontier / plateau ---
    milestones, plateau_iter, n_iters_hist = _frontier_milestones(evo)
    fr_rows = [(it, f"{rw:+.4f}") for it, rw in milestones]
    frontier_table = _table(["iteration", "new best reward (valid combined_score)"], fr_rows)
    wasted = (n_iters_hist - plateau_iter) if (plateau_iter and n_iters_hist) else 0
    frontier_note = (
        f"The validation frontier was last improved at <b>iteration {plateau_iter}</b> of {n_iters_hist}; the "
        f"remaining <b>{wasted}</b> iterations explored variants without beating the global best. V3 #5 now "
        f"implements the plateau-aware response: stop or switch seed/split/objective after a patience window."
        if plateau_iter else "")

    # --- parameter contract scan ---
    used_p, decl_p, undeclared_p = _param_contract(evo)
    pc_rows = [(p, "tuned (in ###Parameters)" if p in decl_p else
                ("epsilon (constant)" if p == "epsilon" else "<b>hardcoded default — never tuned</b>"))
               for p in used_p]
    pc_table = _table(["parameter the factor reads", "status"], pc_rows)
    pc_note = (
        f"The evolved factor's code reads <b>{len(used_p)}</b> parameters but only <b>{len(decl_p)}</b> were "
        f"declared for Bayesian tuning; <b>{', '.join(undeclared_p) or 'none'}</b> stayed at hardcoded defaults, "
        f"so the headline factor is <i>under-optimized</i>. V3 #2/#6 now closes this loophole: auto-scan every "
        f"<code>parameters.get()</code> key, force it into <code>###Parameters</code>, and make hidden knobs "
        f"visible to the selection penalty and parser audit." if undeclared_p else
        f"All {len(used_p)} parameters the factor reads were declared for tuning — no hidden knobs.")

    # --- year-by-year overfit ---
    of_rows = []
    for uni in results["universes"]:
        b, w, n, wy, wd = _yearly_overfit(results["universes"][uni])
        of_rows.append((uni.upper(), f"{b}/{n} years better", f"{w}/{n} years worse",
                        wy or "—", _num(wd)))
    overfit_table = _table(["universe", "augmented IC better", "augmented IC worse",
                            "worst year", "worst IC Δ"], of_rows)

    # --- excess-return findings and concrete improvement plan ---
    excess_rows = []
    for uni in results["universes"]:
        U = results["universes"][uni]
        b = U["backtest"]["baseline"]
        a = U["backtest"]["augmented"]
        m_b = U["model"]["baseline"]
        m_a = U["model"]["augmented"]
        excess_rows.append((
            uni.upper(),
            _num(m_b["IC"]),
            _num(m_a["IC"]),
            _pct(b["AER"]),
            _pct(a["AER"]),
            _pct(a["AER"] - b["AER"]),
            _pct(b.get("ann_excess")),
            _pct(a.get("ann_excess")),
            f'{b["SR"]:.2f}→{a["SR"]:.2f}',
            f'{_pct(b.get("ann_cost"))}→{_pct(a.get("ann_cost"))}',
            f'{b.get("ann_turnover", 0):.1f}→{a.get("ann_turnover", 0):.1f}',
        ))
    excess_table = _table(["universe", "base IC", "+FE IC", "base AER", "+FE AER",
                           "AER Δ", "base ann excess", "+FE ann excess", "SR",
                           "ann cost", "ann turnover"], excess_rows)

    def _gross_aer(m):
        try:
            return float(m.get("AER", 0)) + float(m.get("ann_cost", 0))
        except Exception:
            return 0.0

    def _turnover_cut_net(m, cut=3.0):
        try:
            return _gross_aer(m) - float(m.get("ann_cost", 0)) / cut
        except Exception:
            return 0.0

    e3_levers = ex.get("csi300", {}).get("levers", {}) if ex else {}
    e5_levers = ex.get("csi500", {}).get("levers", {}) if ex else {}
    t3_levers = ext2.get("csi300", {}).get("levers", {}) if ext2 else {}
    t5_levers = ext2.get("csi500", {}).get("levers", {}) if ext2 else {}
    csi300_cost_case = u3["backtest"]["augmented"]
    csi500_cost_case = e5_levers.get("A3_mutual_ortho") or u5["backtest"]["augmented"]
    gross_rows = []
    for label, m in [
        ("CSI300 V4 default", csi300_cost_case),
        ("CSI500 V4 + A3", csi500_cost_case),
    ]:
        gross_rows.append((
            label,
            _pct(m.get("AER")),
            _pct(m.get("ann_cost")),
            _pct(_gross_aer(m)),
            f'{m.get("ann_turnover", 0):.1f}×',
            _pct(_turnover_cut_net(m)),
        ))
    gross_table = _table(["case", "net AER", "ann cost drag", "gross AER proxy",
                          "turnover", "net AER if cost / 3"], gross_rows)

    if ortho and ortho.get("universes"):
        _u3o = ortho["universes"].get("csi300", {})
        _u5o = ortho["universes"].get("csi500", {})
        aux_note = (
            f"Fresh V3 diagnostics are informative: on CSI300, raw robust selection lifts model IC "
            f"{_num(_u3o.get('baseline',{}).get('IC'))}→{_num(_u3o.get('robust_raw',{}).get('IC'))} "
            f"but still leaves Sharpe below baseline ({_u3o.get('baseline',{}).get('SR',0):.2f}→"
            f"{_u3o.get('robust_raw',{}).get('SR',0):.2f}); on CSI500, raw/orthogonal robust factors lift "
            f"IC and Sharpe versus baseline (raw IC {_num(_u5o.get('robust_raw',{}).get('IC'))}, "
            f"orth SR {_u5o.get('robust_orthogonal',{}).get('SR',0):.2f}). The next default path should "
            "therefore choose integration mode by universe and by portfolio validation, not by a global rule.")
    else:
        aux_note = (
            "Run the V3 robust/orthogonal diagnostics before finalizing the default path, so the model can choose "
            "the integration mode that improves portfolio metrics rather than only model IC.")

    excess_plan_html = (
        "<h2>Excess-return findings and improvement plan</h2>"
        "<p><b>Finding:</b> V4 improves benchmark-relative return on both universes, but the remaining failure is "
        "mostly <b>execution cost</b>, not missing signal. A simple gross-vs-net decomposition "
        "(gross proxy = net AER + annualized transaction cost) shows the strategy already earns positive gross "
        "excess return; 75-84× annual turnover consumes it.</p>"
        f"{excess_table}"
        "<h3>Gross-vs-net decomposition</h3>"
        f"{gross_table}"
        "<p class=\"small\">This reframes the next step: making net AER positive is less about inventing a stronger "
        "factor and more about trading the existing signal more slowly. A mechanical 3× turnover/cost cut would "
        "flip both shown cases positive if gross excess stayed roughly intact.</p>"
        f"<p class=\"small\">{aux_note}</p>"
        "<ol>"
        "<li><b>Tier 1: run a turnover sweep on cached V4 predictions.</b> Test EWM signal smoothing "
        "(span 5/10), longer holds (10/15/20 days), and true rank-band rebalancing (buy top-30, sell outside "
        "top-100). This is backtest-only and directly targets the 9-10% cost drag.</li>"
        "<li><b>Combine the best turnover levers.</b> C2 selection hysteresis barely reduced turnover because the "
        "5-day, 5-tranche book still forces churn. The next sweep should attack prediction stability and holding "
        "length, not only selection retention.</li>"
        "<li><b>Select per universe.</b> CSI300 currently likes the beta/size-neutral tier-2 variant; CSI500 likes "
        "A3 factor-set decorrelation alone. A single shared integration rule leaves money on the table.</li>"
        "<li><b>Tier 2: evolve with a true net objective.</b> <code>portfolio_v4</code> optimized gross top-decile "
        "excess with turnover/complexity penalties. A <code>portfolio_v5</code> objective should subtract the "
        "actual A-share cost model from estimated turnover so the search optimizes tradeable net alpha.</li>"
        "<li><b>Persist portfolio diagnostics for every arm.</b> Store gross AER proxy, net AER, IR, RMDD, "
        "annualized excess, annualized turnover, and annualized cost so the report can separate signal quality "
        "from implementation drag.</li>"
        "</ol>")

    v3_html = (
        "<ol>"
        "<li><b>#1 Robust elite selection — implemented.</b> Rank factors by multi-window, cross-universe "
        "robustness (train∧validation sign consistency plus degradation penalty), not validation fitness alone "
        "(Result 4 / <code>robust_elite.json</code>).</li>"
        "<li><b>#2 Complexity &amp; hidden-knob penalty — implemented.</b> Penalize parameter/transform count and "
        "undeclared <code>parameters.get()</code> defaults, so factors with hidden knobs lose score "
        "(Result 4 and the Parameter Contract audit).</li>"
        "<li><b>#3 Redundancy cap / orthogonalization — implemented.</b> Residualize robust factors against "
        "Alpha158 and gate on marginal IC, so the model receives orthogonal alpha instead of duplicate "
        "turnover/liquidity exposure (Result 5 / <code>orthogonal_elite.json</code>).</li>"
        "<li><b>#4 Portfolio-aware objective — demonstrated.</b> Re-rank by IC + yearly stability − turnover, "
        "which promotes lower-turnover, regime-stable candidates over raw-IC winners "
        "(Result 6 / <code>v3_4to6.json</code>).</li>"
        "<li><b>#5 Plateau-aware compute — implemented/demonstrated.</b> This run found the frontier by iteration "
        f"{plateau_iter or '~99'}; <code>EvolutionConfig.patience</code> and the post-hoc audit show the same best "
        "factor could be reached with less compute once the frontier plateaus (Result 6).</li>"
        "<li><b>#6 Tighter LLM prompt/parser + parameter contract — implemented.</b> The Kimi response format now "
        "requires one mutation theme, all tunables declared in JSON ranges, no look-ahead, and a CSI300/CSI500 "
        "microstructure rationale; the parser auto-declares hidden <code>parameters.get()</code> knobs and Result 6 "
        "audits their effect.</li>"
        "</ol>")

    # --- V3 #1+#2 applied: robust-selection A/B (if scripts/06 was run) ---
    v3result_html = ""
    if robust and robust.get("universes"):
        cc = robust.get("candidates", {})
        gk = cc.get("claude_gk_lowvol", {})
        ab_rows = []
        for uni, u in robust["universes"].items():
            b, vo, rb = u["baseline"], u["validation_only"], u["robust"]
            ab_rows.append((uni.upper(), _num(b["IC"]), _num(vo["IC"]), _num(rb["IC"]),
                            f"{b['SR']:.2f}", f"{vo['SR']:.2f}", f"{rb['SR']:.2f}"))
        ab_table = _table(["universe", "baseline IC", "valid-only IC", "robust IC",
                           "base SR", "valid SR", "robust SR"], ab_rows)
        u3r = robust["universes"].get("csi300", {})
        dvo = u3r.get("validation_only", {}).get("IC", 0) - u3r.get("baseline", {}).get("IC", 0)
        drb = u3r.get("robust", {}).get("IC", 0) - u3r.get("baseline", {}).get("IC", 0)
        valid_names = ", ".join(robust.get("validation_only", [])) or "—"
        robust_names = ", ".join(robust.get("robust", [])) or "—"
        robust_status = "degradation eliminated" if drb >= 0 else "degradation reduced"
        gk_note = (f" <code>gk_lowvol</code> remains a compact comparator "
                   f"(validation→test IC {_num(gk.get('ic_valid'))}→{_num(gk.get('ic_test'))}, "
                   "1 param, 0 hidden)." if gk else "")
        v3result_html = (
            "<h2>Result 4 — V3 #1–#2 implemented: robust selection stress-tests the elite set</h2>"
            "<p>Re-selecting elite factors with the V3 rule — sign-consistent on <b>train AND validation</b> "
            "plus a parsimony / hidden-knob penalty — using the <i>existing</i> 300-iter run (<b>no LLM "
            "re-run</b>). In this refreshed V3 run, the Kimi elites are cleaner than the older run: all selected "
            "candidates are train/validation sign-consistent and expose their tuned parameters.</p>"
            "<ul>"
            f"<li><b>Validation-only</b> picks <code>{valid_names}</code>.</li>"
            f"<li><b>Robust</b> reorders/picks <code>{robust_names}</code>.{gk_note}</li>"
            "</ul>"
            f"{ab_table}"
            f"<p class=\"small\">On CSI300 robust selection moves the augmented model from "
            f"{_num(u3r.get('validation_only',{}).get('IC'))} (<b>{dvo:+.4f}</b> vs baseline) to "
            f"{_num(u3r.get('robust',{}).get('IC'))} (<b>{drb:+.4f}</b> — <b>{robust_status}</b>). "
            f"Sharpe remains a constraint ({u3r.get('validation_only',{}).get('SR',0):.2f}→"
            f"{u3r.get('robust',{}).get('SR',0):.2f}, baseline {u3r.get('baseline',{}).get('SR',0):.2f}), "
            "so this fix improves IC robustness but does not by itself solve portfolio excess return.</p>")

    # --- V3 #3 applied: orthogonalization A/B (if scripts/07 was run) ---
    ortho_html = ""
    if ortho and ortho.get("universes"):
        rows = []
        for uni, u in ortho["universes"].items():
            b, rr, oo = u["baseline"], u["robust_raw"], u["robust_orthogonal"]
            rows.append((uni.upper(), _num(b["IC"]), _num(rr["IC"]), _num(oo["IC"]),
                         f"{b['SR']:.2f}", f"{rr['SR']:.2f}", f"{oo['SR']:.2f}"))
        ab = _table(["universe", "baseline IC", "robust-raw IC", "robust-orthogonal IC",
                     "base SR", "raw SR", "orth SR"], rows)
        u3 = ortho["universes"].get("csi300", {})
        u5 = ortho["universes"].get("csi500", {})
        d3_raw = u3.get("robust_raw", {}).get("IC", 0) - u3.get("baseline", {}).get("IC", 0)
        d3_orth = u3.get("robust_orthogonal", {}).get("IC", 0) - u3.get("baseline", {}).get("IC", 0)
        d5_raw = u5.get("robust_raw", {}).get("IC", 0) - u5.get("baseline", {}).get("IC", 0)
        d5 = u5.get("robust_orthogonal", {}).get("IC", 0) - u5.get("baseline", {}).get("IC", 0)
        beats5 = d5 > 0
        orth_basis = rc.get("orthogonal_max_features")
        orth_note = (f" This report run used the top-{orth_basis} Alpha158 columns as the residualization basis "
                     "for tractability." if orth_basis else "")
        ortho_html = (
            "<h2>Result 5 — V3 #3 implemented: orthogonal residual alpha is universe-dependent</h2>"
            "<p>Each robust factor is residualized per date against the 128 Alpha158 features and gated on "
            "marginal (residual) IC — feeding the model only the component Alpha158 does <i>not</i> already "
            f"capture.{orth_note}</p>"
            f"{ab}"
            f"<p class=\"small\">On <b>CSI500</b>, raw robust factors lift test IC by <b>{d5_raw:+.4f}</b>; "
            f"orthogonal residuals retain a smaller "
            f"{_num(u5.get('baseline',{}).get('IC'))}→{_num(u5.get('robust_orthogonal',{}).get('IC'))} "
            f"(<b>{d5:+.4f}</b>, {'+' if beats5 else ''}{100*d5/max(1e-9,u5.get('baseline',{}).get('IC',1)):.0f}% over baseline) "
            f"and Sharpe {u5.get('baseline',{}).get('SR',0):.2f}→{u5.get('robust_orthogonal',{}).get('SR',0):.2f}. "
            f"On <b>CSI300</b>, raw robust factors add <b>{d3_raw:+.4f}</b> IC but orthogonalization cuts the "
            f"model to {_num(u3.get('robust_orthogonal',{}).get('IC'))} (<b>{d3_orth:+.4f}</b> vs baseline). "
            "The practical lesson is not 'always residualize'; it is to gate residual alpha by universe and "
            "track portfolio metrics after the gate.</p>")

    # --- V3 #4–#6 demonstrated on the existing run (scripts/08) ---
    v3_456_html = ""
    if v3_456:
        pc = v3_456.get("param_contract", {})
        pf = v3_456.get("portfolio", {})
        pl = v3_456.get("plateau", {})
        # #4 table — candidates ranked by portfolio score
        cand = pf.get("candidates", {})
        prows = []
        for nm in sorted(cand, key=lambda k: cand[k]["score"], reverse=True):
            c = cand[nm]
            star = ' ★' if nm in pf.get("portfolio_top", []) else ''
            prows.append((f"<code>{nm}</code>{star}", _num(c["ic"]), f"{c['turnover']:.3f}",
                          _num(c["min_year_ic"]), _num(c["score"])))
        ptable = _table(["candidate", "IC", "turnover", "min-yr IC", "portfolio score"], prows)
        bef, aft = pc.get("before", {}), pc.get("after", {})
        s30 = pl.get("scenarios", {}).get("30", {})
        s50 = pl.get("scenarios", {}).get("50", {})
        hidden_knobs = pc.get("hidden_knobs", [])
        ic_top = ", ".join(pf.get("ic_only_top", [])) or "—"
        port_top = ", ".join(pf.get("portfolio_top", [])) or "—"
        leader = (pf.get("portfolio_top") or [""])[0]
        leader_stats = cand.get(leader, {})
        contract_html = (
            "<p>The refreshed best Kimi factor passes the parameter-contract audit: every tunable it reads is "
            "declared for Bayesian search, so there are <b>no hidden knobs</b> to auto-promote. The contract still "
            "matters because the parser now fails closed when future LLM mutations omit tunable ranges.</p>"
            f"<ul><li>valid fitness {_num(bef.get('valid_fit'),3)} → <b>{_num(aft.get('valid_fit'),3)}</b> "
            "(no-op because no hidden knobs were found);</li>"
            f"<li>test IC {_num(bef.get('test_ic'))} → {_num(aft.get('test_ic'))}.</li></ul>"
            if not hidden_knobs else
            "<p>The best Kimi factor left hidden knobs at hardcoded defaults that never entered Bayesian search. "
            "The V3 prompt/parser auto-declares every such <code>parameters.get()</code> default into the search "
            "space. Re-tuning just those hidden knobs via Optuna (30 trials, on validation):</p>"
            f"<ul><li>valid fitness {_num(bef.get('valid_fit'),3)} → <b>{_num(aft.get('valid_fit'),3)}</b>;</li>"
            f"<li>test IC {_num(bef.get('test_ic'))} → {_num(aft.get('test_ic'))}.</li></ul>"
        )
        v3_456_html = (
            "<h2>Result 6 — V3 #4–#6 implemented/demonstrated (portfolio objective, prompt/parser contract, plateau stop)</h2>"
            "<p>The final three protocol items, exercised on the <i>existing</i> 300-iter run "
            "(<b>no LLM re-run</b>) so each is reproducible offline.</p>"

            "<h3>#4 — Portfolio-aware objective (IC + yearly stability − turnover)</h3>"
            "<p>Ranking by raw IC alone ignores trading cost and regime stability. Re-scoring the candidate pool by "
            "<code>IC + 0.5·min-yearly-IC − 0.4·turnover</code> (turnover = 1 − consecutive-date rank "
            "autocorrelation) changes the podium:</p>"
            f"<ul><li><b>IC-only</b> top-3: <code>{ic_top}</code>.</li>"
            f"<li><b>Portfolio-aware</b> top-3: <code>{port_top}</code>. The top-ranked candidate "
            f"<code>{leader}</code> combines IC {_num(leader_stats.get('ic'))}, turnover "
            f"{leader_stats.get('turnover',0):.3f}, and min-year IC {_num(leader_stats.get('min_year_ic'))}.</li></ul>"
            f"{ptable}"
            "<p class=\"small\">The seed scores worst (turnover ≈1.0 — it re-shuffles the whole cross-section daily). "
            "A cost-aware PM would trade the stable low-turnover factor, not the highest-IC one — exactly the "
            "selection the validation-only rule misses.</p>"

            "<h3>#6 — Tighter prompt/parser + parameter-contract enforcement</h3>"
            "<p>The V3 prompt/parser now requires one mutation theme, no look-ahead, a CSI300/CSI500 "
            "microstructure rationale, and declared JSON ranges for every tunable; the parser also auto-declares "
            "every <code>parameters.get()</code> default into the search space.</p>"
            f"{contract_html}"

            "<h3>#5 — Plateau-aware compute</h3>"
            f"<p>The validation frontier last improved at iteration <b>{pl.get('plateau_iter')}</b> of "
            f"{pl.get('n_iters')}; everything after was wasted on this run. Post-hoc early-stopping at the same best "
            "factor would have saved:</p>"
            "<ul>"
            f"<li>patience=30 → stop at iter {s30.get('stop_iter')}/{pl.get('n_iters')}, "
            f"<b>−{s30.get('saved_s',0)/3600:.1f}h ({100*s30.get('saved_frac',0):.0f}%)</b>;</li>"
            f"<li>patience=50 → stop at iter {s50.get('stop_iter')}/{pl.get('n_iters')}, "
            f"<b>−{s50.get('saved_s',0)/3600:.1f}h ({100*s50.get('saved_frac',0):.0f}%)</b>.</li>"
            "</ul>"
            "<p class=\"small\">Wired as <code>EvolutionConfig.patience</code> (early-stop after N non-improving "
            f"iterations). At patience=30 this run saves <b>{100*s30.get('saved_frac',0):.0f}%</b> of the compute "
            "while preserving the same best factor.</p>")

    if ortho and ortho.get("universes"):
        _u3o = ortho["universes"].get("csi300", {})
        _u5o = ortho["universes"].get("csi500", {})
        _d3r = _u3o.get("robust_raw", {}).get("IC", 0) - _u3o.get("baseline", {}).get("IC", 0)
        _d3o = _u3o.get("robust_orthogonal", {}).get("IC", 0) - _u3o.get("baseline", {}).get("IC", 0)
        _d5r = _u5o.get("robust_raw", {}).get("IC", 0) - _u5o.get("baseline", {}).get("IC", 0)
        _d5o = _u5o.get("robust_orthogonal", {}).get("IC", 0) - _u5o.get("baseline", {}).get("IC", 0)
        closed_loop_note = (
            "And we closed the loop: OOS-robust + parsimony-penalized selection (Result 4) repairs the "
            f"CSI300 IC degradation (<b>{_d3r:+.4f}</b> raw robust vs baseline), while per-date "
            f"orthogonalization shows a smaller but positive CSI500 residual lift (<b>{_d5o:+.4f}</b>; raw "
            f"robust is <b>{_d5r:+.4f}</b>). CSI300 orthogonalization is too aggressive in this run "
            f"(<b>{_d3o:+.4f}</b>), so V3's lesson is universe-aware gating, not blind residualization. ")
    elif v3result_html:
        closed_loop_note = (
            "And we closed the loop: applying OOS-robust + parsimony-penalized selection to the same run "
            "repairs the CSI300 IC degradation, confirming the fault was validation-only selection rather "
            "than Kimi connectivity. ")
    else:
        closed_loop_note = ""

    if v3_456_html:
        _pct30 = 100 * (v3_456.get("plateau", {}).get("scenarios", {}).get("30", {}).get("saved_frac", 0) or 0)
        _hidden = v3_456.get("param_contract", {}).get("hidden_knobs", [])
        _contract = ("the prompt/parser contract finds no hidden knobs in the refreshed best Kimi factor"
                     if not _hidden else
                     "the prompt/parser contract exposes hidden knobs for audit and tuning")
        result6_note = (
            f"Result 6 then exercises the final protocol items on the same run: a portfolio-aware objective "
            f"re-ranks toward lower-turnover, regime-stable candidates; {_contract}; and plateau-aware "
            f"early-stopping would save {_pct30:.0f}% of compute at patience=30. ")
    else:
        result6_note = ""

    # --- Result 7 — excess-return levers (scripts/09_excess_return.py) ---
    excess_html = ""
    if ex:
        def _ex_table(uni):
            lv = ex.get(uni, {}).get("levers", {})
            b_aer = lv.get("V3_baseline", {}).get("AER", 0)
            labels = {
                "V3_baseline": "V4 default",
                "B1_size_neutral": "B1 size-neutral",
                "C1_rank_weight": "C1 rank-weight",
                "C2_hyst_0.5": "C2 hysteresis 0.5",
                "A3_mutual_ortho": "A3 mutual-ortho",
                "A3_plus_B1": "A3 + B1",
                "A3_B1_C2": "A3 + B1 + C2",
            }
            order = ["V3_baseline", "B1_size_neutral", "C1_rank_weight", "C2_hyst_0.5",
                     "A3_mutual_ortho", "A3_plus_B1", "A3_B1_C2"]
            best_key = max(
                (k for k, v in lv.items() if isinstance(v, dict) and "AER" in v),
                key=lambda k: lv[k].get("AER", -99),
                default=None,
            )
            rows = []
            for k in order:
                m = lv.get(k)
                if not m:
                    continue
                d = m.get("AER", 0) - b_aer
                star = ' class="best"' if k == best_key else ''
                rows.append((f'<span{star}>{labels.get(k, k)}</span>', _num(m.get("AER")), _num(m.get("IR"), 3),
                             _num(m.get("AR")), _num(m.get("SR"), 3),
                             f'{m.get("ann_turnover", 0):.1f}', _num(d)))
            return _table(["lever (test)", "AER", "IR", "AR", "SR", "turnover", "ΔAER"], rows)

        e3 = ex.get("csi300", {}).get("levers", {})
        e5 = ex.get("csi500", {}).get("levers", {})

        def _g(d, k, f):
            return d.get(k, {}).get(f, 0)

        def _best_lever(d):
            vals = [(k, v) for k, v in d.items() if isinstance(v, dict) and "AER" in v]
            return max(vals, key=lambda kv: kv[1].get("AER", -99)) if vals else ("—", {})

        b3_key, b3_val = _best_lever(e3)
        b5_key, b5_val = _best_lever(e5)

        excess_html = (
            "<h2>Result 7 — excess-return levers (adjusting the factor set)</h2>"
            "<p>The excess-return lever files are A/B'd against the current <b>V4 default</b> arm "
            "(kept under the historical JSON key <code>V3_baseline</code> for compatibility). Net AER remains "
            "slightly negative because turnover is very high, not because gross signal is absent. Four offline "
            "levers were tested without an LLM re-run: <b>A3</b> mutually orthogonalizes the elite set, "
            "<b>B1</b> size-neutralizes the combined signal, <b>C1</b> rank-weights holdings by conviction, and "
            "<b>C2</b> adds a selection no-trade band.</p>"
            "<h3>CSI300 (large-cap)</h3>" + _ex_table("csi300") +
            "<h3>CSI500 (mid-cap)</h3>" + _ex_table("csi500") +
            "<p class=\"small\"><b>Finding.</b> The post-hoc levers are useful but not decisive. "
            f"On <b>CSI300</b>, the best tier-1 lever is <code>{b3_key}</code>, lifting AER "
            f"{_pct(_g(e3,'V3_baseline','AER'))}→<b>{_pct(b3_val.get('AER'))}</b>; on "
            f"<b>CSI500</b>, <code>{b5_key}</code> is best, lifting AER "
            f"{_pct(_g(e5,'V3_baseline','AER'))}→<b>{_pct(b5_val.get('AER'))}</b>. "
            "C1 is rejected because conviction-weighting amplifies unstable tail names. C2 helps CSI300 a little "
            "in this V4 run, but it barely changes turnover, so it is not the true solution. The remaining gap is "
            "a turnover/cost problem: the signal is gross-positive, while the 5-day overlapping-tranche book keeps "
            "annualized turnover near 75-84×.</p>")

        # tier-2 levers (scripts/10) — reported per universe because the best arm differs.
        if ext2:
            def _t2(uni, k, f="AER"):
                return ext2.get(uni, {}).get("levers", {}).get(k, {}).get(f, 0)
            t3_key, t3_val = _best_lever(ext2.get("csi300", {}).get("levers", {}))
            t5_key, t5_val = _best_lever(ext2.get("csi500", {}).get("levers", {}))
            excess_html += (
                "<h3>Tier 2 — beta-neutralization &amp; IC-weighted combine</h3>"
                "<p class=\"small\">The tier-2 results are universe-dependent. "
                f"On <b>CSI300</b>, <code>{t3_key}</code> is currently best "
                f"(AER <b>{_pct(t3_val.get('AER'))}</b> vs V4 default {_pct(_t2('csi300','V3_baseline'))}), "
                "so beta/size neutralization is worth carrying forward there. On "
                f"<b>CSI500</b>, <code>{t5_key}</code> remains best "
                f"(AER <b>{_pct(t5_val.get('AER'))}</b>), and beta neutralization dilutes the mid-cap signal. "
                f"<b>B2</b> (IC-weighted <i>linear</i> combine instead of the LightGBM tree-merge) is rejected "
                f"(CSI300 {_pct(_t2('csi300','B2_ic_weighted'))}, CSI500 {_pct(_t2('csi500','B2_ic_weighted'))}): "
                "a single-feature-IC linear sum cannot model the tree's interactions and double-counts collinear "
                "Alpha158 features. <b>D1:</b> the gross-excess <code>portfolio_v4</code> and cost-net "
                "<code>portfolio_v5</code> objectives are implemented, but — per <b>Result 8</b> — a now-fixed "
                "micro-search bug meant no run had actually optimized them; a genuine objective comparison is "
                "pending a fresh run.</p>")

    # --- Result 8 — portfolio_v4 vs portfolio_v3 (evolving FOR excess return) ---
    v3v4_html = ""
    if results_v3 and results.get("universes"):
        def _au(res, uni, sect, key):
            return res.get("universes", {}).get(uni, {}).get(sect, {}).get("augmented", {}).get(key, 0)
        rows = []
        for uni in results.get("universes", {}):
            if uni not in results_v3.get("universes", {}):
                continue
            rows.append((uni.upper(),
                         _num(_au(results_v3, uni, "model", "IC")), _num(_au(results, uni, "model", "IC")),
                         _num(_au(results_v3, uni, "backtest", "AER")), _num(_au(results, uni, "backtest", "AER")),
                         f'{_au(results_v3, uni, "backtest", "SR"):.2f}', f'{_au(results, uni, "backtest", "SR"):.2f}'))
        v3v4_tbl = _table(["universe", "IC run A", "IC run B", "AER run A", "AER run B", "SR run A", "SR run B"], rows)
        v3v4_html = (
            "<h2>Result 8 — methodological correction: the search objective never actually changed</h2>"
            "<p>The plan's lever D1 — evolve <i>for</i> excess return by switching the search objective from IC "
            "(<code>portfolio_v3</code>) to a gross-excess objective (<code>portfolio_v4</code>) — was implemented "
            "and unit-tested, and an earlier draft of this report credited it with the improvement below. "
            "<b>That attribution was wrong.</b> A variable-shadowing bug in the Optuna micro-search "
            "(<code>optimize_parameters</code> defined a local <code>objective</code> function that shadowed its "
            "<code>objective</code> string argument) silently routed <b>every</b> run through "
            "<code>portfolio_v3</code>, whatever <code>--objective</code> was passed.</p>"
            "<p><b>Proof:</b> recomputing each run's best factor, the stored reward matches the "
            "<code>portfolio_v3</code> score to five decimals (the \"v4\" run 0.31262 vs the v4-formula 0.30149; "
            "the live \"v5\" run 0.32357 vs the v5-formula 0.37043). So the two live-Kimi runs below both "
            "optimized <code>portfolio_v3</code> — their differences are <b>LLM-trajectory / mutation-count "
            "noise</b> (run A ≈250 accepted mutations, run B ≈10), not the objective:</p>"
            f"{v3v4_tbl}"
            "<p class=\"small\"><b>Fix &amp; hardening.</b> The nested function was renamed so the objective "
            "string is honored, and <code>evaluate_objective</code> now <b>raises on an unknown objective</b> "
            "rather than silently defaulting to <code>portfolio_v3</code> — a stale build fails loudly instead of "
            "mislabelling a run. Validated: the micro-search now threads <code>portfolio_v3 / v4 / v5</code> "
            "correctly. This is precisely the silent-failure class the project's OOS-rigor theme targets — a "
            "config flag recorded but not honored — and the guard is what surfaced it. <b>A genuine objective "
            "comparison (v3 vs gross-excess v4 vs cost-net v5) now requires a fresh run with the fix</b> "
            "(pending). The downstream <b>Result 9</b> turnover finding is unaffected: it is a backtest-side "
            "analysis of the elite predictions, independent of the evolution objective.</p>")

    if fe_helps:
        exec_takeaway = (
            f"<p>The live <b>{llm_provider}/{llm_model}</b> macro-agent drove a 300-iteration "
            f"FactorEngine run on real A-share data — <b>{n_llm} of {n_iters_cfg}</b> accepted steps were "
            f"Kimi-authored mutations — lifting validation fitness <b>{best_fit/seed_fit:.1f}×</b> "
            f"({seed_fit:+.3f}→{best_fit:+.3f}). The augmented multi-factor model's IC is positive on both "
            f"universes (CSI300 {_b3}→{_a3}, CSI500 {_b5}→{_a5}) and its backtest earns <b>positive gross</b> "
            "excess return — but heavy turnover erases it: CSI300 earns about "
            f"<b>{_pct(_gross_aer(csi300_cost_case))}</b> gross AER before "
            f"<b>{_pct(csi300_cost_case.get('ann_cost'))}</b> annualized cost, CSI500 about "
            f"<b>{_pct(_gross_aer(csi500_cost_case))}</b> before <b>{_pct(csi500_cost_case.get('ann_cost'))}</b>. "
            "The signal is real; the bottleneck is <b>turnover/cost</b>, which <b>Result 9</b> fixes — a longer "
            "holding period delivers <b>positive net excess return</b> on both universes. <b>Correction "
            "(Result 8):</b> a now-fixed micro-search bug meant the evolution objective could not actually be "
            "varied, so no finding here should be attributed to changing it.</p>")
    else:
        exec_takeaway = (
            f"<p>The live <b>{llm_provider}/{llm_model}</b> macro-agent successfully drove FactorEngine's "
            f"evolution on real A-share data — <b>{n_llm} of {n_iters_cfg}</b> steps were Kimi-authored "
            f"mutations — and lifted in-sample validation fitness <b>{best_fit/seed_fit:.1f}×</b> "
            f"({seed_fit:+.3f}→{best_fit:+.3f}). The honest result is that this does <b>not</b> transfer: "
            "selecting elite factors by validation fitness alone <b>overfits</b>, so the augmented multi-factor "
            f"model <b>does not transfer robustly out-of-sample</b>: CSI300 degrades ({_d3}), while CSI500 gets "
            f"only a small IC lift ({_d5}) that fails to improve AR/SR. The bottleneck is "
            "<b>selection/integration, not LLM connectivity</b> — see the V3 protocol.</p>")

    # --- Result 9 — turnover control delivers positive net excess return (scripts/11) ---
    sweep_html = ""
    if sweep:
        def _cfg_label(k):
            arm, s, h, b = k.split("|")
            parts = [("A3" if arm == "a3" else "base"), f"{h[1:]}d hold"]
            if s != "s1":
                parts.append(f"EWM{s[1:]}")
            if b != "b0":
                parts.append("band")
            return ", ".join(parts)
        srows, best_by_uni = [], {}
        for uni in sweep:
            grid = sweep[uni].get("grid", {})
            if not grid:
                continue
            dft = grid.get("base|s1|h5|b0", {})
            bestk = max(grid, key=lambda k: grid[k]["AER"])
            best = grid[bestk]
            best_by_uni[uni] = (bestk, best)
            srows.append((f"{uni.upper()} — default (5d hold)", _num(dft.get("AER")), _num(dft.get("IR"), 3),
                          _num(dft.get("SR"), 3), f'{dft.get("turnover", 0):.0f}×', _pct(dft.get("ann_cost")),
                          _num(dft.get("gross_AER"))))
            srows.append((f'<span class="best">{uni.upper()} — {_cfg_label(bestk)}</span>', _num(best["AER"]),
                          _num(best["IR"], 3), _num(best["SR"], 3), f'{best["turnover"]:.0f}×',
                          _pct(best["ann_cost"]), _num(best["gross_AER"])))
        stbl = _table(["config", "net AER", "IR", "SR", "turnover", "ann cost", "gross AER"], srows)

        def _hold(uni):                       # holding days of the best config for a universe
            k = best_by_uni.get(uni, ("|||",))[0]
            return k.split("|")[2][1:] if uni in best_by_uni else "?"
        k3, m3 = best_by_uni.get("csi300", ("", {}))
        k5, m5 = best_by_uni.get("csi500", ("", {}))
        sweep_html = (
            "<h2>Result 9 — closing the gap: turnover control gives positive net excess return</h2>"
            "<p>Result 7's decomposition showed the augmented model earns <b>+8–10% gross</b> excess return but "
            "~9–10% annualized transaction cost (turnover ~75–84×) erases it — so the gap was never a "
            "<i>signal</i> problem, it was a <b>cost</b> problem. Trading the <i>same</i> v4 predictions less "
            "often — the holding period, EWM smoothing, and a rank band, all backtest-only on the cached preds "
            "(no model rebuild) — collapses turnover and flips net excess return <b>positive on both "
            "universes</b>:</p>"
            f"{stbl}"
            "<p class=\"small\"><b>The dominant lever is the holding period.</b> A fine sweep over "
            "{5,10,12,15,20,25,30,40}-day holds locates the optima: <b>CSI500 peaks at a "
            f"{_hold('csi500')}-day hold</b> (<b>{_num(m5.get('AER'))}</b> net AER, IR {_num(m5.get('IR'),3)}, "
            f"turnover {m5.get('turnover',0):.0f}×) — an interior optimum, since gross excess decays beyond it; "
            f"<b>CSI300 keeps improving out to {_hold('csi300')} days</b> (<b>{_num(m3.get('AER'))}</b>, IR "
            f"{_num(m3.get('IR'),3)}, turnover {m3.get('turnover',0):.0f}×), i.e. it is cost-dominated and prefers "
            "low frequency. Gross excess falls with the longer hold (the alpha is short-horizon) but cost falls "
            "faster up to the optimum, so net wins. <b>EWM smoothing was rejected</b> (it shaved gross faster than "
            "it saved cost once the hold handled turnover); the rank band helped only marginally (CSI500). The "
            "chain now closes end-to-end: positive IC → positive <i>gross</i> excess → cost is the killer → "
            "longer holding → <b>positive net excess return over the index</b>. Caveat: a 25–40-day hold is a "
            "lower-frequency strategy than the paper's 5-day design — the honest tradeoff for a cost-dominated "
            "signal. The next refinement, <code>portfolio_v5</code>, bakes the cost model into the evolution "
            "objective so the search itself prefers tradeable (net-positive) factors.</p>")

    # --- Result 10 — beating the baseline AND the index (residual stack + optimal hold) ---
    beat_html = ""
    if beat:
        def _br(uni, tag):
            return beat.get(uni, {}).get("holds", {}).get(tag, {})

        def _beat_tbl(uni):
            oh = beat.get(uni, {}).get("opt_hold", "?")
            rows = []
            for arm in ("baseline", "augmented", "stack"):
                h5 = _br(uni, "h5").get(arm, {})
                op = _br(uni, "opt").get(arm, {})
                rows.append((arm, f'{h5.get("cum_excess", 0):.3f}',
                             f'<b>{op.get("cum_excess", 0):.3f}</b>' + (' ✓' if op.get("beats_index") else ''),
                             _num(op.get("AER")), _num(op.get("IR"), 3), f'{op.get("turnover", 0):.0f}×'))
            return oh, _table([f"{uni.upper()} arm", "cum-excess 5d", f"cum-excess {oh}d", "AER", "IR", "turnover"], rows)

        oh3, tbl3 = _beat_tbl("csi300")
        oh5, tbl5 = _beat_tbl("csi500")
        b3 = {a: _br("csi300", "opt").get(a, {}).get("cum_excess", 0) for a in ("baseline", "augmented", "stack")}
        b5 = {a: _br("csi500", "opt").get(a, {}).get("cum_excess", 0) for a in ("baseline", "augmented", "stack")}
        beat_html = (
            "<h2>Result 10 — beating the baseline <i>and</i> the index (cumulative excess)</h2>"
            "<p>Result 2 showed the augmented model's cumulative excess return barely exceeds the Alpha158 "
            "baseline and trails the index (curve below 1.0). Two reasons: the evolved OHLCV factors are largely "
            "<b>spanned by Alpha158</b> (their marginal IC vs the 128 baseline features is ≈0 on CSI300, +0.0008 "
            "on CSI500), and the 5-day book is <b>cost-dominated</b> (Result 9). Two fixes: <b>residual "
            "stacking</b> — train Alpha158 first, then a small FE model on <i>only</i> the baseline's residual, so "
            "the factors can add but not dilute — and the <b>Result-9 optimal hold</b>. Final cumulative excess "
            "(portfolio ÷ index; &gt;1.0 beats the index):</p>"
            f"{tbl3}{tbl5}"
            "<p class=\"small\"><b>Findings.</b> (1) At the optimal hold <b>every arm beats the index</b> "
            f"(CSI300 ≈<b>{max(b3.values()):.3f}</b>, CSI500 ≈<b>{max(b5.values()):.3f}</b>) — the holding period, "
            "not the factors, is what clears the index; at 5 days all three trail it (&lt;1.0). (2) <b>Residual "
            f"stacking beats the baseline on CSI500</b> (cum-excess {b5['baseline']:.3f}→<b>{b5['stack']:.3f}</b>, "
            "IR 0.41→0.46) — it harvests the small orthogonal mid-cap alpha. (3) On <b>CSI300 the baseline is "
            f"unbeaten</b> ({b3['baseline']:.3f} &gt; stack {b3['stack']:.3f}): Alpha158 is <b>near-efficient for "
            "OHLCV large-caps</b>, so same-modality factors only dilute it. Beating the baseline there needs a "
            "<b>different data modality</b> (intraday/overnight split, Amihud illiquidity, Garman–Klass vol, or "
            "non-OHLCV fundamentals) — exactly the families the engine's last run gravitated toward "
            "(<code>gk_lowvol</code>, <code>overnight_intraday</code>, <code>amihud_reversal</code>). The honest "
            "ceiling: with OHLCV-only data, the marginal value of evolved factors over Alpha158 is small and "
            "mid-cap-specific.</p>")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FactorEngine — Reproduction Report</title><style>{CSS}</style></head><body>
<h1>FactorEngine (FE) — Reproduction Report</h1>
<p class="sub">Reproduction of <i>FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework</i>
(arXiv:2603.16365) · generated by Claude Code with the quant-paper-reproduction skill</p>
<p class="sub">Run: <b>{es.get('panel','—')}</b> · {n_iters_cfg} iterations · {agent} ({llm_provider}/{llm_model},
n_llm={n_llm}) · {DATASRC.split('(')[0].strip()} · runtime {runtime_s/3600:.1f}h</p>

	<h2>Executive takeaway</h2>
	{exec_takeaway}

<h2>TL;DR</h2>
{hero}
<div class="box ok"><b>What reproduces (on {DATASRC}).</b> FE's macro–micro co-evolution engine — a UCT
program tree, macro code mutations, and Optuna Bayesian micro-search across islands — evolves the paper's
near-useless seed factor (validation fitness {seed_fit:+.3f}) into one with <b>{best_fit/seed_fit:.1f}× higher
validation fitness</b> ({best_fit:+.3f}), driven by {llm_gap}, rediscovering the kind of refinements the paper
highlights (<b>{transforms}</b>). A faithful <b>Alpha158-128</b> baseline (no qlib package) reaches CSI300
model IC {_num(md['baseline']['IC'])} (vs the paper's Alpha158 0.0299), and the pipeline runs the paper's exact
backtest. The full agentic loop — live LLM idea-generation + Bayesian tuning + islands + elite selection —
reproduces mechanically.{transfer}</div>
<div class="box warn"><b>What does not (and why).</b> {overfit_para}
<br><br>Separately, absolute levels trail paper Table&nbsp;1 for documented reasons: <b>(1)</b> our pure-pandas
<b>Alpha158 reimplements 128 of 158</b> features on a noisier free community bundle (baseline
{_num(md['baseline']['IC'])} ≈1.9× under the paper's 0.0299); <b>(2)</b> {es['n_evals']} evals at the run's
300-iteration budget (between the paper's 200/400 settings, but not the full 400-run grid); <b>(3)</b> {llm_sub}; <b>(4)</b> we run the <b>FE-alpha</b> variant
(report-bootstrapping out of scope), so the fair target is FE-alpha-2 (0.0315/0.0417), not FE-report-2.</div>

<h2>Run diagnostics</h2>
{diag_table}

<h2>Method &amp; pipeline</h2>
<p>Factors are Turing-complete Polars programs under a fixed I/O contract
<code>factor(pricing_data, parameters) → [instrument, datetime, Factor]</code>. The engine runs the paper's
four-stage loop per iteration: <b>(1) Program Selection</b> by UCT (Eq.1, c=√2) over a program tree;
<b>(2) Idea Generation</b> — a macro code mutation (turnover weighting, mid-price centering,
rank-normalization, EWM smoothing) conditioned on a chain-of-experience; <b>(3) Implementation</b> — Optuna
TPE Bayesian search over declared parameter ranges, maximizing <code>combined_score</code> (IC/ICIR
aggregated across lags 1/3/5/10) on validation; <b>(4) Analysis</b> — instantiate the child and
backpropagate Q/N. Two islands migrate top-3 programs every 7 iterations. Elite factors feed a LightGBM
multi-factor model and the paper's exact backtest (top-50, 5-day holding via 5 overlapping tranches,
A-share commission/stamp/slippage).</p>
<p class="small">Mining ran on the <code>{es['panel']}</code> panel: {es['n_evals']} factor evaluations,
{es['n_nodes']} programs, {es['elapsed_s']}s. Discovered parameters: {params}.</p>

<h2>Result 1 — the evolution engine improves the factor</h2>
<div class="fig">{_b64(f_conv)}</div>
<p class="small">{frontier_note}</p>
{frontier_table}
<div class="fig">{_b64(f_head)}</div>
<p>Single-factor out-of-sample (test) metrics — selection used the <b>validation</b> split only, so test is a
true hold-out. The FE-evolved factor beats the seed out-of-sample on CSI500 ({_num(sf5_oos)}); on CSI300 its
single-factor edge is marginal ({_num(sf['fe_engine_evolved']['IC'])}). The paper's own evolved artifact
(Listing 1.4), transferred verbatim, is shown for reference.</p>
{sf_table}
<div class="fig">{_b64(f_div)}</div>

	<h2>Result 2 — multi-factor integration and implementation cost</h2>
<p>The augmented model merges the <b>top-{u3.get('config', {}).get('n_elite_factors', 1)} elite evolved
factors</b> (fitness &gt; 0.4, paper §4.3) with the baseline set — a multi-factor model, not a single factor.</p>
{mb_table}
<p class="small">We skip the report-bootstrapping module (no report corpus), so this is the paper's
<b>FE-alpha</b> variant — <b>FE-alpha-2</b> is the fair comparison; <i>FE-report-2</i> is the report-seeded
ceiling that needs the corpus. {result2_note}</p>
<div class="fig">{_b64(eq_figs.get('csi300'))}</div>
<div class="fig">{_b64(eq_figs.get('csi500'))}</div>

{excess_plan_html}

<h2>Parameter contract warning</h2>
<p>A constructive insight, not a fatal flaw: the macro-agent learned useful structure but left some of its own
knobs at hardcoded defaults that never entered the Bayesian micro-search.</p>
{pc_table}
<p class="small">{pc_note}</p>

<h2>Result 3 — alpha decay / year-by-year transfer (Fig. 4 analog)</h2>
<p>Adding the elite factors helps in some years and hurts in others — the instability behind the overfit. Counts
of years where the augmented model's IC beat the Alpha158-128 baseline:</p>
{overfit_table}
<div class="fig">{_b64(yic_figs.get('csi300'))}</div>
<div class="fig">{_b64(yic_figs.get('csi500'))}</div>

{v3result_html}

{ortho_html}

{v3_456_html}

{excess_html}

{v3v4_html}

{sweep_html}

{beat_html}

<h2>Ablations</h2>
{abl_html or '<p>(not run)</p>'}

<h2>Robustness across data realizations</h2>
{rob_html}

<h2>Findings beyond the paper</h2>
<ul>
{findings_lead}
<li><b>The seed and the paper's own evolved artifact are ~zero/negative on the real 2017–24 test.</b> Seed
CSI300 test IC {_num(sf['seed']['IC'])}, artifact {_num(sf['paper_evolved_artifact']['IC'])} — the paper itself
flags 2017–21 as a hard period for A-share cross-sectional factors (App. A.2). So the seed is genuinely weak;
the <i>engine</i> re-run per dataset, not the transferred artifact, is what matters.</li>
<li><b>Turnover and volatility scaling are the load-bearing edits in this live Kimi run.</b> The best saved
factor is <code>llm + turnover + volscale</code>; EWM/rank-normalization remain useful in ablations and
deterministic controls, but they are not the current live-run headline transform.</li>
<li><b>RankIC ≫ Pearson IC in stability</b> for these turnover-weighted factors; reporting RankIC and the
rank-inclusive fitness (Eq.5) is the robust choice.</li>
</ul>

<h2>Implemented V3 protocol</h2>
<p>The run shows the bottleneck is <b>selection/integration</b>, not LLM connectivity.
{"<b>All six items are now demonstrated on the existing run</b> (Results 4–6): robust selection repairs the CSI300 IC degradation (#1–#2), orthogonal residual alpha remains positive on CSI500 but must be gated by universe (#3), and the portfolio objective / prompt-parser contract / plateau stop are exercised in Result 6 (#4–#6) — no LLM re-run needed." if v3_456_html else "<b>V3 status artifacts are incomplete for this build.</b> Run scripts/06_robust_elite.py, scripts/07_orthogonal_elite.py, and scripts/08_v3_4to6.py before rebuilding the report."}</p>
{v3_html}

<h2>Reproduction plan status</h2>
<p>The current reproduction plan is embedded below so the report carries both the empirical results and the
phase-by-phase execution map. This appendix is generated from <code>REPRODUCTION_PLAN.md</code> at report
build time.</p>
<div class="plan-appendix">
{plan_appendix}
</div>

<h2>Conclusion</h2>
<p>On {DATASRC}, FactorEngine's <b>machinery</b> reproduces faithfully — program-level macro mutation separated
from Bayesian micro-optimization, a UCT tree with multi-island diversity, elite selection, and (this run) a
<b>live Kimi LLM</b> agent driving idea-generation ({n_llm} accepted mutations) — and it reliably evolves a
near-useless seed into a high-<i>validation</i>-fitness factor ({best_fit/seed_fit:.1f}×). The honest result is
that {conclusion_verdict} {closed_loop_note}{result6_note}This is exactly the kind of finding a faithful reproduction should surface: the
method runs end-to-end, and V3 shows the practical discipline needed to separate a high-validation factor from a
durable one: robust elite selection, orthogonalized integration, portfolio-aware scoring, parser-enforced
parameters, and plateau-aware compute.</p>
<p class="small">Substitutions vs the paper: pure-pandas <b>Alpha158-128</b> vs full Alpha158 (exact set needs
the qlib package); {llm_sub}; 300-iteration budget (between the paper's 200/400 settings, not the full 400-run grid); report-bootstrapping (FE-report) out of
scope → FE-alpha variant. Data IS real (Qlib <code>.bin</code> bundle, 2008–2024).
See REPRODUCTION_PLAN.md / REAL_DATA.md.</p>
</body></html>"""

    out = outputs / "reproduction_report.html"
    out.write_text(html)
    return out
