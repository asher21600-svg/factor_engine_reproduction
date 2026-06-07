#!/usr/bin/env python
"""Build a separate draft-v2 reproduction report.

This script intentionally does not call fe.report.build_report.build_html(),
because that function writes to outputs/reproduction_report.html.  The user
asked to leave the current HTML/PDF untouched, so this draft writes only:

  outputs/reproduction_report_v2_draft.html
  outputs/reproduction_report_v2_draft.pdf
"""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from fe import config
from fe.report.build_pdf import html_to_pdf


OUT_HTML = "reproduction_report_v2_draft.html"
OUT_PDF = "reproduction_report_v2_draft.pdf"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _num(x, n: int = 4) -> str:
    try:
        return f"{float(x):+.{n}f}"
    except Exception:  # noqa: BLE001
        return "-"


def _plain_num(x, n: int = 2) -> str:
    try:
        return f"{float(x):.{n}f}"
    except Exception:  # noqa: BLE001
        return "-"


def _pct(x, n: int = 2) -> str:
    try:
        return f"{100 * float(x):+.{n}f}%"
    except Exception:  # noqa: BLE001
        return "-"


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _b64_img(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="missing">Missing figure: <code>{_esc(path)}</code></p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{_esc(alt)}">'
        f"<figcaption>{_esc(alt)}</figcaption></figure>"
    )


def _table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    klass = f' class="{cls}"' if cls else ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table{klass}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _history_stats(history: list) -> dict:
    points: list[tuple[int, float]] = []
    for item in history:
        if isinstance(item, dict):
            i = item.get("iter")
            v = item.get("best_reward", item.get("reward"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            i, v = item[0], item[1]
        else:
            continue
        try:
            points.append((int(i), float(v)))
        except Exception:  # noqa: BLE001
            continue
    if not points:
        return {"n": 0, "best": None, "first_best_iter": None, "records": []}
    best = max(v for _, v in points)
    first = next(i for i, v in points if abs(v - best) < 1e-12)
    records: list[tuple[int, float]] = []
    current = float("-inf")
    for i, v in points:
        if v > current + 1e-12:
            records.append((i, v))
            current = v
    return {"n": len(points), "best": best, "first_best_iter": first, "records": records}


def _parameter_contract(evo: dict) -> tuple[list[str], list[str], list[str]]:
    code = evo.get("best_code", "") or ""
    used = sorted(set(re.findall(r"parameters\.get\(['\"]([^'\"]+)['\"]", code)))
    declared = sorted((evo.get("best_params") or {}).keys())
    missing = sorted(set(used) - set(declared) - {"epsilon"})
    extra = sorted(set(declared) - set(used))
    return used, missing, extra


def _metric_delta(a: float, b: float, n: int = 4) -> str:
    return _num(float(a) - float(b), n)


def _runtime(elapsed_s: float | int | None) -> str:
    try:
        seconds = float(elapsed_s)
    except Exception:  # noqa: BLE001
        return "-"
    hours = seconds / 3600
    return f"{hours:.1f}h ({seconds:,.0f}s)"


def _top_feature_list(features: list) -> str:
    if not features:
        return "-"
    return ", ".join(f"{_esc(k)}:{_esc(v)}" for k, v in features[:6])


def build(outputs: Path | None = None) -> tuple[Path, Path | None]:
    outputs = Path(outputs or config.OUTPUTS)
    figs = outputs / "figures"
    results = _read_json(outputs / "results.json")
    evo = _read_json(outputs / "evolution.json")
    abl = _read_json(outputs / "ablations.json") if (outputs / "ablations.json").exists() else {}

    universes = results["universes"]
    u300 = universes["csi300"]
    u500 = universes["csi500"]
    es = results["evolution_summary"]
    hist = _history_stats(evo.get("history") or [])
    iterations = hist["n"] or 300
    n_llm = int(evo.get("n_llm", 0) or 0)
    llm_rate = n_llm / iterations if iterations else 0
    best_iter = hist["first_best_iter"] or "-"
    plateau = (iterations - int(best_iter)) if isinstance(best_iter, int) else "-"
    used_params, missing_params, extra_params = _parameter_contract(evo)

    def model_delta(uni: str, metric: str) -> str:
        m = universes[uni]["model"]
        return _metric_delta(m["augmented"][metric], m["baseline"][metric])

    def backtest_delta(uni: str, metric: str) -> str:
        b = universes[uni]["backtest"]
        if metric in {"AR", "AER", "MDD", "RMDD", "ann_excess", "ann_turnover", "ann_cost"}:
            return _pct(float(b["augmented"][metric]) - float(b["baseline"][metric]))
        return _num(float(b["augmented"][metric]) - float(b["baseline"][metric]), 3)

    seed_fit = float(es["seed_fitness"])
    best_fit = float(es["best_fitness"])
    fitness_lift = best_fit / seed_fit if seed_fit else float("nan")
    transforms = ", ".join(f"<code>{_esc(t)}</code>" for t in es.get("best_transforms", []))
    declared_params = ", ".join(
        f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in (evo.get("best_params") or {}).items()
    )

    llm_meta = evo.get("llm") or {}
    last_success = llm_meta.get("last_success") or {}
    llm_provider = last_success.get("provider") or llm_meta.get("provider") or "kimi-cn"
    llm_model = last_success.get("model") or llm_meta.get("model") or "kimi-k2.6"
    llm_base = (
        last_success.get("base_url")
        or ", ".join(llm_meta.get("base_urls") or [])
        or "https://api.moonshot.cn/v1"
    )
    llm_temperature = last_success.get("temperature", llm_meta.get("temperature", "1"))

    csi300_model = u300["model"]
    csi500_model = u500["model"]
    csi300_bt = u300["backtest"]
    csi500_bt = u500["backtest"]
    csi500_single = u500["single_factor"]["fe_engine_evolved"]["IC"]

    hero_rows = [
        ["Validation lift", f"{fitness_lift:.1f}x", "seed fitness to best evolved fitness"],
        ["Live Kimi accepted", f"{n_llm}/{iterations}", f"{100 * llm_rate:.1f}% of evolution iterations"],
        ["Best frontier reached", f"iter {best_iter}", f"{plateau} later iterations did not improve best reward"],
        ["CSI300 model IC delta", model_delta("csi300", "IC"), "augmented minus baseline on 2017-2024 test"],
        ["CSI500 model IC delta", model_delta("csi500", "IC"), "small IC lift, but backtest quality worsened"],
        ["CSI500 single-factor IC", _num(csi500_single), "standalone FE-evolved signal"],
    ]

    diag_rows = [
        ["Panel evolved", _esc(es.get("panel", "-")), "Evolution ran on CSI300 and transferred to CSI500."],
        ["Iterations", str(iterations), "Requested live macro-agent run."],
        ["Accepted live LLM mutations", str(n_llm), "Connectivity worked; fallback handled transient/unparseable calls."],
        ["Provider / model", f"<code>{_esc(llm_provider)}</code> / <code>{_esc(llm_model)}</code>", "Moonshot China path."],
        ["Endpoint", f"<code>{_esc(llm_base)}</code>", "Direct HTTPS route used by the evolution call path."],
        ["Temperature", f"<code>{_esc(llm_temperature)}</code>", "Model-specific Moonshot constraint."],
        ["Runtime", _runtime(es.get("elapsed_s")), f"{_plain_num(float(es.get('elapsed_s', 0)) / max(iterations, 1), 1)}s per iteration."],
        ["Evaluations / nodes", f"{_esc(es.get('n_evals'))} / {_esc(es.get('n_nodes'))}", "Micro-search and tree size."],
        ["Best transforms", transforms or "-", "Current live run: LLM idea plus turnover and volatility scaling."],
        ["Declared tuned params", declared_params or "-", "Only these entered Bayesian search."],
    ]

    single_rows: list[list[str]] = []
    for uni, U in universes.items():
        sf = U["single_factor"]
        for key, label in [
            ("seed", "Seed"),
            ("paper_evolved_artifact", "Paper artifact"),
            ("fe_engine_evolved", "FE-evolved"),
        ]:
            r = sf[key]
            mark = ' <span class="pill good">best standalone</span>' if key == "fe_engine_evolved" and uni == "csi500" else ""
            single_rows.append(
                [
                    f"{uni.upper()} - {label}{mark}",
                    _num(r["IC"]),
                    _num(r["ICIR"], 3),
                    _num(r["RIC"]),
                    _num(r["RICIR"], 3),
                    _num(r["fitness"], 3),
                ]
            )

    model_rows: list[list[str]] = []
    for uni, U in universes.items():
        for key, label in [("baseline", "Alpha158-128"), ("gplearn", "GPLearn arm"), ("augmented", "+ FE elites")]:
            m = U["model"][key]
            b = U["backtest"][key]
            label_html = f"{uni.upper()} - {label}"
            if uni == "csi500" and key == "augmented":
                label_html += ' <span class="pill note">IC winner, not portfolio winner</span>'
            if uni == "csi300" and key == "gplearn":
                label_html += ' <span class="pill good">IC winner</span>'
            model_rows.append(
                [
                    label_html,
                    _num(m["IC"]),
                    _num(m["RIC"]),
                    _pct(b["AR"]),
                    f"{float(b['SR']):+.2f}",
                    _pct(b["MDD"]),
                    str(m.get("n_features", "-")),
                    _top_feature_list(m.get("top_features") or []),
                ]
            )

    delta_rows = [
        ["CSI300", model_delta("csi300", "IC"), model_delta("csi300", "RIC"), backtest_delta("csi300", "AR"), backtest_delta("csi300", "SR"), backtest_delta("csi300", "ann_cost")],
        ["CSI500", model_delta("csi500", "IC"), model_delta("csi500", "RIC"), backtest_delta("csi500", "AR"), backtest_delta("csi500", "SR"), backtest_delta("csi500", "ann_cost")],
    ]

    yearly_rows: list[list[str]] = []
    for uni, U in universes.items():
        yb = U.get("yearly_ic", {}).get("baseline", {})
        ya = U.get("yearly_ic", {}).get("augmented", {})
        years = sorted(set(yb) | set(ya))
        worse = sum(1 for y in years if ya.get(y, {}).get("ic", 0) < yb.get(y, {}).get("ic", 0))
        better = len(years) - worse
        worst_year = None
        worst_delta = 0.0
        for y in years:
            d = float(ya.get(y, {}).get("ic", 0)) - float(yb.get(y, {}).get("ic", 0))
            if worst_year is None or d < worst_delta:
                worst_year, worst_delta = y, d
        yearly_rows.append([uni.upper(), f"{better}/{len(years)} years better", f"{worse}/{len(years)} years worse", _esc(worst_year), _num(worst_delta)])

    record_rows = [[str(i), _num(v)] for i, v in (hist.get("records") or [])]

    bayes = abl.get("bayes_ablation") or {}
    island = abl.get("island_ablation") or {}
    bayes_rows = []
    if bayes:
        for key, label in [("with_bayes", "Bayesian ON"), ("without_bayes", "Bayesian OFF")]:
            d = bayes[key]
            bayes_rows.append([label, _num(d["best_reward"], 3), _num(d["best_fitness"], 3), ", ".join(_esc(t) for t in d.get("best_transforms", []))])
    island_rows = []
    if island:
        for key, d in island.items():
            t = d.get("csi300_test", {})
            island_rows.append([f"{key} island(s)", _num(d.get("best_fitness_mining"), 3), _num(t.get("RIC")), _num(t.get("fitness"), 3), ", ".join(_esc(x) for x in d.get("best_transforms", [])[:5])])

    missing_html = ", ".join(f"<code>{_esc(k)}</code>" for k in missing_params) or "None"
    used_html = ", ".join(f"<code>{_esc(k)}</code>" for k in used_params) or "None"
    extra_html = ", ".join(f"<code>{_esc(k)}</code>" for k in extra_params) or "None"

    CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0;background:#f5f6f8;color:#172033;line-height:1.55}
.page{max-width:1080px;margin:0 auto;background:#fff;min-height:100vh;padding:34px 42px 56px;box-shadow:0 0 0 1px #e6e8ee}
h1{font-size:30px;line-height:1.15;margin:0 0 6px}
h2{font-size:21px;margin:34px 0 12px;padding-bottom:6px;border-bottom:2px solid #e7ebf2}
h3{font-size:16px;margin:22px 0 8px}.sub{color:#687386;margin:0 0 22px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #dfe5ee;border-radius:8px;padding:14px;background:#fbfcfe}.card .k{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#647184}
.card .v{font-size:25px;font-weight:750;margin:3px 0;color:#0e5a8a}.card .d{font-size:12px;color:#667085}
.box{border:1px solid #dfe5ee;background:#f8fafc;border-radius:8px;padding:14px 16px;margin:14px 0}.box.good{background:#eff8f1;border-color:#bfe2c7}.box.warn{background:#fff5e6;border-color:#efcf95}.box.bad{background:#fff1f1;border-color:#efb4b4}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 20px}th,td{border:1px solid #e0e5ec;padding:7px 9px;text-align:right;vertical-align:top}th{background:#f1f4f8;text-align:center;font-weight:650}td:first-child,th:first-child{text-align:left}.wide td:last-child{text-align:left}
code{background:#eef2f6;border-radius:4px;padding:1px 5px;font-size:12px}.pill{display:inline-block;border-radius:999px;padding:1px 7px;font-size:11px;margin-left:5px;border:1px solid #cfd7e3}.pill.good{background:#edf8ef;border-color:#b8dfc1}.pill.note{background:#eef6ff;border-color:#b8d8f0}
figure{margin:16px 0;text-align:center}figure img{max-width:100%;height:auto;border:1px solid #e6e9ef;border-radius:6px;background:#fff}figcaption{font-size:12px;color:#687386;margin-top:5px}.small{font-size:12px;color:#697386}.callout-title{font-weight:700;margin-bottom:5px}ul{padding-left:22px}li{margin:7px 0}
@media(max-width:760px){.page{padding:24px 18px}.grid{grid-template-columns:1fr}table{font-size:12px}}
"""

    cards = "\n".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div><div class="d">{_esc(d)}</div></div>'
        for k, v, d in hero_rows
    )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FactorEngine Reproduction Report - Draft V2</title>
  <style>{CSS}</style>
</head>
<body>
<main class="page">
  <h1>FactorEngine Reproduction Report - Draft V2</h1>
  <p class="sub">A sharper interpretation of the live Kimi/Moonshot 300-iteration run on real CSI300/CSI500 A-share data. This is a separate draft artifact and does not replace the current report.</p>

  <section>
    <h2>Executive Takeaway</h2>
    <div class="grid">{cards}</div>
    <div class="box good">
      <div class="callout-title">What is solid</div>
      The live macro-agent path is now real, not simulated: <b>{n_llm}</b> accepted Kimi/Moonshot mutations were applied through the same path used by evolution. The engine lifted validation fitness from <b>{_num(seed_fit, 3)}</b> to <b>{_num(best_fit, 3)}</b>, a <b>{fitness_lift:.1f}x</b> improvement, with a best factor built around {transforms}.
    </div>
    <div class="box bad">
      <div class="callout-title">What the current report should say more carefully</div>
      The augmented model is not a clean out-of-sample win. On CSI300 it lowers model IC from <b>{_num(csi300_model["baseline"]["IC"])}</b> to <b>{_num(csi300_model["augmented"]["IC"])}</b>. On CSI500 it slightly raises model IC from <b>{_num(csi500_model["baseline"]["IC"])}</b> to <b>{_num(csi500_model["augmented"]["IC"])}</b>, but annual return and Sharpe deteriorate versus baseline. The right conclusion is not "Kimi failed"; it is "validation-only elite selection overfit, and the portfolio layer did not convert standalone signal into robust PnL."
    </div>
  </section>

  <section>
    <h2>Run Diagnostics</h2>
    {_table(["Item", "Value", "Interpretation"], diag_rows, "wide")}
    <p class="small">The best validation frontier was reached by iteration <b>{best_iter}</b>. The remaining <b>{plateau}</b> iterations explored variants without improving the global best reward, so future runs should add early stopping or switch objective once the frontier plateaus.</p>
    {_table(["Record iteration", "Best reward"], record_rows)}
  </section>

  <section>
    <h2>Result 1 - Single-Factor Mining</h2>
    <p>The evolution loop does produce a better factor, especially on CSI500. This matters because it separates the factor-mining result from the later model-combination result.</p>
    {_table(["Universe / factor", "IC", "ICIR", "RankIC", "RankICIR", "Fitness"], single_rows)}
    {_b64_img(figs / "headline_ic.png", "Single-factor IC comparison")}
    {_b64_img(figs / "convergence.png", "Validation convergence")}
  </section>

  <section>
    <h2>Result 2 - Multi-Factor Integration Is The Weak Link</h2>
    <p>The augmented model merges elite FE factors with Alpha158-128 features. The corrected reading is mixed: CSI300 gets worse; CSI500 model IC improves slightly, but the backtest does not improve versus the baseline. This points to integration and selection, not raw LLM connectivity, as the next bottleneck.</p>
    {_table(["Universe / model", "IC", "RankIC", "AR", "SR", "|MDD|", "Features", "Top features"], model_rows, "wide")}
    {_table(["Universe", "Delta IC", "Delta RankIC", "Delta AR", "Delta SR", "Delta annual cost"], delta_rows)}
    {_b64_img(figs / "equity_csi300.png", "CSI300 equity curve: baseline vs augmented")}
    {_b64_img(figs / "equity_csi500.png", "CSI500 equity curve: baseline vs augmented")}
  </section>

  <section>
    <h2>Where The Overfit Shows Up</h2>
    <p>Validation lift is large, but the test behavior is uneven by year and by universe. The CSI500 standalone factor has signal, while the elite bundle and model weighting are fragile.</p>
    {_table(["Universe", "Augmented IC better", "Augmented IC worse", "Worst year", "Worst IC delta"], yearly_rows)}
    {_b64_img(figs / "yic_csi300.png", "CSI300 yearly IC")}
    {_b64_img(figs / "yic_csi500.png", "CSI500 yearly IC")}
  </section>

  <section>
    <h2>Parameter Contract Warning</h2>
    <div class="box warn">
      <div class="callout-title">Kimi introduced knobs outside the Bayesian search contract</div>
      The best code reads these parameters: {used_html}. The saved Bayesian-tuned params are: {declared_params or "None"}. Undeclared runtime defaults are: {missing_html}. Extra declared-but-unused params are: {extra_html}.
    </div>
    <p>This is a constructive bug/insight rather than a fatal flaw. The macro-agent learned useful structure, but some choices stayed as hard-coded defaults. For the next run, every <code>parameters.get(...)</code> key should be auto-scanned and either forced into <code>###Parameters</code> or rejected before evaluation.</p>
  </section>

  <section>
    <h2>Ablation Reading</h2>
    <p>Bayesian search still appears load-bearing, but the island result warns that more diversity can increase validation fitness while harming transfer if selection remains validation-only.</p>
    {_table(["Control", "Best reward", "Best fitness", "Best transforms"], bayes_rows, "wide") if bayes_rows else "<p>No Bayesian ablation saved.</p>"}
    {_b64_img(figs / "bayes.png", "Bayesian micro-search ablation")}
    {_table(["Islands", "Mining fitness", "CSI300 test RankIC", "CSI300 test fitness", "Transforms"], island_rows, "wide") if island_rows else "<p>No island ablation saved.</p>"}
    {_b64_img(figs / "diversity.png", "Tree diversity")}
  </section>

  <section>
    <h2>Recommended V3 Protocol</h2>
    <ul>
      <li><b>Change elite selection.</b> Select factors by multi-window, cross-universe robustness: validation fitness, test-like rolling subwindows, sign consistency, and degradation penalties.</li>
      <li><b>Penalize complexity.</b> Add a penalty for parameter count, transform count, and undeclared defaults. A factor with hidden knobs should lose score until the knobs are tuned.</li>
      <li><b>Cap redundancy.</b> Reject or downweight factors highly correlated with Alpha158 features or existing elite factors. The model needs orthogonal alpha, not more versions of the same turnover/liquidity exposure.</li>
      <li><b>Make objective portfolio-aware.</b> Include turnover, cost, and yearly stability in the micro objective, not only IC/ICIR on one validation regime.</li>
      <li><b>Use plateau-aware compute.</b> This run found the frontier by iteration {best_iter}. After 30-50 stagnant iterations, switch to a new seed, new split, or robustness objective instead of spending another 200 iterations on the same frontier.</li>
      <li><b>Tighten the LLM prompt and parser.</b> Require one mutation theme per response, all tunables declared in JSON ranges, no lookahead, and a short rationale tied to CSI300/CSI500 microstructure.</li>
    </ul>
  </section>

  <section>
    <h2>Draft Conclusion</h2>
    <p>The strongest defensible conclusion is: <b>the live Kimi/Moonshot macro-agent can drive FactorEngine evolution and lift validation fitness, but the current validation-only elite selection overfits.</b> The factor itself is not useless; CSI500 standalone OOS IC is <b>{_num(csi500_single)}</b>, and CSI500 augmented model IC is slightly positive versus baseline. The failure is that the selected elite bundle does not translate reliably into out-of-sample portfolio quality. The next improvement should target the selection/objective layer before increasing iteration count or model size.</p>
  </section>
</main>
</body>
</html>
"""

    html_path = outputs / OUT_HTML
    pdf_path = outputs / OUT_PDF
    html_path.write_text(doc)
    pdf = html_to_pdf(html_path, pdf_path)
    return html_path, pdf


def main() -> None:
    html_path, pdf_path = build(config.OUTPUTS)
    print(f"-> draft HTML report: {html_path}")
    if pdf_path:
        print(f"-> draft PDF report:  {pdf_path}")


if __name__ == "__main__":
    main()
