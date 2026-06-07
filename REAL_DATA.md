# Running on real Qlib A-share data (paper-faithful)

This environment can't host Qlib (no C/Cython compiler; `pyqlib` won't build) or
the CN data bundle (China endpoints are network-blocked here), so the runs in
`outputs/` use the synthetic panel. The code below is the **turnkey path on a
machine that has Qlib + the data** — it auto-detects `~/.qlib/qlib_data/cn_data`
and uses the exact Alpha158 set, the Qlib label, the real index benchmark, and
point-in-time CSI300/500 membership.

## 0. Set up env + get the data (no qlib package needed)
```bash
# 1) environment (~5 min)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # macOS LightGBM also needs: brew install libomp

# 2) download a prebuilt Qlib CN .bin bundle (~1.5 GB) -> ~/.qlib/qlib_data/cn_data
python data/setup_qlib.py                     # default: chenditc/investment_data (current)
#   custom mirror/path: QLIB_DATA_URL=<url> python data/setup_qlib.py --target /data/cn_data

# 3) verify it loaded (pure-NumPy reader; no qlib import)
python tests/test_data.py
```
The reproduction reads Qlib's `.bin` format directly (`fe/data/qlib_bin.py`), so the
**`qlib` package is NOT required**. If you prefer the official Microsoft bundle (ends
~2020) or to build current data from a vendor (AkShare/baostock/Tushare → `dump_bin.py`),
point `--target`/`QLIB_DATA_URL` at it — anything in standard Qlib layout works.

## 1–5. The pipeline (real data)
```bash
# Set PY to a REAL interpreter path (don't leave it unset — an empty $PY makes
# bash try to exec the .py directly => "Permission denied"). In the base conda
# env, plain `python` also works, and the scripts are chmod +x (shebang), so
# `scripts/01_build_data.py ...` runs directly too.
export PY=/Users/difeisu/miniconda3/bin/python   # interpreter with the repo deps + pyqlib

# 1) build real panels (qlib_loader picks up ~/.qlib/qlib_data/cn_data; falls back
#    to synthetic if the field/symbol is missing)
$PY scripts/01_build_data.py --market csi300 --profiles csi300
$PY scripts/01_build_data.py --market csi500 --profiles csi500

# 2) evolve from the seed on real data (FE-alpha). 200/400 iters = the paper's budget.
$PY scripts/02_run_evolution.py --panel csi300 --iterations 200 --islands 2 --trials 18
#    Live Kimi/Moonshot macro-agent (China platform):
#      export FE_LLM_PROVIDER=kimi-cn
#      export FE_LLM_BASE_URL=https://api.moonshot.cn/v1
#      export FE_LLM_API_KEY=sk-...   FE_LLM_MODEL=kimi-k2.6   FE_LLM_TIMEOUT=120
#      unset FE_LLM_PROXY        # direct HTTPS is preferred when located in China
#      # optional only if your network requires an explicit HTTPS proxy:
#      # export FE_LLM_PROXY=https://127.0.0.1:7890
#      # TEST THE SAME PATH THE ENGINE USES:
#      $PY scripts/check_llm.py
#      FE_REQUIRE_LLM=1 $PY scripts/02_run_evolution.py --panel csi300 --iterations 200 --use-llm

# 3) evaluate paper-faithful: exact Alpha158 + Qlib LABEL0 + real index benchmark
$PY scripts/04_evaluate.py --baseline alpha158 --label label --use-benchmark

# 4) ablations + 5) report
$PY scripts/03_ablations.py
$PY scripts/05_build_report.py    # outputs/reproduction_report.{html,pdf}
```

## What each flag maps to
| Flag / file | Paper element |
|---|---|
| `--market csi300/csi500` (01) | universe via `D.instruments(market)` — point-in-time membership |
| `--baseline alpha158` (04) | exact Qlib Alpha158 set merged with evolved factors (`fe/integration/alpha158_qlib.py`) |
| `--label label` (04) | Qlib Alpha158 `LABEL0` = `Ref($close,-2)/Ref($close,-1)-1` |
| `--use-benchmark` (04) | excess returns vs the real CSI300 (`SH000300`) / CSI500 (`SH000905`) index |
| `FE_LLM_*` + `--use-llm` (02) | LLM idea-generation (paper: Gemini-2.5-Pro; here: Kimi/Moonshot China, any OpenAI-compatible endpoint, or Anthropic) |

## Honest limits even with real data
- **FE-report** also needs a corpus of **pre-2017 Chinese research reports** for the
  bootstrapping module — proprietary, not redistributable. **FE-alpha** (seeded from
  the Alpha158 factors) needs no reports and is fully runnable above.
- **Bit-exact numbers are unattainable**: LLM sampling is non-deterministic (the paper
  itself notes trajectories diverge under identical configs), and the exact Gemini
  snapshot/report corpus aren't reproducible. Target = faithful on the same
  universe/period/split, not identical decimals.
