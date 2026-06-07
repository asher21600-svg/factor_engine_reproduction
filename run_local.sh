#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FactorEngine reproduction: real-data evolution with Kimi / Moonshot.
#
# Usage:
#   export FE_LLM_API_KEY='YOUR_MOONSHOT_KEY'
#   ./run_local.sh
#
# Common overrides:
#   ITERS=400 ./run_local.sh
#   FE_LLM_PROXY= ./run_local.sh                      # direct HTTPS to .cn
#   FE_LLM_PROXY=https://127.0.0.1:7890 ./run_local.sh # explicit HTTPS proxy, if needed
#   FE_LLM_BASE_URL=https://api.moonshot.ai/v1 ./run_local.sh
#   FE_LLM_MODEL=kimi-k2.6 FE_LLM_THINKING=enabled ./run_local.sh
# ---------------------------------------------------------------------------
set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

cd "$(dirname "$0")"

DEFAULT_PY="/Users/difeisu/miniconda3/bin/python"
if [[ -n "${PY:-}" ]]; then
  :
elif [[ -x "$DEFAULT_PY" ]]; then
  PY="$DEFAULT_PY"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "[ERROR] No Python interpreter found. Set PY=/path/to/python"
  exit 1
fi

ITERS="${ITERS:-100}"          # paper uses 200/400; raise for full fidelity
SEED="${SEED:-1}"

# Kimi / Moonshot China access. Official China platform uses api.moonshot.cn.
export FE_LLM_PROVIDER="${FE_LLM_PROVIDER:-kimi-cn}"
export FE_LLM_BASE_URL="${FE_LLM_BASE_URL:-https://api.moonshot.cn/v1}"
export FE_LLM_MODEL="${FE_LLM_MODEL:-kimi-k2.6}"
export FE_LLM_TIMEOUT="${FE_LLM_TIMEOUT:-120}"
export FE_LLM_TEMPERATURE="${FE_LLM_TEMPERATURE:-1}"
export FE_LLM_THINKING="${FE_LLM_THINKING:-disabled}"

# Default is strict for this runner: do not present deterministic evolution as
# a live-Kimi run. After Kimi has produced at least one accepted mutation,
# transient endpoint failures fall back for that step. Set
# FE_REQUIRE_LLM_EVERY_CALL=1 only if you want any later LLM hiccup to abort.
export FE_REQUIRE_LLM="${FE_REQUIRE_LLM:-1}"

if [[ -z "${FE_LLM_API_KEY:-}" && -n "${MOONSHOT_API_KEY:-}" ]]; then
  export FE_LLM_API_KEY="$MOONSHOT_API_KEY"
fi
if [[ -z "${FE_LLM_API_KEY:-}" ]]; then
  echo "[ERROR] Missing FE_LLM_API_KEY or MOONSHOT_API_KEY."
  echo "Run: export FE_LLM_API_KEY='YOUR_MOONSHOT_KEY'"
  exit 1
fi

# Mirror into OpenAI-compatible names for any helper library that expects them.
export OPENAI_API_KEY="$FE_LLM_API_KEY"
export OPENAI_BASE_URL="$FE_LLM_BASE_URL"

# FE_LLM_PROXY is consumed directly by scripts/check_llm.py and the evolution
# engine. Do not mirror it into HTTPS_PROXY/HTTP_PROXY here, because pip and
# data downloads would inherit it and fail when the local LLM proxy is offline.
# In mainland China, leave FE_LLM_PROXY empty and call https://api.moonshot.cn/v1
# directly.

echo "== config =="
echo "repo              : $(pwd)"
echo "python            : $PY"
echo "iterations        : $ITERS"
echo "seed              : $SEED"
echo "llm provider      : $FE_LLM_PROVIDER"
echo "llm base url      : $FE_LLM_BASE_URL"
echo "llm model         : $FE_LLM_MODEL"
echo "llm thinking      : $FE_LLM_THINKING"
echo "llm proxy         : ${FE_LLM_PROXY:-none}"
echo "system proxy      : ${FE_LLM_USE_SYSTEM_PROXY:-0}"
echo "require llm       : $FE_REQUIRE_LLM"
echo "require every llm : ${FE_REQUIRE_LLM_EVERY_CALL:-0}"
echo

FE_PIP_INDEX_URL="${FE_PIP_INDEX_URL:-https://pypi.org/simple}"
pip_install() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
      -u https_proxy -u http_proxy -u all_proxy \
      -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST \
      PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 \
      "$PY" -m pip --isolated install --index-url "$FE_PIP_INDEX_URL" "$@"
}

echo "== 0. deps (idempotent) =="
# Skip pip entirely when everything already imports — avoids needing PyPI/DNS
# (which fails under some VPN/proxy states even though the deps are installed).
FE_CORE_DEPS="numpy,pandas,scipy,sklearn,polars,optuna,lightgbm,jinja2,pyarrow,matplotlib,gplearn"
if [[ "${SKIP_DEPS:-}" == "1" ]] || "$PY" -c "import ${FE_CORE_DEPS}" >/dev/null 2>&1; then
  echo "all deps present       -> skipping pip (no network needed)"
else
  echo "pip index         : $FE_PIP_INDEX_URL"
  pip_install -q -U pip setuptools wheel
  pip_install -q --only-binary=:all: -r requirements.txt || \
    pip_install -q -r requirements.txt
fi

# macOS LightGBM needs OpenMP.
command -v brew >/dev/null && (brew list libomp >/dev/null 2>&1 || brew install libomp) || true

CERT_PATH="$("$PY" - <<'PY' 2>/dev/null || true
try:
    import certifi
    print(certifi.where())
except Exception:
    pass
PY
)"
if [[ -n "$CERT_PATH" ]]; then
  export SSL_CERT_FILE="$CERT_PATH"
  export REQUESTS_CA_BUNDLE="$CERT_PATH"
fi

echo
echo "== 1. LIVE-LLM gate: same call path used by evolution =="
set +e
"$PY" scripts/check_llm.py
LLM_STATUS=$?
set -e
if [[ "$LLM_STATUS" -eq 2 ]]; then
  USE_LLM="--use-llm --require-llm"
  export FE_LLM_GATE_PASSED=1
  echo "   -> Kimi is reachable; smoke mutation was not parseable, but evolution will retry live calls."
elif [[ "$LLM_STATUS" -ne 0 ]]; then
  echo
  echo "[ERROR] Kimi/Moonshot did not pass the live mutation gate (status $LLM_STATUS)."
  echo "Try FE_LLM_PROXY=..., FE_LLM_BASE_URL=https://api.moonshot.ai/v1, or a different FE_LLM_MODEL."
  if [[ "$FE_REQUIRE_LLM" == "1" ]]; then
    echo "FE_REQUIRE_LLM=1, so exiting instead of silently falling back."
    exit 1
  fi
  USE_LLM=""
else
  USE_LLM="--use-llm --require-llm"
  export FE_LLM_GATE_PASSED=1
  echo "   -> Kimi is reachable and parseable; evolution will run with the live agent."
fi

echo
echo "== 2. data (downloads Qlib CN bundle if absent; resumable) =="
"$PY" data/setup_qlib.py
"$PY" tests/test_data.py

echo
echo "== 3. build real panels (CSI300 + CSI500, point-in-time, winsorized) =="
"$PY" scripts/01_build_data.py --profiles csi300 csi500 --market real

echo
echo "== 4. evolve on real CSI300${USE_LLM:+ with the live Kimi LLM agent} =="
"$PY" scripts/02_run_evolution.py --panel csi300 \
      --iterations "$ITERS" --islands 2 --trials 12 --seed "$SEED" $USE_LLM

echo
echo "== 5. evaluate: Alpha158 + Qlib label + real index benchmark + GPLearn =="
"$PY" scripts/04_evaluate.py --baseline alpha158 --label label --use-benchmark --gplearn

echo
echo "== 6. ablations + report =="
"$PY" scripts/03_ablations.py || echo "(ablations optional; skipped on error)"
"$PY" scripts/05_build_report.py

echo
echo "DONE -> outputs/reproduction_report.html  (and .pdf)"
echo "Compare the CSI300/500 model IC to paper FE-alpha-2 (0.0315 / 0.0417)."
