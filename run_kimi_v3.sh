#!/usr/bin/env bash
# One-command live-Kimi V3 evolution re-run (gate -> evolve), self-contained so you
# don't have to re-export env vars in every new Terminal window.
#
# Usage (set the key once per shell, or put it in your profile / a gitignored .env):
#   export FE_LLM_API_KEY='YOUR_MOONSHOT_KEY'
#   ./run_kimi_v3.sh
#
# Everything below is overridable, e.g.:
#   ITERS=150 FE_LLM_MODEL=kimi-k2.6 ./run_kimi_v3.sh
#
# This runs ONLY the gate + evolution (assumes outputs/<panel>_panel.parquet exists).
# For the full pipeline (data -> panels -> evolve -> eval -> report) use ./run_local.sh.
set -euo pipefail
cd "$(dirname "$0")"

# --- interpreter (mirrors run_local.sh) -----------------------------------------
DEFAULT_PY="/Users/difeisu/miniconda3/bin/python"
if [[ -n "${PY:-}" ]]; then :;
elif [[ -x "$DEFAULT_PY" ]]; then PY="$DEFAULT_PY";
elif command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)";
else PY="$(command -v python)"; fi

# --- LLM env (your verified Kimi/Moonshot .cn settings; all overridable) --------
export FE_LLM_PROVIDER="${FE_LLM_PROVIDER:-kimi-cn}"
export FE_LLM_BASE_URL="${FE_LLM_BASE_URL:-https://api.moonshot.cn/v1}"
export FE_LLM_MODEL="${FE_LLM_MODEL:-kimi-k2.6}"
export FE_LLM_TEMPERATURE="${FE_LLM_TEMPERATURE:-0.6}"
export FE_LLM_TIMEOUT="${FE_LLM_TIMEOUT:-180}"
export FE_LLM_RESOLVE_IP="${FE_LLM_RESOLVE_IP:-8.147.223.37}"   # DNS pin; TLS SNI stays api.moonshot.cn
export FE_REQUIRE_LLM="${FE_REQUIRE_LLM:-1}"

# key from env only — never hardcode credentials in this script
if [[ -z "${FE_LLM_API_KEY:-}" && -n "${MOONSHOT_API_KEY:-}" ]]; then
  export FE_LLM_API_KEY="$MOONSHOT_API_KEY"
fi
: "${FE_LLM_API_KEY:?Set it first:  export FE_LLM_API_KEY='YOUR_MOONSHOT_KEY'}"
export OPENAI_API_KEY="$FE_LLM_API_KEY"
export OPENAI_BASE_URL="$FE_LLM_BASE_URL"

# --- run params -----------------------------------------------------------------
PANEL="${PANEL:-csi300}"
ITERS="${ITERS:-300}"
SEED="${SEED:-1}"
OBJECTIVE="${OBJECTIVE:-portfolio_v3}"
ELITE_RULE="${ELITE_RULE:-robust}"
PATIENCE="${PATIENCE:-50}"
OUT="${OUT:-outputs/evolution.json}"

if [[ ! -f "outputs/${PANEL}_panel.parquet" ]]; then
  echo "[ERROR] outputs/${PANEL}_panel.parquet not found — build panels first via ./run_local.sh"
  exit 1
fi

echo "py            : $PY"
echo "llm           : $FE_LLM_PROVIDER  $FE_LLM_MODEL  @ $FE_LLM_BASE_URL  (resolve $FE_LLM_RESOLVE_IP)"
echo "run           : panel=$PANEL iters=$ITERS objective=$OBJECTIVE elite=$ELITE_RULE patience=$PATIENCE seed=$SEED"

# --- 1. gate (same call path the engine uses) -----------------------------------
echo; echo "== 1. live-LLM gate =="
set +e
"$PY" scripts/check_llm.py
LLM_STATUS=$?
set -e
if [[ "$LLM_STATUS" -eq 0 || "$LLM_STATUS" -eq 2 ]]; then
  export FE_LLM_GATE_PASSED=1   # in-run transient blips fall back per-step, never abort
  echo "   -> Kimi reachable; evolution will use the live agent (gate status $LLM_STATUS)."
else
  echo "[ERROR] gate failed (status $LLM_STATUS). Try FE_LLM_BASE_URL=https://api.moonshot.ai/v1,"
  echo "        a different FE_LLM_MODEL, or FE_LLM_PROXY=socks5h://..."
  exit 1
fi

# --- 2. evolve with the V3 protocol active --------------------------------------
echo; echo "== 2. evolve ($PANEL, $ITERS iters, live Kimi, V3 objective+robust+patience) =="
"$PY" scripts/02_run_evolution.py \
      --panel "$PANEL" \
      --iterations "$ITERS" \
      --islands 2 \
      --trials 12 \
      --use-llm --require-llm \
      --objective "$OBJECTIVE" \
      --elite-rule "$ELITE_RULE" \
      --patience "$PATIENCE" \
      --seed "$SEED" \
      --out "$OUT"

echo; echo "DONE -> $OUT"
echo "Next: $PY scripts/04_evaluate.py --baseline alpha158 --label label --use-benchmark --gplearn"
echo "      $PY scripts/05_build_report.py"
