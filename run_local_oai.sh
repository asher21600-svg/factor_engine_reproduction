#!/usr/bin/env bash
# Compatibility wrapper. The Moonshot/Kimi OpenAI-compatible path now lives in
# run_local.sh so there is one source of truth for the real-data LLM run.
set -euo pipefail
cd "$(dirname "$0")"
echo "run_local_oai.sh is deprecated; delegating to run_local.sh"
exec ./run_local.sh "$@"
