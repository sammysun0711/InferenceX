#!/usr/bin/env bash
#
# Local runner for InferenceX benchmark: qwen3.5_fp8_mi350x.sh
# Runs the single-node Qwen3.5-397B-A17B-FP8 SGLang benchmark without GitHub Actions.
#
# Usage:
#   ./scripts/run_qwen3.5_fp8_mi350x_local.sh [CONC] [ISL] [OSL]
#
# Examples:
#   ./scripts/run_qwen3.5_fp8_mi350x_local.sh              # defaults: CONC=4, 1k1k
#   ./scripts/run_qwen3.5_fp8_mi350x_local.sh 8            # CONC=8, 1k1k
#   ./scripts/run_qwen3.5_fp8_mi350x_local.sh 16 1024 8192 # CONC=16, 1k8k
#
# Override via environment (optional):
#   INFERENCEX_WORKSPACE  - directory for server log, results, GPU metrics (default: $(pwd)/bench_workspace)
#   MODEL                 - HuggingFace model (default: Qwen/Qwen3.5-397B-A17B-FP8)
#   RUN_EVAL              - set to "true" to run lm-eval after throughput (default: false)
#

set -e

# --- Repo root (script lives in InferenceX/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Workspace for logs and results (avoids hardcoded /workspace)
export WORKSPACE="${INFERENCEX_WORKSPACE:-$(pwd)/bench_workspace}"
mkdir -p "$WORKSPACE"
echo "[run_local] WORKSPACE=$WORKSPACE"

# --- Parameters
export MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
export TP="${TP:-4}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
export RUN_EVAL="${RUN_EVAL:-false}"

# Optional: sequence length and concurrency from args or env
CONC="${1:-${CONC:-4}}"
ISL="${2:-${ISL:-1024}}"
OSL="${3:-${OSL:-1024}}"
export CONC ISL OSL

# Result filename (match CI pattern for compatibility with process_result.py)
# Format: exp-name_precision_framework_tpX-epY-dpaFalse_disagg-false_spec-none_concZ_local
EXP_NAME="qwen3.5_${ISL}_${OSL}"
export RESULT_FILENAME="${EXP_NAME}_fp8_sglang_tp${TP}-ep1-dpaFalse_disagg-false_spec-none_conc${CONC}_local"

echo "[run_local] MODEL=$MODEL TP=$TP CONC=$CONC ISL=$ISL OSL=$OSL"
echo "[run_local] RESULT_FILENAME=$RESULT_FILENAME"
echo "[run_local] RUN_EVAL=$RUN_EVAL"
echo ""

# --- Run benchmark: replace /workspace with $WORKSPACE so it works outside CI
# The benchmark script and benchmark_lib.sh use /workspace for SERVER_LOG, result-dir, GPU csv.
# We build temp copies with /workspace replaced and point the benchmark at the patched lib.
BENCH_SCRIPT="$REPO_ROOT/benchmarks/single_node/qwen3.5_fp8_mi350x.sh"
BENCH_LIB="$REPO_ROOT/benchmarks/benchmark_lib.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Error: Benchmark script not found: $BENCH_SCRIPT"
  exit 1
fi
if [[ ! -f "$BENCH_LIB" ]]; then
  echo "Error: Benchmark lib not found: $BENCH_LIB"
  exit 1
fi

LIB_TMP=$(mktemp)
BENCH_TMP=$(mktemp)
trap "rm -f '$LIB_TMP' '$BENCH_TMP'" EXIT

sed "s|/workspace|$WORKSPACE|g" "$BENCH_LIB" > "$LIB_TMP"
sed "s|/workspace|$WORKSPACE|g" "$BENCH_SCRIPT" | \
  sed "s|source \"\$(dirname \"\$0\")/../benchmark_lib.sh\"|source \"$LIB_TMP\"|" > "$BENCH_TMP"

bash "$BENCH_TMP"

echo "[run_local] Done. Results under $WORKSPACE (e.g. ${RESULT_FILENAME}.json)"
