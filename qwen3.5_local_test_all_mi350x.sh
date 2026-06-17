#!/usr/bin/env bash
#
# Run Qwen3.5 BF16 and FP8 local benchmarks on MI355X.
# From InferenceX repo root.
#
# Matrix:
#   BF16:  BF16_CONC_LIST -> 4, 8, 16, 32, 64; TP=8 (runner default)
#   FP8:   FP8_TP8_CONC_LIST -> 4, 8, 16, 32, 64; TP=8
#          FP8_TP4_CONC_LIST -> 16, 32, 64, 128; TP=4
#   Seq:   (ISL 1024, OSL 1024), (ISL 8192, OSL 1024) for all sweeps
#

set -e

export MODEL="${MODEL:-/models/Qwen3.5-397B-A17B-FP8}"

# (ISL, OSL) pairs: 1k1k, 8k1k
# SEQ_CONFIGS=(1024:1024 8192:1024)
SEQ_CONFIGS=(8192:1024)
# export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# export OPTFLAG="w8a8_gemm,moe"

# BF16_CONC_LIST=(4 8 16 32 64)
# for isl_osl in "${SEQ_CONFIGS[@]}"; do
#   IFS=: read -r isl osl <<< "$isl_osl"
#   for conc in "${BF16_CONC_LIST[@]}"; do
#     echo "=== BF16 CONC=$conc ISL=$isl OSL=$osl ==="
#     ./scripts/run_qwen3.5_bf16_mi305x_local.sh "$conc" "$isl" "$osl"
#   done
# done

FP8_TP8_CONC_LIST=(2 4 8 16 32 64 128)
for isl_osl in "${SEQ_CONFIGS[@]}"; do
  IFS=: read -r isl osl <<< "$isl_osl"
  for conc in "${FP8_TP8_CONC_LIST[@]}"; do
    echo "=== FP8 TP=8 CONC=$conc ISL=$isl OSL=$osl ==="
    TP=8 ./scripts/run_qwen3.5_fp8_mi350x_local.sh "$conc" "$isl" "$osl"
  done
done


FP8_TP4_CONC_LIST=(2 4 8 16 32 64 128)
for isl_osl in "${SEQ_CONFIGS[@]}"; do
  IFS=: read -r isl osl <<< "$isl_osl"
  for conc in "${FP8_TP4_CONC_LIST[@]}"; do
    echo "=== FP8 TP=4 CONC=$conc ISL=$isl OSL=$osl ==="
    TP=4 ./scripts/run_qwen3.5_fp8_mi350x_local.sh "$conc" "$isl" "$osl"
  done
done

echo "=== All runs finished ==="
