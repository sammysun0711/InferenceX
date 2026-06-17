#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_VLM_CACHE_SIZE_MB=8192 #ali use0
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_AITER=1
export USE_AITER_COMM=1
python3 -m sglang.launch_server \
    --host=0.0.0.0 \
    --port $PORT \
    --model-path $MODEL \
    --tensor-parallel-size $TP \
    --enable-multimodal \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.8 \
    --max-prefill-tokens 32768 \
    --reasoning-parser qwen3 \
    --attention-backend aiter \
    --mm-attention-backend aiter_attn \
    --max-running-requests 128 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --disable-custom-all-reduce > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
