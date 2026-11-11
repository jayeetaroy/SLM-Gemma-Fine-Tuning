#!/usr/bin/env bash
set -euo pipefail
: "${ADAPTER_PATH:?Set ADAPTER_PATH to the saved LoRA adapter directory}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-1b-it}"
ARGS=(--model "$BASE_MODEL" --served-model-name base --enable-lora --lora-modules "tool-specialist=$ADAPTER_PATH" --max-lora-rank 16 --max-model-len 2048 --gpu-memory-utilization 0.85 --port 8000)
if [[ -n "${BASE_REVISION:-}" ]]; then ARGS+=(--revision "$BASE_REVISION"); fi
exec vllm serve "${ARGS[@]}"
