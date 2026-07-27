#!/usr/bin/env bash

set -euo pipefail

ROOT="${MECHPARSER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${MECHPARSER_DATA_ROOT:-$ROOT/data}"
OUTPUT_ROOT="${MECHPARSER_OUTPUT_ROOT:-$ROOT/outputs}"
BASE_MODEL="${MECHPARSER_BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${MECHPARSER_PYTHON:-python}"
DEEPSPEED_BIN="${MECHPARSER_DEEPSPEED:-deepspeed}"

mkdir -p "$OUTPUT_ROOT" "$ROOT/slurm_log"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/mechparser-triton-${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TRITON_CACHE_DIR"

num_gpus() {
    local value="${MECHPARSER_NUM_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$value"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l
        return
    fi
    printf '1\n'
}

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Required file is missing or empty: $1" >&2
        exit 2
    fi
}

gradient_accumulation() {
    local global_batch="$1" per_device="$2" devices="$3"
    local denominator=$((per_device * devices))
    if (( global_batch % denominator != 0 )); then
        echo "Global batch $global_batch is not divisible by per-device batch $per_device x $devices GPUs" >&2
        exit 2
    fi
    printf '%s\n' $((global_batch / denominator))
}

flash_attention_disabled() {
    if "$PYTHON_BIN" -c "import flash_attn" >/dev/null 2>&1; then
        printf 'False\n'
    else
        printf 'True\n'
    fi
}