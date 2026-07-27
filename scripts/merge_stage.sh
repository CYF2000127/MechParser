#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ADAPTER_PATH="${1:?Usage: merge_stage.sh <adapter> <base-model> <output>}"
MODEL_BASE="${2:?Usage: merge_stage.sh <adapter> <base-model> <output>}"
SAVE_PATH="${3:?Usage: merge_stage.sh <adapter> <base-model> <output>}"

require_file "$ADAPTER_PATH/adapter_config.json"
mkdir -p "$(dirname "$SAVE_PATH")"
"$PYTHON_BIN" "$ROOT/src/merge_lora_weights.py" \
    --model-path "$ADAPTER_PATH" \
    --model-base "$MODEL_BASE" \
    --save-model-path "$SAVE_PATH" \
    --safe-serialization