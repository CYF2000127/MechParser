#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DEVICES="$(num_gpus)"
MODEL_ID="${MECHPARSER_STAGE3B_MODEL:-$OUTPUT_ROOT/merged/stage3a}"
DATA_PATH="${MECHPARSER_STAGE3B_TRAIN:-$DATA_ROOT/real/grpo_train.json}"
IMAGE_FOLDER="${MECHPARSER_REAL_IMAGES:-$DATA_ROOT/real/images}"
OUTPUT_DIR="${MECHPARSER_STAGE3B_ADAPTER:-$OUTPUT_ROOT/adapters/stage3b}"
DISABLE_FLASH_ATTN="$(flash_attention_disabled)"

require_file "$DATA_PATH"
echo "Stage=3b model=$MODEL_ID data=$DATA_PATH output=$OUTPUT_DIR GPUs=$DEVICES"

"$DEEPSPEED_BIN" --num_gpus="$DEVICES" "$ROOT/src/train/train_grpo.py" \
    --use_liger_loss False \
    --lora_enable True --vision_lora True --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
    --lora_bias none --num_lora_modules -1 \
    --deepspeed "$ROOT/configs/zero2.json" \
    --model_id "$MODEL_ID" \
    --reward_funcs_module src.train.arrow_reward_funcs --reward_weights 3.0 1.0 \
    --data_path "$DATA_PATH" --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower True --freeze_llm True --freeze_merger True \
    --bf16 True --fp16 False --disable_flash_attn2 "$DISABLE_FLASH_ATTN" \
    --output_dir "$OUTPUT_DIR" --num_train_epochs 8 \
    --num_generations 6 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_completion_length 512 --max_prompt_length 4096 \
    --image_min_pixels $((256 * 32 * 32)) --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate 1e-6 --beta 0.04 --temperature 1.1 \
    --weight_decay 0.1 --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --logging_steps 1 --tf32 True --gradient_checkpointing False \
    --report_to tensorboard --lazy_preprocess True \
    --save_strategy steps --save_steps 50 --save_total_limit 5 \
    --eval_strategy no --dataloader_num_workers 4