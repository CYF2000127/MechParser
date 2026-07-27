#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

STAGE="${1:?Usage: train_sft_stage.sh <1|2|3a>}"
DEVICES="$(num_gpus)"
DISABLE_FLASH_ATTN="$(flash_attention_disabled)"

case "$STAGE" in
    1)
        MODEL_ID="${MECHPARSER_STAGE1_MODEL:-$BASE_MODEL}"
        DATA_PATH="${MECHPARSER_STAGE1_TRAIN:-$DATA_ROOT/stage1/arrow_train.json}"
        EVAL_PATH="${MECHPARSER_STAGE1_EVAL:-$DATA_ROOT/stage1/arrow_eval.json}"
        IMAGE_FOLDER="${MECHPARSER_STAGE1_IMAGES:-$DATA_ROOT/stage1/images}"
        OUTPUT_DIR="${MECHPARSER_STAGE1_ADAPTER:-$OUTPUT_ROOT/adapters/stage1}"
        # Appendix D, Stage 1: vision tower frozen, peak LR 1e-4, 3% warmup, 2 epochs
        GLOBAL_BATCH=48; PER_DEVICE=8; EPOCHS=2
        LR=1e-4; VISION_LR=5e-5; MERGER_LR=1e-5
        WARMUP=0.03; LORA_DROPOUT=0.05
        VISION_LORA=False; FREEZE_MERGER=False; GRAD_CHECKPOINT=True
        DEEPSPEED_CONFIG="$ROOT/configs/zero2.json"
        ;;
    2)
        MODEL_ID="${MECHPARSER_STAGE2_MODEL:-$OUTPUT_ROOT/merged/stage1}"
        DATA_PATH="${MECHPARSER_STAGE2_TRAIN:-$DATA_ROOT/stage2/arrow_train.json}"
        EVAL_PATH="${MECHPARSER_STAGE2_EVAL:-$DATA_ROOT/stage2/arrow_eval.json}"
        IMAGE_FOLDER="${MECHPARSER_STAGE2_IMAGES:-$DATA_ROOT/stage2/images}"
        OUTPUT_DIR="${MECHPARSER_STAGE2_ADAPTER:-$OUTPUT_ROOT/adapters/stage2}"
        # Appendix D, Stage 2: vision tower unfrozen, peak LR 5e-5 with a
        # 2x-smaller LR on the vision-side LoRA, 3% warmup, 2 epochs
        GLOBAL_BATCH=48; PER_DEVICE=8; EPOCHS=2
        LR=5e-5; VISION_LR=2.5e-5; MERGER_LR=5e-6
        WARMUP=0.03; LORA_DROPOUT=0.05
        VISION_LORA=True; FREEZE_MERGER=True; GRAD_CHECKPOINT=False
        DEEPSPEED_CONFIG="$ROOT/configs/zero2.json"
        ;;
    3a)
        MODEL_ID="${MECHPARSER_STAGE3A_MODEL:-$OUTPUT_ROOT/merged/stage2}"
        DATA_PATH="${MECHPARSER_STAGE3A_TRAIN:-$DATA_ROOT/real/mixed_train.json}"
        EVAL_PATH="${MECHPARSER_REAL_EVAL:-$DATA_ROOT/real/real_eval.json}"
        IMAGE_FOLDER="${MECHPARSER_REAL_IMAGES:-$DATA_ROOT/real/images}"
        OUTPUT_DIR="${MECHPARSER_STAGE3A_ADAPTER:-$OUTPUT_ROOT/adapters/stage3a}"
        # Appendix D, Stage 3a: mixed 3,000 synthetic + 620 real,
        # reduced peak LR 2e-5, 5% warmup, 3 epochs
        GLOBAL_BATCH=48; PER_DEVICE=8; EPOCHS=3
        LR=2e-5; VISION_LR=1e-5; MERGER_LR=1e-6
        WARMUP=0.05; LORA_DROPOUT=0.05
        VISION_LORA=True; FREEZE_MERGER=True; GRAD_CHECKPOINT=False
        DEEPSPEED_CONFIG="$ROOT/configs/zero2.json"
        ;;
    *)
        echo "Unknown SFT stage: $STAGE" >&2
        exit 2
        ;;
esac

require_file "$DATA_PATH"
GRAD_ACCUM="$(gradient_accumulation "$GLOBAL_BATCH" "$PER_DEVICE" "$DEVICES")"

EVAL_ARGS=(--eval_strategy no --prediction_loss_only True)
if [[ "${MECHPARSER_ENABLE_TRAIN_EVAL:-0}" == 1 ]]; then
    require_file "$EVAL_PATH"
    EVAL_ARGS=(
        --eval_path "$EVAL_PATH"
        --eval_strategy epoch
        --per_device_eval_batch_size 1
        --generation_max_new_tokens 512
        --prediction_loss_only False
    )
fi

echo "Stage=$STAGE model=$MODEL_ID data=$DATA_PATH output=$OUTPUT_DIR"
echo "GPUs=$DEVICES global_batch=$GLOBAL_BATCH per_device=$PER_DEVICE grad_accum=$GRAD_ACCUM"

"$DEEPSPEED_BIN" --num_gpus="$DEVICES" "$ROOT/src/train/train_sft.py" \
    --use_liger_kernel False \
    --lora_enable True \
    --vision_lora "$VISION_LORA" \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 --lora_alpha 64 --lora_dropout "$LORA_DROPOUT" \
    --lora_bias none --num_lora_modules -1 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower True --freeze_llm True --freeze_merger "$FREEZE_MERGER" \
    --bf16 True --fp16 False --disable_flash_attn2 "$DISABLE_FLASH_ATTN" \
    --output_dir "$OUTPUT_DIR" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --image_min_pixels $((256 * 32 * 32)) --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate "$LR" --vision_lr "$VISION_LR" --merger_lr "$MERGER_LR" \
    --weight_decay 0.1 --warmup_ratio "$WARMUP" --lr_scheduler_type cosine \
    --logging_steps 1 --tf32 True --gradient_checkpointing "$GRAD_CHECKPOINT" \
    --report_to tensorboard --lazy_preprocess True \
    --save_strategy steps --save_steps 500 --save_total_limit 3 \
    --dataloader_num_workers 4 \
    "${EVAL_ARGS[@]}"