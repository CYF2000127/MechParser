#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ACCOUNT="${MECHPARSER_ACCOUNT:?Set MECHPARSER_ACCOUNT}"
S1_PART="${MECHPARSER_STAGE1_PARTITION:?Set MECHPARSER_STAGE1_PARTITION}"
S1_GRES="${MECHPARSER_STAGE1_GRES:?Set MECHPARSER_STAGE1_GRES}"
S2_PART="${MECHPARSER_STAGE2_PARTITION:?Set MECHPARSER_STAGE2_PARTITION}"
S2_GRES="${MECHPARSER_STAGE2_GRES:?Set MECHPARSER_STAGE2_GRES}"
S3A_PART="${MECHPARSER_STAGE3A_PARTITION:?Set MECHPARSER_STAGE3A_PARTITION}"
S3A_GRES="${MECHPARSER_STAGE3A_GRES:?Set MECHPARSER_STAGE3A_GRES}"
S3B_PART="${MECHPARSER_STAGE3B_PARTITION:?Set MECHPARSER_STAGE3B_PARTITION}"
S3B_GRES="${MECHPARSER_STAGE3B_GRES:?Set MECHPARSER_STAGE3B_GRES}"
MERGE_PART="${MECHPARSER_MERGE_PARTITION:?Set MECHPARSER_MERGE_PARTITION}"
MERGE_GRES="${MECHPARSER_MERGE_GRES:?Set MECHPARSER_MERGE_GRES}"

mkdir -p "$ROOT/slurm_log"

submit_train() {
    local name="$1" partition="$2" gres="$3" dependency="$4" script="$5"
    local dep_args=()
    [[ -n "$dependency" ]] && dep_args=(--dependency="afterok:$dependency")
    sbatch --parsable --account="$ACCOUNT" --partition="$partition" --gres="$gres" \
        --job-name="$name" --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=0 \
        --time=3-00:00:00 --output="$ROOT/slurm_log/${name}_%j.out" \
        --error="$ROOT/slurm_log/${name}_%j.err" --export=ALL \
        "${dep_args[@]}" "$script"
}

submit_merge() {
    local name="$1" dependency="$2" adapter="$3" base="$4" output="$5"
    sbatch --parsable --account="$ACCOUNT" --partition="$MERGE_PART" --gres="$MERGE_GRES" \
        --job-name="$name" --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --time=04:00:00 --output="$ROOT/slurm_log/${name}_%j.out" \
        --error="$ROOT/slurm_log/${name}_%j.err" --export=ALL \
        --dependency="afterok:$dependency" "$ROOT/scripts/merge_stage.sh" \
        "$adapter" "$base" "$output"
}

s1="$(submit_train mechparser-s1 "$S1_PART" "$S1_GRES" "" "$ROOT/scripts/train_stage1.sh")"
m1="$(submit_merge mechparser-merge-s1 "$s1" "$OUTPUT_ROOT/adapters/stage1" "$BASE_MODEL" "$OUTPUT_ROOT/merged/stage1")"
s2="$(submit_train mechparser-s2 "$S2_PART" "$S2_GRES" "$m1" "$ROOT/scripts/train_stage2.sh")"
m2="$(submit_merge mechparser-merge-s2 "$s2" "$OUTPUT_ROOT/adapters/stage2" "$OUTPUT_ROOT/merged/stage1" "$OUTPUT_ROOT/merged/stage2")"
s3a="$(submit_train mechparser-s3a "$S3A_PART" "$S3A_GRES" "$m2" "$ROOT/scripts/train_stage3a.sh")"
m3a="$(submit_merge mechparser-merge-s3a "$s3a" "$OUTPUT_ROOT/adapters/stage3a" "$OUTPUT_ROOT/merged/stage2" "$OUTPUT_ROOT/merged/stage3a")"
s3b="$(submit_train mechparser-s3b "$S3B_PART" "$S3B_GRES" "$m3a" "$ROOT/scripts/train_stage3b.sh")"
final="$(submit_merge mechparser-merge-final "$s3b" "$OUTPUT_ROOT/adapters/stage3b" "$OUTPUT_ROOT/merged/stage3a" "$OUTPUT_ROOT/merged/final")"

printf 'stage1=%s\nmerge1=%s\nstage2=%s\nmerge2=%s\nstage3a=%s\nmerge3a=%s\nstage3b=%s\nfinal_merge=%s\n' \
    "$s1" "$m1" "$s2" "$m2" "$s3a" "$m3a" "$s3b" "$final"