#!/usr/bin/env bash
# Pretokenize and merge all domain data into flat shards.
# No domain weights — every document is used exactly once.
# Files are shuffled before tokenization; shards are written continuously.
# Output is read directly by pretrain.sh (base_train.py).
#
# Memory tuning:
#   peak_per_worker ≈ largest_file_rows × (T+1) × 4 bytes  (dominant)
#                   + rows_per_shard    × (T+1) × 4 bytes  (shard buffer)
#   total_peak = workers × peak_per_worker
#
#   With workers=8 and ~24k rows/file: ~8 × (393 MB + 67 MB) ≈ 3.7 GB
#   With workers=4:                     ~4 × (393 MB + 67 MB) ≈ 1.8 GB

SOURCE_ROOT=/mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_v1
OUTPUT_ROOT=/mnt/data/users/truongnp5/final_clean_data/vi_en_merged_v1
TOKENIZER=/mnt/data/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68

python -m scripts.pretokenize_and_merge \
    --source-root     "$SOURCE_ROOT" \
    --output-root     "$OUTPUT_ROOT" \
    --tokenizer       "$TOKENIZER" \
    --T               4096 \
    --rows-per-shard  4096 \
    --workers         8 \
    --seed            42
