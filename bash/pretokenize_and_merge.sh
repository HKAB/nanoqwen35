#!/usr/bin/env bash
# Pretokenize and merge all domain data into flat shards.
# No domain weights — every document is used exactly once.
# Output is read directly by pretrain.sh (base_train.py).

SOURCE_ROOT=/mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_v1
OUTPUT_ROOT=/mnt/data/users/truongnp5/final_clean_data/vi_en_merged_v1
TOKENIZER=/mnt/data/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68

python -m scripts.pretokenize_and_merge \
    --source-root    "$SOURCE_ROOT" \
    --output-root    "$OUTPUT_ROOT" \
    --tokenizer      "$TOKENIZER" \
    --T              4096 \
    --num-shards     256 \
    --num-val-shards 32 \
    --workers        13 \
    --seed           42
