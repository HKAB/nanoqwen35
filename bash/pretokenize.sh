#!/usr/bin/env bash
# Pretokenize jsonl data into packed parquet shards.
# Each file is split into CPU_COUNT byte-ranges processed in parallel; documents
# are tokenized, concatenated with EOS, and packed into fixed-size blocks.
# State is tracked in a sqlite DB so the run is resumable.

INPUT_DIR=/mnt/data/users/truongnp5/final_clean_data/final_merge/**/*.jsonl
OUTPUT_DIR=/mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_v1_pretokenized/
TOKENIZER=Qwen/Qwen3.5-0.8B-Base
HF_HOME=/mnt/data/users/truongnp5/cache/huggingface

python -m scripts.pretokenize \
    --input-dir       "$INPUT_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --tokenizer       "$TOKENIZER" \
    --hf-home         "$HF_HOME" \
    --chunk-size      8192 \
    --cpu-count       16 \
    --write-threshold 20000
