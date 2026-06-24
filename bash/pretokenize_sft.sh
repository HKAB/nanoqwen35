#!/usr/bin/env bash
# Pretokenize ShareGPT SFT data into neat-packed parquet shards.
# Each conversation is rendered with the chat template, long conversations are
# split at user-turn boundaries (smart chunking), and items are best-fit packed
# into fixed seq-len blocks. Each block stores input_ids, loss_mask and seq_lens
# (segment lengths for block-diagonal attention). State is tracked in sqlite so
# the run is resumable.
#
# IMPORTANT: train with --max-seq-len equal to (SEQ_LEN - 1) and --pretokenized.

INPUT_DIR=/mnt/data/users/truongnp5/sft_data/**/*.jsonl
OUTPUT_DIR=/mnt/data/users/truongnp5/sft_data_pretokenized/
TOKENIZER=Qwen/Qwen3.5-0.8B-Base
HF_HOME=/mnt/data/users/truongnp5/cache/huggingface
SEQ_LEN=2049   # train --max-seq-len must be 2048

python -m scripts.pretokenize \
    --mode            sft \
    --input-dir       "$INPUT_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --tokenizer       "$TOKENIZER" \
    --hf-home         "$HF_HOME" \
    --seq-len         "$SEQ_LEN" \
    --pack-chunk-size 50000 \
    --long-doc        smart_chunk \
    --cpu-count       16 \
    --write-threshold 20000 \
    --ignore
