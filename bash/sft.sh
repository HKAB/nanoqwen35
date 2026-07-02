#!/usr/bin/env bash
# SFT training from a previous pretrain checkpoint.
#
# Expected pipeline:
#   raw ShareGPT jsonl -> bash/pretokenize_sft.sh -> this script
#
# Notes:
# - This script loads weights (and optimizer state) from base pretrain phase.
# - If your SFT data is raw messages parquet (not pretokenized), remove --pretokenized.

export WANDB_MODE=offline
export NANOQWEN_BASE_DIR=/mnt/data/users/truongnp5/uv_env/nanoqwen35/.cache/nanoqwen35

torchrun --nproc_per_node=8 --rdzv-conf "timeout=7200" -m scripts.chat_sft -- \
    --run qwen_0.8B_sft \
    --model-tag pretrained_0.8B \
    --model-step 57220 \
    --load-optimizer 1 \
    --dataset-root /mnt/data/users/truongnp5/sft_data_pretokenized \
    --pretokenized \
    --max-seq-len 2048 \
    --num-iterations 3000 \
    --device-batch-size 2 \
    --total-batch-size 262144 \
    --embedding-lr 1e-5 \
    --unembedding-lr 1e-5 \
    --matrix-lr 1e-5 \
    --init-lr-frac 1.0 \
    --warmup-ratio 0.03 \
    --warmdown-ratio 0.6 \
    --final-lr-frac 0.05 \
    --eval-every 200 \
    --eval-tokens 1048576 \
    --chatcore-every -1
