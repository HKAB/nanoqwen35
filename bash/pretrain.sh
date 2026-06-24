# Pre-training on pre-tokenized parquet files.
# Pipeline: raw jsonl → pretokenize.sh → this script.

export WANDB_MODE=offline

torchrun --nproc_per_node=8 --rdzv-conf "timeout=7200" -m scripts.base_train -- \
    --run qwen_0.8B \
    --wandb-project nanoqwen35 \
    --wandb-entity hkab \
    --wandb-tags "0.8B,pretrain" \
    --pretrained-model-path /mnt/data/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --dataset-root /mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_v1_pretokenized \
    --max-seq-len 8192 \
    --num-iterations 57220 \
    --device-batch-size 2 \
    --total-batch-size 1048576 \
    --embedding-lr 5e-5 \
    --unembedding-lr 5e-5 \
    --matrix-lr 5e-5 \
    --scalar-lr 5e-5 \
    --warmdown-ratio 0.1 \
    --warmup-steps 2000 \
    --optimizer muon \
    --weight-decay 0.1 \
    --final-lr-frac 0.1 \
    --eval-every 1000 \
    --eval-tokens 1048576 \
    --core-metric-every 500 \
    --core-metric-max-per-task 500 \
    --sample-every 500 \
    --save-every 2000
