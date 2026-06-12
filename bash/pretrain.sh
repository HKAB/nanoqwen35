# Training with weighting
# {
#     "math_coding": 0.15, # 	942,341,016 + 391,859,673 + 420,501,058
#     "fineweb2-vi-edu": 0.15, # 7,147,560,008
#     "fineweb-edu": 0.1, # 15,143,783,278
#     "vi_wiki": 0.15, # 499,762,832
#     "vigpt": 0.07, # 16,566,012,949
#     "vietnamese_curated_dataset": 0.05, # 10,324,888,638
#     "ocr": 0.07, "law": 0.05, "book": 0.09, # 1,271,336,637
#     "vietjack": 0.03, # ~100,000,000
#     "finepdfs-edu-vie": 0.055, # ~100,000,000
#     "politic_sensitive": 0.03, # 1,249,071
#     "driver_license_cert": 0.005 # 123,007
# }
python -m scripts.base_train \
    --pretrained-model-path /mnt/data/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --dataset-root /mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_v1 \
    --domain-weights '{"math_coding": 0.15, "fineweb2-vi-edu": 0.15, "fineweb-edu": 0.1, "vi_wiki": 0.15, "vigpt": 0.07, "vietnamese_curated_dataset": 0.05, "ocr": 0.07, "law": 0.05, "book": 0.09, "vietjack": 0.03, "finepdfs-edu-vie": 0.05, "politic_sensitive": 0.03, "driver_license_cert": 0.01}' \
    --max-seq-len 4096 \
    --num-iterations 14200 \
    --device-batch-size 8 \
    --gradient-checkpointing \
    --total-batch-size 131072 \
    --embedding-lr 5e-5 \
    --unembedding-lr 5e-5 \
    --matrix-lr 5e-5 \
    --scalar-lr 5e-5 \
    --warmdown-ratio 0.1 \
    --optimizer muon \
    --weight-decay 0.1 \
    --final-lr-frac 0.1 \
    --eval-every 1000 \
    --eval-tokens 131072 \
    --core-metric-every 1000 \
    --core-metric-max-per-task 500 \
    --sample-every 1000 \
    --save-every 1000