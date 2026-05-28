# ~ 50B tokens
python -m scripts.base_train \
                            --pretrained-model-path /mnt/data/users/truongnp5/cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
                            --dataset-path /mnt/data/users/truongnp5/final_clean_data/vi_en_clean_dataset_v1 \
                            --max-seq-len 4096 \
                            --num-iterations 1000 \
                            --device-batch-size 4 \
                            --total-batch-size 131072 \
                            --embedding-lr 1e-5 \
                            --unembedding-lr 1e-5 \
                            --matrix-lr 1e-5 \
                            --scalar-lr 1e-5 \
                            --warmdown-ratio 0.1 \
                            --optimizer muon \
                            --weight-decay 0.1 \
                            --final-lr-frac 0.1 \
                            --eval-every 1000 \
                            --eval-tokens 131072 \
                            --core-metric-every 50 \
                            --core-metric-max-per-task 500 \
                            --sample-every 20 \
                            --save-every 1000

# Gemini recommend
# python -m scripts.base_train \
#     --pretrained-model-path /mnt/data/users/truongnp5/cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
#     --dataset-path /mnt/data/users/truongnp5/final_clean_data/vietnamese_clean_dataset_v1 \
#     --max-seq-len 4096 \
#     --num-iterations 23841 \
#     --device-batch-size 16 \
#     --total-batch-size 2097152 \
#     --embedding-lr 5e-5 \
#     --unembedding-lr 5e-5 \
#     --matrix-lr 5e-5 \
#     --scalar-lr 5e-5 \
#     --warmdown-ratio 0.1 \
#     --optimizer muon \
#     --weight-decay 0.1 \
#     --final-lr-frac 0.1 \
#     --eval-every 1000 \
#     --eval-tokens 524288 \
#     --core-metric-every 1000 \
#     --core-metric-max-per-task 500 \
#     --sample-every 1000 \
#     --save-every 2000