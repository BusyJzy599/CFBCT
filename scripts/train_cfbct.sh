
BASE_DIR=""  # Where is the storage path for your dataset
DATASET="BRCA"

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir ${BASE_DIR}/${DATASET} \
--split_dir tcga_${DATASET,,} \
--model_type cfbct \
--W_k 0.6 \
--ema 0.9 \
--tmp 0.1 \
--pooling MQP \
--which_splits 5foldcv \
--apply_sig