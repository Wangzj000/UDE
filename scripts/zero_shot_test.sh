#!/bin/bash
# Zero-shot evaluation script for UDE models
# This script evaluates pretrained models without any fine-tuning

# ============================================================
# Configuration - Modify these paths according to your setup
# ============================================================
export DATA_ROOT=${DATA_ROOT:-"./data"}           # Root directory of datasets
export CKPT_ROOT=${CKPT_ROOT:-"./12Bcheckpoints/new"}  # Root directory of pretrained checkpoints

# ============================================================
# Model configurations:
#   Small:  e_layers=6,  n_heads=8,  ~30M params
#   Medium: e_layers=10, n_heads=12, ~60M params
#   Large:  e_layers=12, n_heads=16, ~90M params
# ============================================================

MODEL_SIZE=${1:-"small"}  # Accept model size as argument: small, medium, large

case $MODEL_SIZE in
    small)
        E_LAYERS=6
        N_HEADS=8
        ;;
    medium)
        E_LAYERS=10
        N_HEADS=12
        ;;
    large)
        E_LAYERS=12
        N_HEADS=16
        ;;
    *)
        echo "Usage: $0 [small|medium|large]"
        exit 1
        ;;
esac

echo "Running zero-shot evaluation with ${MODEL_SIZE} model..."
echo "Data root: ${DATA_ROOT}"
echo "Checkpoint: ${CKPT_ROOT}/${MODEL_SIZE}/checkpoint.pth"

python -u run.py \
  --task_name pretrain \
  --sampling_rate 0.01 \
  --is_training 0 \
  --use_amp \
  --num_workers 8 \
  --devices 0 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/${MODEL_SIZE}/checkpoint.pth \
  --model_id zero_shot_${MODEL_SIZE} \
  --model delayformer_fixpooling \
  --data UTSD_Npy \
  --train_dataset_list UTSD_Npy \
  --test_dataset_list "ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic" \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers ${E_LAYERS} \
  --d_layers 1 \
  --patience 3 \
  --batch_size 512 \
  --learning_rate 0 \
  --l2 0.0 \
  --dropout 0.1 \
  --train_epochs 10 \
  --des 'Exp' \
  --itr 1 \
  --pe fix_pe \
  --n_heads ${N_HEADS} \
  --lradj cosine \
  --n_vars 1 \
  --pooling_type avg \
  --pooling_kernel 30 \
  --L 500 \
  --p1 25 \
  --p2 50
