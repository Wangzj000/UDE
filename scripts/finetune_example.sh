#!/bin/bash
# Fine-tuning script for UDE models
# This script fine-tunes a pretrained model on a specific downstream dataset

# ============================================================
# Configuration - Modify these paths according to your setup
# ============================================================
export DATA_ROOT=${DATA_ROOT:-"./data"}           # Root directory of datasets
export CKPT_ROOT=${CKPT_ROOT:-"./12Bcheckpoints/new"}  # Root directory of pretrained checkpoints

# ============================================================
# Configurable parameters
# ============================================================
MODEL_SIZE=${1:-"small"}         # Model size: small, medium, large
DATASET=${2:-"ETT-small/ETTm2.csv"}  # Dataset path relative to DATA_ROOT
SAMPLING_RATE=${3:-"0.1"}        # Data sampling rate for few-shot learning

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
        echo "Usage: $0 [small|medium|large] [dataset_path] [sampling_rate]"
        exit 1
        ;;
esac

echo "Fine-tuning ${MODEL_SIZE} model on ${DATASET} with sampling_rate=${SAMPLING_RATE}..."

python -u run.py \
  --task_name finetune \
  --sampling_rate ${SAMPLING_RATE} \
  --fc_only \
  --is_training 1 \
  --num_workers 8 \
  --devices 0 \
  --root_path ${DATA_ROOT} \
  --data_path ${DATASET} \
  --model_location ${CKPT_ROOT}/${MODEL_SIZE}/checkpoint.pth \
  --model_id finetune_${MODEL_SIZE} \
  --model delayformer_fixpooling \
  --data UnivariateDatasetBenchmark_finetune \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers ${E_LAYERS} \
  --d_layers 1 \
  --patience 2 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --l2 0. \
  --dropout 0.3 \
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
