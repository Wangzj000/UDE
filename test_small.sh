#!/bin/bash
# SLURM job configuration (modify according to your cluster)
#SBATCH -o ./test_small.out
#SBATCH --partition=gpu
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2 

# Set your data root path
export DATA_ROOT=${DATA_ROOT:-"./data"}
export CKPT_ROOT=${CKPT_ROOT:-"./12Bcheckpoints/new"}

python -u run.py \
  --task_name pretrain \
  --sampling_rate 0.01 \
  --is_training 0 \
  --use_amp \
  --use_multi_gpu \
  --num_workers 12 \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint.pth \
  --model_id finetune_ETTh2 \
  --model delayformer_fixpooling \
  --data UTSD_Npy \
  --train_dataset_list UTSD_Npy \
  --test_dataset_list "ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic" \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 3 \
  --batch_size 1024 \
  --learning_rate 0 \
  --l2 0.0 \
  --dropout 0.1 \
  --train_epochs 10 \
  --l2 0 \
  --des 'Exp' \
  --itr 1 \
  --pe fix_pe \
  --n_heads 8 \
  --lradj cosine \
  --n_vars 1 \
  --pooling_type avg \
  --pooling_kernel 30 \
  --L 500 \
  --p1 25 \
  --p2 50

python -u run.py \
  --task_name pretrain \
  --sampling_rate 0.01 \
  --is_training 0 \
  --use_amp \
  --use_multi_gpu \
  --num_workers 12 \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint_epoch_1.pth \
  --model_id finetune_ETTh2 \
  --model delayformer_fixpooling \
  --data UTSD_Npy \
  --train_dataset_list UTSD_Npy \
  --test_dataset_list "ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic" \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 3 \
  --batch_size 1024 \
  --learning_rate 0 \
  --l2 0.0 \
  --dropout 0.1 \
  --train_epochs 10 \
  --l2 0 \
  --des 'Exp' \
  --itr 1 \
  --pe fix_pe \
  --n_heads 8 \
  --lradj cosine \
  --n_vars 1 \
  --pooling_type avg \
  --pooling_kernel 30 \
  --L 500 \
  --p1 25 \
  --p2 50

python -u run.py \
  --task_name pretrain \
  --sampling_rate 0.01 \
  --is_training 0 \
  --use_amp \
  --use_multi_gpu \
  --num_workers 12 \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint_epoch_2.pth \
  --model_id finetune_ETTh2 \
  --model delayformer_fixpooling \
  --data UTSD_Npy \
  --train_dataset_list UTSD_Npy \
  --test_dataset_list "ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic" \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 3 \
  --batch_size 1024 \
  --learning_rate 0 \
  --l2 0.0 \
  --dropout 0.1 \
  --train_epochs 10 \
  --l2 0 \
  --des 'Exp' \
  --itr 1 \
  --pe fix_pe \
  --n_heads 8 \
  --lradj cosine \
  --n_vars 1 \
  --pooling_type avg \
  --pooling_kernel 30 \
  --L 500 \
  --p1 25 \
  --p2 50
