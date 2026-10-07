#!/bin/bash
# SLURM job configuration (modify according to your cluster)
#SBATCH -o ./ETTm2_small.out
#SBATCH --partition=gpu
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2 

# Set your data root path and checkpoint root
export DATA_ROOT=${DATA_ROOT:-"../data"}
export CKPT_ROOT=${CKPT_ROOT:-"../12Bcheckpoints/new"}

rate=0
python -u ../run.py \
  --task_name finetune \
  --sampling_rate $rate \
  --fc_only \
  --is_training 0 \
  --num_workers 8 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint.pth \
  --model_id finetune_ETTm2_small \
  --model delayformer_fixpooling \
  --data UnivariateDatasetBenchmark_finetune \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 2 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --l2 0. \
  --dropout 0.3 \
  --train_epochs 10 \
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

for rate in 0.01 0.02 0.05 0.1
do
echo "====================================================rate=$rate===================================================="
python -u ../run.py \
  --task_name finetune \
  --sampling_rate $rate \
  --fc_only \
  --is_training 1 \
  --num_workers 8 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint.pth \
  --model_id finetune_ETTm2_small \
  --model delayformer_fixpooling \
  --data UnivariateDatasetBenchmark_finetune \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 2 \
  --batch_size 512 \
  --learning_rate 1e-5 \
  --l2 0. \
  --dropout 0.2 \
  --train_epochs 10 \
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
done

for rate in  0.2 0.3 0.5
do
echo "====================================================rate=$rate===================================================="
python -u ../run.py \
  --task_name finetune \
  --sampling_rate $rate \
  --fc_only \
  --is_training 1 \
  --num_workers 8 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint.pth \
  --model_id finetune_ETTm2_small \
  --model delayformer_fixpooling \
  --data UnivariateDatasetBenchmark_finetune \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 2 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --l2 0. \
  --dropout 0.4 \
  --train_epochs 10 \
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
done


for rate in  0.8 1
do
echo "====================================================rate=$rate===================================================="
python -u ../run.py \
  --task_name finetune \
  --sampling_rate $rate \
  --fc_only \
  --is_training 1 \
  --num_workers 8 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ${DATA_ROOT} \
  --data_path ETT-small/ETTm2.csv \
  --model_location ${CKPT_ROOT}/small/checkpoint.pth \
  --model_id finetune_ETTm2_small \
  --model delayformer_fixpooling \
  --data UnivariateDatasetBenchmark_finetune \
  --seq_len 1024 \
  --pred_len 96 \
  --e_layers 6 \
  --d_layers 1 \
  --patience 2 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --l2 0. \
  --dropout 0.1 \
  --train_epochs 10 \
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
done

