#!/bin/bash
#SBATCH --output=result.ppo.out
#SBATCH --error=result.ppo.out
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ppo

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python ttt_ppo.py \
    --dataset_train_split descriptiveness \
    --response_length 1000 \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir ttt-ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --total_episodes 2000 \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct
