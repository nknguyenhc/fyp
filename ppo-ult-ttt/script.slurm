#!/bin/bash
#SBATCH --output=result.ppo.Qwen.Qwen3-8B.out
#SBATCH --error=result.ppo.Qwen.Qwen3-8B.out
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ppo-Qwen

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python ttt_ppo.py \
    --dataset_train_split descriptiveness \
    --response_length 2 \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 2 \
    --output_dir ttt-ppo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --torch_dtype float16 \
    --total_episodes 2000 \
    --model_name_or_path Qwen/Qwen3-8B \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32
