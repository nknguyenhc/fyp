#!/bin/bash
#SBATCH --output=result.pm.LiquidAI.LFM2-350M.out
#SBATCH --error=result.pm.LiquidAI.LFM2-350M.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --job-name=pm-Liquid

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python piece_movement.py \
    --dataset_train_split descriptiveness \
    --response_length_1 1 \
    --response_length_2 3 \
    --learning_rate_1 5e-5 \
    --learning_rate_2 5e-5 \
    --num_ppo_epochs_1 1 \
    --num_ppo_epochs_2 1 \
    --num_mini_batches 4 \
    --output_dir pm.LiquidAI.LFM2-350M \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --total_episodes_1 10000 \
    --total_episodes_2 10000 \
    --trust_remote_code True \
    --model_name_or_path LiquidAI/LFM2-350M \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj
