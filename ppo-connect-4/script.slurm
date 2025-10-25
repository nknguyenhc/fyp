#!/bin/bash
#SBATCH --output=result.ppo.LiquidAI.LFM2-350M.out
#SBATCH --error=result.ppo.LiquidAI.LFM2-350M.out
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ppo-Liquid

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python c_ppo.py \
    --dataset_train_split descriptiveness \
    --response_length 2 \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 2 \
    --output_dir LiquidAI.LFM2-350M \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --torch_dtype float16 \
    --total_episodes 5000 \
    --trust_remote_code True \
    --model_name_or_path LiquidAI/LFM2-350M \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj
