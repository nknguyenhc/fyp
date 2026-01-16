#!/bin/bash
#SBATCH --output=result.ppo.meta-llama.Llama-3.1-8B-Instruct.out
#SBATCH --error=result.ppo.meta-llama.Llama-3.1-8B-Instruct.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200-141:1
#SBATCH --job-name=ppo-Qwen

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python ttt_ppo.py \
    --dataset_train_split descriptiveness \
    --response_length 2 \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir meta-llama.Llama-3.1-8B-Instruct \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --total_episodes 10000 \
    --trust_remote_code True \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --use_peft \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_target_modules q_proj k_proj v_proj o_proj
