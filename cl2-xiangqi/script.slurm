#!/bin/bash
#SBATCH --output=result.ppo.meta-llama.Llama-3.1-8B-Instruct.out
#SBATCH --error=result.ppo.meta-llama.Llama-3.1-8B-Instruct.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100-96:1
#SBATCH --job-name=pm-1e4-4-meta

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python script.py \
    --dataset_train_split descriptiveness \
    --step pm \
    --response_length 1 \
    --learning_rate 5e-5 \
    --num_ppo_epochs 4 \
    --num_mini_batches 4 \
    --output_dir vls.pm.meta-llama.Llama-3.1-8B-Instruct \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --total_episodes 10000 \
    --trust_remote_code True \
    --model_name_or_path ./vls.meta-llama.Llama-3.1-8B-Instruct \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj
