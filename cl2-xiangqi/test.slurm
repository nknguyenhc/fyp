#!/bin/bash
#SBATCH --output=result.google.gemma-2-2b-it.out
#SBATCH --error=result.google.gemma-2-2b-it.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --job-name=cc-google

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

python test.py \
    --step pm \
    --trust_remote_code True \
    --model_name_or_path vls.google.gemma-2-2b-it
