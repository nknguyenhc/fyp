#!/bin/bash
#SBATCH --output=result.google.gemma-2-2b-it.out
#SBATCH --error=result.google.gemma-2-2b-it.out
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ult-google

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python script.py google/gemma-2-2b-it
