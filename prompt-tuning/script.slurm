#!/bin/bash
#SBATCH --output=result.prompt-tuning.google.gemma-2-2b-it.out
#SBATCH --error=result.prompt-tuning.google.gemma-2-2b-it.out
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=prompt-google

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python script.py google/gemma-2-2b-it
