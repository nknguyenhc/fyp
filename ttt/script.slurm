#!/bin/bash
#SBATCH --output=result.openai.gpt-oss-20b.out
#SBATCH --error=result.openai.gpt-oss-20b.out
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ttt

echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python tictactoe.py openai/gpt-oss-20b
