#!/bin/bash
#SBATCH --output=result.google.gemma-2-2b-it.out
#SBATCH --error=result.google.gemma-2-2b-it.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --job-name=ult-google

echo "GPUs: $CUDA_VISIBLE_DEVICES"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python scripts/test.py \
    --mode ppo \
    --model_name_or_path google/gemma-2-2b-it \
    --trust_remote_code True \
    --game ult-ttt
