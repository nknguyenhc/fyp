#!/bin/bash
#SBATCH --output=result.out
#SBATCH --error=result.out
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tool-call

python directly.py