#!/bin/bash
#SBATCH --output=meta-llama.Llama-3.1-8B-Instruct.out
#SBATCH --error=meta-llama.Llama-3.1-8B-Instruct.out
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tool-call-meta

python directly.py meta-llama/Llama-3.1-8B-Instruct