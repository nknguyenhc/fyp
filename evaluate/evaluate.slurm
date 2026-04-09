#!/bin/bash
#SBATCH --output=result.LiquidAI.LFM2-350M.out
#SBATCH --error=result.LiquidAI.LFM2-350M.out
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=eval-Liquid

java -jar ultimate-tictactoe.jar LiquidAI.LFM2-350M
