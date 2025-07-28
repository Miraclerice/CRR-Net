#!/bin/bash
#SBATCH --job-name=LPBA_CRR_Net_train
#SBATCH --output=CRR_Net_train_%j.log
#SBATCH --error=CRR_Net_train_%j.log
#SBATCH --partition=GTX3090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=48:00:00
echo "Starting job at: $(date)"
python -u train.py -m 'CRR_Net'
echo "Finished job at: $(date)"
