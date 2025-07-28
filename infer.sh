#!/bin/bash
#SBATCH --job-name=LPBA_CRR_Net_infer
#SBATCH --output=CRR_Net_infer_%j.log
#SBATCH --error=CRR_Net_infer_%j.log
#SBATCH --partition=GTX3090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=24:00:00
echo "Starting job at: $(date)"
python -u infer.py -m 'CRR_Net'
echo "Finished job at: $(date)"
