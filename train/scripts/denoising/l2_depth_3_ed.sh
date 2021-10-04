#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ../../

python3 train_denoising.py "./config/denoising/l2_depth_3_ed.cfg" --Train.device "'2'" \
&> ./outputs/denoising/l2_depth_3_ed.log