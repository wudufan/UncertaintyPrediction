#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0

python3 train.py error_x0.cfg --Train.device "'2'" &> outputs/error_x0.log &
python3 train.py mean.cfg --Train.device "'3'" &> outputs/mean.log &
wait

python3 train.py error.cfg --Train.device "'3'" &> outputs/error.log