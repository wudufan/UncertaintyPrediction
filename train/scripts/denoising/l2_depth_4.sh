#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ../../

python3 train_denoising.py "./config/denoising/l2_depth_4.cfg" --Train.device "'1'" \
--IO.tag "'l2_depth_4'" \
--Network.input_shape "(640, 640, 1)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[20, 50, 100]" --Train.lr "[3.3e-5, 1e-5, 1e-5]" \
&> ./outputs/denoising/l2_depth_4.log