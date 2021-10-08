#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ../../

python3 train_gaussian_assumption.py "./config/gaussian_assumption/l2_depth_4.cfg" --Train.device "'1'" \
--IO.tag "'l2_depth_4/img'" \
--Network.input_shape "(640, 640, 1)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[20, 50, 1000]" --Train.lr "[3.3e-5, 1e-5, 1e-5]" \
&> ./outputs/gaussian_assumption/l2_depth_4_img.log