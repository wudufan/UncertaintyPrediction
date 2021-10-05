#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ../../

python3 train_uncertainty.py "./config/uncertainty/l2_depth_4_filter.cfg" --Train.device "'2'" \
--Filter.std "5" \
--IO.tag "'l2_depth_4/filter_5/img'" \
--Network.input_shape "(640, 640, 2)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[20, 50, 100]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" \
&> ./outputs/uncertainty/l2_depth_4_img_filter_5.log