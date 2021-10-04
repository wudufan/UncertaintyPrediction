#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ../../

python3 train_uncertainty.py "./config/uncertainty/l2_depth_4.cfg" --Train.device "'2'" \
--IO.tag "'l2_depth_4/img'" \
--Network.input_shape "(640, 640, 2)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[20, 50, 100]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" \
&> ./outputs/uncertainty/l2_depth_4_img.log &

python3 train_uncertainty.py "./config/uncertainty/l2_depth_4.cfg" --Train.device "'3'" \
--IO.tag "'l2_depth_4/patch'" \
--Network.input_shape "(64, 64, 2)" --Data.patch_size "(64, 64, 2)" \
--Data.num_patches_per_slice "25" --Data.num_slices_per_batch "4" \
--Train.epochs "[80, 200, 400]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" --Train.save_freq "20" \
&> ./outputs/uncertainty/l2_depth_4_patch.log &
wait