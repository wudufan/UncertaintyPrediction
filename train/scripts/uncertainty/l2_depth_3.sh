#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


cd ../../

python3 train_uncertainty.py "./config/uncertainty/l2_depth_3.cfg" --Train.device "'2'" \
--IO.tag "'l2_depth_3/img'" \
--Network.input_shape "(640, 640, 2)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[2, 4, 25]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" \
&> ./outputs/uncertainty/l2_depth_3/img.log &

python3 train_uncertainty.py "./config/uncertainty/l2_depth_3.cfg" --Train.device "'3'" \
--IO.tag "'l2_depth_3/patch'" \
--Network.input_shape "(64, 64, 2)" --Data.patch_size "(64, 64, 2)" \
--Data.num_patches_per_slice "25" --Data.num_slices_per_batch "4" \
--Train.epochs "[10, 25, 100]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" \
&> ./outputs/uncertainty/l2_depth_3/patch.log &
wait