#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0

python3 train_gaussian_assumption.py gaussian_assumption.cfg \
--Train.device "'2'" \
--IO.x0 "'forbild/x0.nii'" \
--IO.var_roi_map "'forbild/variance.seg.nrrd'" \
--IO.tag "'forbild/gaussian_assumption'" \
--Network.input_shape "(512,512,1)" \
&> outputs/forbild_gaussian_assumption.log