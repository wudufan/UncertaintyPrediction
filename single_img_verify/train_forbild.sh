#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0

python3 train.py error_x0.cfg --Train.device "'2'" \
--IO.x0 "'forbild/x0.nii'" --IO.var_roi_map "'forbild/variance.seg.nrrd'" --IO.tag "'forbild/error_x0'" \
--Network.input_shape "(512,512,1)" \
&> outputs/forbild_error_x0.log &

python3 train.py mean.cfg --Train.device "'3'" \
--IO.x0 "'forbild/x0.nii'" --IO.var_roi_map "'forbild/variance.seg.nrrd'" --IO.tag "'forbild/mean'" \
--Network.input_shape "(512,512,1)" \
&> outputs/forbild_mean.log &
wait

python3 train.py error.cfg --Train.device "'3'" \
--IO.x0 "'forbild/x0.nii'" --IO.var_roi_map "'forbild/variance.seg.nrrd'" --IO.tag "'forbild/error'" \
--IO.checkpoint "'/home/dwu/trainData/uncertainty_prediction/single_img_verify/forbild/mean/100.h5'" \
--Network.input_shape "(512,512,2)" \
&> outputs/forbild_error.log