#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=uncertainty
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0

DENOISING=nlm_0.02
SRC="[('/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_4.h5', '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/denoising/${DENOISING}/results/dose_rate_4.h5')]"
DST="[('/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_1.h5', '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/denoising/${DENOISING}/results/dose_rate_4.h5')]"
OUT="'/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/uncertainty/${DENOISING}/'"

cd ../../

python3 train_uncertainty.py "./config/uncertainty/l2_depth_4.cfg" --Train.device "'2'" \
--IO.src_datasets "${SRC}" --IO.dst_datasets "${DST}" --IO.output_dir "${OUT}" \
--IO.tag "'l2_depth_4/img'" \
--Network.input_shape "(640, 640, 2)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[20, 50, 100]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" \
&> ./outputs/uncertainty/l2_depth_4_img_${DENOISING}.log &

DENOISING=nlm_0.05
SRC="[('/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_4.h5', '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/denoising/${DENOISING}/results/dose_rate_4.h5')]"
DST="[('/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_1.h5', '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/denoising/${DENOISING}/results/dose_rate_4.h5')]"
OUT="'/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/uncertainty/${DENOISING}/'"

python3 train_uncertainty.py "./config/uncertainty/l2_depth_4.cfg" --Train.device "'3'" \
--IO.src_datasets "${SRC}" --IO.dst_datasets "${DST}" --IO.output_dir "${OUT}" \
--IO.tag "'l2_depth_4/img'" \
--Network.input_shape "(640, 640, 2)" \
--Data.num_patches_per_slice "1" --Data.num_slices_per_batch "1" \
--Train.epochs "[20, 50, 100]" --Train.lr "[1e-4, 3.3e-5, 1e-5]" \
&> ./outputs/uncertainty/l2_depth_4_img_${DENOISING}.log &
wait