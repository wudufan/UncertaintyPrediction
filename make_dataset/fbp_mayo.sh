#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=fbp_mayo
#SBATCH --nodelist=gpu-node008
#SBATCH --cpus-per-task=16
#SBATCH --time=0


echo "1/10" &>> fbp_mayo_20210120_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L067_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210120_0.log &
echo "2/10" &>> fbp_mayo_20210120_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L096_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20210120_1.log &
echo "3/10" &>> fbp_mayo_20210120_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L109_full_sino" --dose_rate "6" --device "2" &>> fbp_mayo_20210120_2.log &
echo "4/10" &>> fbp_mayo_20210120_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L143_full_sino" --dose_rate "6" --device "3" &>> fbp_mayo_20210120_3.log &
wait
echo "4/10"
echo "5/10" &>> fbp_mayo_20210120_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L192_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210120_0.log &
echo "6/10" &>> fbp_mayo_20210120_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L286_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20210120_1.log &
echo "7/10" &>> fbp_mayo_20210120_2.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L291_full_sino" --dose_rate "6" --device "2" &>> fbp_mayo_20210120_2.log &
echo "8/10" &>> fbp_mayo_20210120_3.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L310_full_sino" --dose_rate "6" --device "3" &>> fbp_mayo_20210120_3.log &
wait
echo "8/10"
echo "9/10" &>> fbp_mayo_20210120_0.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L333_full_sino" --dose_rate "6" --device "0" &>> fbp_mayo_20210120_0.log &
echo "10/10" &>> fbp_mayo_20210120_1.log
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/deep_denoiser_ensemble/data/mayo/dose_rate_6" --name "L506_full_sino" --dose_rate "6" --device "1" &>> fbp_mayo_20210120_1.log &
wait
cat fbp_mayo_20210120_0.log fbp_mayo_20210120_1.log fbp_mayo_20210120_2.log fbp_mayo_20210120_3.log > fbp_mayo_20210120.log
rm fbp_mayo_20210120_0.log
rm fbp_mayo_20210120_1.log
rm fbp_mayo_20210120_2.log
rm fbp_mayo_20210120_3.log
