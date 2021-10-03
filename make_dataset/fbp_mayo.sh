#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=fbp_mayo
#SBATCH --nodelist=gpu-node007
#SBATCH --cpus-per-task=16
#SBATCH --time=0


echo "0/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L067_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L067_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "2/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L096_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L096_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "4/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L109_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L109_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "6/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L143_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L143_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "8/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L192_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L192_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "10/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L286_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L286_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "12/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L291_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L291_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "14/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L310_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L310_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "16/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L333_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L333_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
echo "18/20"
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1" --name "L506_full_sino" --dose_rate "1" --device "2" &>> fbp_mayo_20211003_0.log &
python3 fbp_mayo.py --input_dir "/home/dwu/data/lowdoseCTsets/" --geometry "/home/dwu/data/lowdoseCTsets/geometry.cfg" --N0 "1e5" --imgNorm "0.019" --output_dir "/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_4" --name "L506_full_sino" --dose_rate "4" --device "3" &>> fbp_mayo_20211003_1.log &
wait
cat fbp_mayo_20211003_0.log fbp_mayo_20211003_1.log > fbp_mayo_20211003.log
rm fbp_mayo_20211003_0.log
rm fbp_mayo_20211003_1.log
