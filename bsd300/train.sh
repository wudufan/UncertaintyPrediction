#!/bin/bash

DEPTH=1
NOISE=50
EPOCHS=500

for DEPTH in 1
do
    for NOISE in 50
    do
        for ((SEED = 0; SEED < 20; SEED++))
        do
            python train.py --device 0 --loss_norm 2 --noise_std_1 $NOISE --noise_std_2 $NOISE --epochs $EPOCHS --seed $SEED &
            python train.py --device 1 --loss_norm 2 --noise_std_1 $NOISE --noise_std_2 0 --epochs $EPOCHS --seed $SEED &
            python train.py --device 2 --loss_norm 1 --noise_std_1 $NOISE --noise_std_2 $NOISE --epochs $EPOCHS --seed $SEED &
            python train.py --device 3 --loss_norm 1 --noise_std_1 $NOISE --noise_std_2 0 --epochs $EPOCHS --seed $SEED &
            wait
        done
    done
done