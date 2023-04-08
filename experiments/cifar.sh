#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("frog" "bird" "cat" "deer" "dog"  "horse" "ship" "truck" "airplane" "automobile")
for i in "${arr[@]}";
do
    echo "Running CIFAR10. Anomaly Class: $i "
    python train.py --dataset cifar10 --batchsize 128 --isize 32 --niter 40 --abnormal_class $i --model skipganomaly --gen  cnn --outf cnn_4 --lr  0.0002 --dim 40
done
exit 0
