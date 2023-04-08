#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("bird" "car" "cat" "deer" "dog"  "horse" "monkey" "ship" "truck"  "airplane")
for i in "${arr[@]}";
do
    echo "Running STL10. Anomaly Class: $i "
    python train.py --dataset STL10 --batchsize 128 --isize 32 --niter 80 --abnormal_class $i --model skipganomaly --gen  cnn --outf cnn_4 --lr 0.0002 --dim 40
done
exit 0
