#!/bin/bash

model=vgg
defense=vi_SEBR_01
data=cifar100
root=../data
n_ensemble=50
steps=(15)
attack=Linf
max_norm=0,0.002,0.01,0.02
echo "Attacking" ./checkpoint/${data}_${model}_${defense}.pth
for k in "${steps[@]}"
do
    echo "running" $k "steps"
    CUDA_VISIBLE_DEVICES=0 python3 acc_under_attack.py \
        --model $model \
        --defense $defense \
        --data $data \
        --root $root \
        --n_ensemble $n_ensemble \
        --steps $k \
        --max_norm $max_norm \
        --attack $attack
done
