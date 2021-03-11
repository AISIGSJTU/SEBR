#!/bin/bash

lr=0.01
sigma_0=0.15
init_s=0.15
data=cifar100
root=../data
model=vgg
model_out=./checkpoint/${data}_${model}_vi_SEBR_01
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0 python3 ./main_vi_SEBR.py \
                        --lr ${lr} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #--resume \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
