#!/bin/sh
# same as the original train_cifar100_qat8_mobilenet_v2_0.75_fullbit.sh, but added CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=0 python train.py --deterministic --epochs 300 --optimizer SGD --lr 0.1 --compress policies/schedule-cifar100-mobilenetv2.yaml --model ai87netmobilenetv2cifar100_m0_75 --dataset CIFAR100 --device MAX78002 --batch-size 128 --print-freq 100 --validation-split 0 --use-bias --qat-policy None "$@"
