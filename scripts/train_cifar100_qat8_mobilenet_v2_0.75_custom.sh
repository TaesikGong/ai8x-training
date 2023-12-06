#!/bin/sh
python train.py --deterministic --epochs 2 --optimizer SGD --lr 0.1 --compress policies/schedule-cifar100-mobilenetv2.yaml --model ai85netmobilenetv2cifar100_m0_75_custom --dataset CIFAR100 --device MAX78000 --batch-size 128 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_cifar100_mobilenetv2_quick.yaml "$@"
