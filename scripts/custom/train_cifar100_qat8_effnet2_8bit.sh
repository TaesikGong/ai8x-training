#!/bin/sh
CUDA_VISIBLE_DEVICES=4 python train.py --deterministic --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100-effnet2.yaml --model ai87effnetv2 --dataset CIFAR100 --device MAX78002 --batch-size 100 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_late_cifar.yaml "$@"
