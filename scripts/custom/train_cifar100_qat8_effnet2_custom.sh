#!/bin/sh
python train.py --deterministic --epochs 2 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100-effnet2.yaml --model ai85effnetv2_custom --dataset CIFAR100 --device MAX78000 --batch-size 100 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_quick_cifar.yaml "$@"
