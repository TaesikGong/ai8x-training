#!/bin/sh
python train.py --epochs 2 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenetwide2x --dataset CIFAR100 --device MAX78000 --batch-size 100 --print-freq 100 --validation-split 0 --qat-policy policies/qat_policy_cifar100_quick.yaml --use-bias "$@"
