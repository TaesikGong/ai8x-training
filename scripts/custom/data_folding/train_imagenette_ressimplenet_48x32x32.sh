#!/bin/sh
echo CUDA_VISIBLE_DEVICES=3 python train.py --epochs 500 --deterministic --optimizer Adam --lr 0.00064 --wd 0 --compress policies/schedule-cifar100-ressimplenet.yaml --model ai85ressimplenet --dataset Imagenette_48x32x32 --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 "$@"
