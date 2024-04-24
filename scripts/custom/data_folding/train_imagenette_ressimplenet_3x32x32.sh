#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python train.py --epochs 500 --deterministic --optimizer Adam --lr 0.00064 --wd 0 --compress policies/schedule-cifar100-ressimplenet.yaml --model ai85ressimplenet --dataset Imagenette_3x32x32 --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 "$@"
