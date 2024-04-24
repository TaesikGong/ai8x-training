#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python train.py --deterministic --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --model ai85unetlarge --out-fold-ratio 8 --use-bias --dataset CamVid_s352_c3_f8 --device MAX78002 --batch-size 32 --qat-policy policies/qat_policy_camvid.yaml --compress policies/schedule-camvid.yaml --validation-split 0 "$@"
