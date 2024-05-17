#!/bin/bash

### PS2 (1 2 / Food101/ 3 6 18 36 64/ simplenet widenet)
### PS4 (0 1 / Imagenette Caltech101 / 51 / simplenet widenet / data-augment)
#max_jobs_per_gpu=1
#num_workers=8

### PS3 (0 1 / Caltech256 / 3 6 18 36 64 /  widenet  / data-reshape) [done]
### PS3 (1 2 / Caltech101/ 3 6 18 36 64 /  widenet  / data-reshape) [done]
### PS3 (2 / Imagenette/ 3 6 18 36 64 /  widenet  / data-reshape) [done]
### PS3 (0 / Food101 / 3 6 18 36 64 /  widenet  / data-reshape)
### PS3 (1 / Food101 / 3 6 18 36 /  widenet  / data-reshape)
max_jobs_per_gpu=1
num_workers=8

### PS4 (1 2 / Food101/ 3 6 18 36 64/ efficientnetv2 mobilenetv2_075 / data-reshape)
### PS4 (0 1 / Imagenette Caltech101 Caltech256 / 51 / efficientnetv2 mobilenetv2_075 / data-augment)
#max_jobs_per_gpu=1
#num_workers=8

seeds="0"
datasets="Imagenette Caltech101" # Imagenette Caltech101 Caltech256 Food101 Flower102 CUB StanfordCars DTD ###  PACS-P PACS-A PACS-C PACS-S
num_channels="64" #3 6 18 36 64 ; : coordconv: 5 ; data-augment: 51
models="efficientnetv2 mobilenetv2_075" # simplenet widenet efficientnetv2 mobilenetv2_075
# ####### convnet5 ressimplenet
reshape="--data-reshape" # "--data-augment" "--coordconv" "--data-reshape"
reshape_methods="tile_per_channel random_stack" # "dex" "tile_per_channel" "random_stack"

augs="0000000000000000" #1111111111111111
#"
#0000000000000001
#0000000000000010
#0000000000000100
#0000000000001000
#0000000000010000
#0000000000100000
#0000000001000000
#0000000010000000
#0000000100000000
#0000001000000000
#0000010000000000
#0000100000000000
#0001000000000000
#0010000000000000
#0100000000000000
#1000000000000000 "
#"0000000000000000"


declare -a pid_array=()
declare -a cuda_array=(0 1 2 3 4 5 6 7) # Adjust if you have fewer devices
declare -a cuda_usage=(0 0 0 0 0 0 0 0) # Initialize with zero, tracking scripts per GPU


# Function to find and return the CUDA device with the least usage and less than 2 running tasks

find_available_cuda() {
  local min_usage=$max_jobs_per_gpu # Set initial minimal usage to max allowed jobs per GPU
  local min_index=-1 # Initialize with an invalid index

  for i in "${!cuda_usage[@]}"; do
    if [ "${cuda_usage[i]}" -lt "$max_jobs_per_gpu" ]; then
      if [ "${cuda_usage[i]}" -lt "$min_usage" ]; then
        min_usage="${cuda_usage[i]}"
        min_index=$i
      fi
    fi
  done

  if [ "$min_index" -ne -1 ]; then
    echo $min_index
    return 0 # Successfully found an available device
  else
    return 1 # Return an error if no available device is found
  fi
}

# Function to update CUDA usage tracking array
update_cuda_usage() {
  local device=$1
  local change=$2 # +1 or -1
  cuda_usage[$device]=$((${cuda_usage[$device]} + $change))
}
for seed in $seeds; do
  for dataset in $datasets; do

#  if [ "$dataset" = "Imagenette" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "Flower102" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "Food101" ]; then
#    batch_size="128"
#  elif [ "$dataset" = "Caltech101" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "CUB" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "StanfordCars" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "DTD" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "PACS-P" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "PACS-A" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "PACS-C" ]; then
#    batch_size="32"
#  elif [ "$dataset" = "PACS-S" ]; then
#    batch_size="32"
#  fi
    for num_channel in $num_channels; do
      for model in $models; do
        for aug in $augs; do
          for reshape_method in $reshape_methods; do

              common_args="--seed $seed --deterministic --workers $num_workers --validation-split 0 --dataset ${dataset}_${num_channel}x32x32 --aug $aug $reshape --data-reshape-method $reshape_method" #--batch-size $batch_size

              if [ "$model" = "convnet5" ]; then
                args=$common_args" --epochs 200 --optimizer SGD --lr 0.1 --compress policies/schedule.yaml --model ai85net5 --param-hist --pr-curves --print-freq 100 --embedding --device MAX78000 "

              elif [ "$model" = "simplenet" ]; then # 32
                # train_cifar100_qat_mixed.sh
                args=$common_args" --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenet --batch-size 32 --device MAX78000 --print-freq 100 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"

                # train_cifar100.sh (NOTE: no QAT!)
        #        args=$common_args" --epochs 600 --optimizer Adam --lr 0.00032 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenet --device MAX78000 --print-freq 100 --qat-policy None --use-bias"

              elif [ "$model" = "ressimplenet" ]; then # 32
                args=$common_args" --epochs 500 --optimizer Adam --lr 0.00064 --wd 0 --compress policies/schedule-cifar100-ressimplenet.yaml --model ai85ressimplenet --batch-size 32 --device MAX78000 --print-freq 100"

              elif [ "$model" = "widenet" ]; then #100
                args=$common_args" --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenetwide2x  --device MAX78000 --batch-size 100 --print-freq 100 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"

              elif [ "$model" = "efficientnetv2" ]; then #100
                args=$common_args" --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100-effnet2.yaml --model ai87effnetv2 --device MAX78002 --batch-size 100 --print-freq 100 --use-bias --qat-policy policies/qat_policy_late_cifar.yaml"

                ## efficientnet for imagenet
        #        args=$common_args" --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-imagenet-effnet2.yaml --model ai87imageneteffnetv2 --dataset ${dataset}_${num_channel}x112x112 --device MAX78002 --print-freq 100 --use-bias --qat-policy policies/qat_policy_imagenet.yaml"

              elif [ "$model" = "mobilenetv2_075" ]; then #128
                args=$common_args" --epochs 300 --optimizer SGD --lr 0.1 --compress policies/schedule-cifar100-mobilenetv2.yaml --model ai87netmobilenetv2cifar100_m0_75 --batch-size 128 --device MAX78002 --print-freq 100 --use-bias --qat-policy policies/qat_policy_cifar100_mobilenetv2.yaml"
              fi



            # Wait for a CUDA device to be available if all are currently fully utilized
            while true; do
              cuda_id=$(find_available_cuda)
              if [ -n "$cuda_id" ]; then
                update_cuda_usage $cuda_id 1
                break
              fi
              # Periodically check and clean up finished jobs, updating GPU usage
              for idx in "${!pid_array[@]}"; do
                if ! kill -0 ${pid_array[idx]} 2>/dev/null; then
                  update_cuda_usage $idx -1
                  unset pid_array[idx]
                fi
              done
            done


            current_time=$(date "+%y%m%d-%H%M%S")
            echo "Executing on CUDA_VISIBLE_DEVICES=$cuda_id: python train.py $args"
            CUDA_VISIBLE_DEVICES=$cuda_id python train.py $args 2>&1 | tee logs/${current_time}_${dataset}_${model}_${num_channel}_${reshape:2}_${aug}_${reshape_method}_${seed}.txt &

            pid=$!
            pid_array[$cuda_id]=$pid
            sleep 1.5

          done
        done
      done
    done
  done
done

wait