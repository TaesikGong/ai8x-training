#!/bin/bash

datasets="Food101" #Caltech101 Imagenette Flower102 Food101
num_channels="3 12 48 64" #3 12 48 64
models="simplenet ressimplenet widenet efficientnetv2 mobilenetv2_075" # simplenet ressimplenet widenet efficientnetv2 mobilenetv2_075 ######## convnet5

declare -a pid_array=()
declare -a cuda_array=(0 1 2 3 4 5 6 7) # Adjust if you have fewer devices
declare -a cuda_usage=(0 0 0 0 0 0 0 0) # Initialize with zero, tracking scripts per GPU
max_jobs_per_gpu=2

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
for dataset in $datasets; do

  if [ "$dataset" = "Imagenette" ]; then
    batch_size="32"
  elif [ "$dataset" = "Flower102" ]; then
    batch_size="32"
  elif [ "$dataset" = "Food101" ]; then
    batch_size="128"
  elif [ "$dataset" = "Caltech101" ]; then
    batch_size="32"
  fi

  for num_channel in $num_channels; do
    for model in $models; do
      if [ "$model" = "convnet5" ]; then
        args="--deterministic --workers 8 --epochs 200 --optimizer SGD --lr 0.1  --compress policies/schedule.yaml --model ai85net5 --dataset ${dataset}_${num_channel}x32x32 --param-hist --pr-curves  --print-freq 100 --embedding --device MAX78000 --batch-size $batch_size --validation-split 0"

      elif [ "$model" = "simplenet" ]; then
        # train_cifar100_qat_mixed.sh
        args="--deterministic --workers 8 --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenet --dataset ${dataset}_${num_channel}x32x32 --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"

        # train_cifar100.sh (NOTE: no QAT!)
#        args="--deterministic --workers 8 --epochs 600 --optimizer Adam --lr 0.00032 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenet --dataset ${dataset}_${num_channel}x32x32 --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0 --qat-policy None --use-bias"

      elif [ "$model" = "ressimplenet" ]; then
        args="--deterministic --workers 8 --epochs 500 --optimizer Adam --lr 0.00064 --wd 0 --compress policies/schedule-cifar100-ressimplenet.yaml --model ai85ressimplenet --dataset ${dataset}_${num_channel}x32x32 --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0"

      elif [ "$model" = "widenet" ]; then
        args="--deterministic --workers 8 --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenetwide2x --dataset ${dataset}_${num_channel}x32x32 --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"

      elif [ "$model" = "efficientnetv2" ]; then
        args="--deterministic --workers 8 --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100-effnet2.yaml --model ai87effnetv2 --dataset ${dataset}_${num_channel}x112x112 --device MAX78002 --batch-size $batch_size --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_late_cifar.yaml"

        ## efficientnet for imagenet
#        args="--deterministic --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-imagenet-effnet2.yaml --model ai87imageneteffnetv2 --dataset ${dataset}_${num_channel}x112x112 --device MAX78002 --batch-size $batch_size --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_imagenet.yaml"

      elif [ "$model" = "mobilenetv2_075" ]; then
        args="--deterministic --workers 8 --epochs 300 --optimizer SGD --lr 0.1 --compress policies/schedule-cifar100-mobilenetv2.yaml --model ai87netmobilenetv2cifar100_m0_75 --dataset ${dataset}_${num_channel}x32x32 --device MAX78002 --batch-size $batch_size --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_cifar100_mobilenetv2.yaml"
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
      CUDA_VISIBLE_DEVICES=$cuda_id python train.py $args 2>&1 | tee logs/${current_time}_${dataset}_${model}_${num_channel}.txt &

      pid=$!
      pid_array[$cuda_id]=$pid
      sleep 1.5



    done
  done
done

wait