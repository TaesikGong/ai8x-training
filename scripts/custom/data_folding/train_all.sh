#!/bin/bash

datasets="Imagenette"
sizes="3x32x32 12x32x32 48x32x32"
models="convnet5 simplenet ressimplenet"
#models="convnet5 widenet simplenet_cifar simplenet_mnist_v1"
declare -a pid_array=()
declare -a cuda_array=(0 1 2 3 4 5 6 7 8) # Adjust if you have fewer devices

# Function to find and return the first available CUDA device
find_available_cuda() {
  for i in "${!cuda_array[@]}"; do
    if [ -n "${cuda_array[i]}" ]; then
      echo $i
      return
    fi
  done
}

# Function to update CUDA array marking device as used or free
update_cuda_array() {
  local device=$1
  local status=$2 # 0 for free, 1 for used
  if [ "$status" -eq 0 ]; then
    cuda_array[$device]=$device
  else
    cuda_array[$device]=''
  fi
}

for dataset in $datasets; do

  if [ "$dataset" = "Imagenette" ]; then
    batch_size="32"
  fi

  for size in $sizes; do
    for model in $models; do
      if [ "$model" = "convnet5" ]; then
        args="--lr 0.1 --optimizer SGD --epochs 200 --deterministic --compress policies/schedule.yaml --model ai85net5 --dataset ${dataset}_${size} --confusion --param-hist --pr-curves --embedding --device MAX78000 --batch-size $batch_size"
      elif [ "$model" = "widenet" ]; then
        args="--epochs 300 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenetwide2x --dataset ${dataset}_${size} --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0 --qat-policy policies/qat_policy_cifar100.yaml --use-bias"
      elif [ "$model" = "simplenet" ]; then
        args="--epochs 600 --optimizer Adam --lr 0.00032 --wd 0 --compress policies/schedule-cifar100.yaml --model ai85simplenet --dataset ${dataset}_${size} --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0 --qat-policy None --use-bias"
      elif [ "$model" = "ressimplenet" ]; then
        args="--epochs 500 --deterministic --optimizer Adam --lr 0.00064 --wd 0 --compress policies/schedule-cifar100-ressimplenet.yaml --model ai85ressimplenet --dataset ${dataset}_${size} --device MAX78000 --batch-size $batch_size --print-freq 100 --validation-split 0"
      elif [ "$model" = "efficientnetv2" ]; then
        args="--deterministic --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-imagenet-effnet2.yaml --model ai87imageneteffnetv2 --dataset ${dataset}_${size} --device MAX78002 --batch-size $batch_size --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_imagenet.yaml"
      fi

      # Wait for a CUDA device to be available if all are currently in use
      while [ ${#pid_array[@]} -ge 8 ] || [ $(find_available_cuda) == '' ]; do
        for i in "${!pid_array[@]}"; do
          if ! kill -0 ${pid_array[i]} 2>/dev/null; then
            # Mark CUDA device as available
            update_cuda_array ${i} 0
            unset pid_array[i]
          fi
        done
        sleep 1
      done

      # Find an available CUDA device
      cuda_id=$(find_available_cuda)
      update_cuda_array $cuda_id 1

      current_time=$(date "+%y%m%d-%H%M%S")
      # Execute the script with the specified CUDA device and log output

      echo "Executing on CUDA_VISIBLE_DEVICES=$cuda_id: python train.py $args"
      CUDA_VISIBLE_DEVICES=$cuda_id python train.py $args 2>&1 | tee logs/${current_time}_${dataset}_${model}_${size}.txt &

      pid=$!
      pid_array[$cuda_id]=$pid
      sleep 1.5
    done
  done
done

# Wait for all background processes to complete
wait
