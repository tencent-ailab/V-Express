#!/bin/bash
export NO_ALBUMENTATIONS_UPDATE="1"

zero_config=./training_configs/zero2_config.json
# zero_config=./training_configs/zero2_config_bf16.json

# config=./training_configs/stage_1.yaml
# config=./training_configs/stage_2.yaml
config=./training_configs/stage_3.yaml

export CHIEF_IP=127.0.0.1
export HOST_NUM=1
export INDEX=0

export HOST_GPU_NUM=8

PROCESS_NUM=$((HOST_GPU_NUM * HOST_NUM))
echo ${PROCESS_NUM}
export NCCL_IB_DISABLE=1

accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --use_deepspeed --num_processes ${PROCESS_NUM} \
    --deepspeed_config_file $zero_config \
    --num_machines "${HOST_NUM}" --machine_rank "${INDEX}" --main_process_ip "${CHIEF_IP}" --main_process_port 21003 \
    --deepspeed_multinode_launcher standard \
    train.py  --config $config
