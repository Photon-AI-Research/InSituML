#!/usr/bin/env sh

cp_dir=./checkpoints
mkdir -p "$cp_dir"

# Set `lr=0` to use learning rate from config.
lr=0
python3 main/Application.py \
        --modelPath "$cp_dir" \
        --datasetName pc_field \
        --nTasks 1 \
        --batchSize 1 \
        --lr "$lr" \
        --datasetPath main/ModelHelpers/cINN/test_cfg.txt
