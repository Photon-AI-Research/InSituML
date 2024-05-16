#!/bin/bash

# Default values for arguments
WORLD_SIZE=1
TORCH_RANKS_PER_NODE=1
NTASKS_PER_NODE=1
CPUS_PER_TASK=2
GPUS=1
MEM="128G"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --world-size) WORLD_SIZE="$2"; shift ;;
        --torch-ranks-per-node) TORCH_RANKS_PER_NODE="$2"; shift ;;
        --ntasks-per-node) NTASKS_PER_NODE="$2"; shift ;;
        --cpus-per-task) CPUS_PER_TASK="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        --mem) MEM="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Using WORLD_SIZE=$WORLD_SIZE"
echo "Using TORCH_RANKS_PER_NODE=$TORCH_RANKS_PER_NODE"
echo "Using NTASKS_PER_NODE=$NTASKS_PER_NODE"
echo "Using CPUS_PER_TASK=$CPUS_PER_TASK"
echo "Using GPUS=$GPUS"
echo "Using MEM=$MEM"

# Allocate resources and run job on Hemera
allocate_and_run_hemera() {
    echo "Allocating resources on Hemera..."
    srun --time=10:00:00 --ntasks-per-node=$NTASKS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --gres=gpu:$GPUS --mem=$MEM --partition=gpu --pty bash -c "
        
        echo 'Creating new environment on Hemera...'
        . ddp_tested_hemera_env.sh
        export openPMD_USE_MPI=ON

        echo 'Starting training...'
        export WORLD_SIZE=$WORLD_SIZE
        export MASTER_PORT=12340
        master_addr=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n 1)
        export MASTER_ADDR=\$master_addr
        mpirun -n $TORCH_RANKS_PER_NODE python tools/openpmd-streaming-continual-learning.py --io_config=io_config_hemera.py --type_streamer=offline
    "
}

# Start the allocation and training process
allocate_and_run_hemera
