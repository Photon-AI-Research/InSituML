#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --account=hai_sciml
#SBATCH --partition=develbooster
#SBATCH --time=00:10:00

project_dir=/p/project/hai_sciml
pic_env_dir="$project_dir"/pic_env
venv_dir="$project_dir"/pic_venv

[ -x "$(command -v deactivate)" ] && deactivate
module purge

if ! [ -d "$pic_env_dir" ]; then
    echo "$pic_env_dir could not be found. Please create a PIConGPU " \
         "environment at that location or adjust \`pic_env_dir\`."
    exit 1
fi
source "$pic_env_dir"/env.sh

if ! [ -d "$venv_dir" ]; then
    echo "$venv_dir could not be found. Please create a Python virtual " \
         "environment at that location or adjust \`venv_dir\`."
    exit 1
fi
source "$venv_dir"/bin/activate

num_nodes="$SLURM_JOB_NUM_NODES"

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ]; then
    # Append "i" to hostname on Jülich machines to connect across
    # InfiniBand cells.
    master_addr="$master_addr"i
    # Use WandB in offline mode or disable it because we don't have
    # internet access. We assume the `WANDB_API_KEY` envvar has been
    # set if the user owns a WandB account.
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_MODE=offline
    else
        export WANDB_MODE=disabled
    fi
fi
master_port=29501

export OMP_NUM_THREADS=1

cp_dir=./checkpoints
mkdir -p "$cp_dir"

# Set `lr=0` to use learning rate from config.
lr=0
python3 -m torch.distributed.run \
        --nnodes="$num_nodes" \
        --nproc_per_node=4 \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_endpoint="$master_addr:$master_port" \
        --rdzv_backend=c10d \
        main/Application.py \
        --modelPath "$cp_dir" \
        --datasetName pc_field \
        --nTasks 1 \
        --batchSize 1 \
        --lr "$lr" \
        --datasetPath main/ModelHelpers/cINN/test_cfg.txt
