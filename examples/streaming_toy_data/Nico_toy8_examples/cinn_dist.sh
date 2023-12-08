#!/bin/bash
#SBATCH -A csc372
#SBATCH -J rocm5.6.0-nodes-2-gpus-8
#SBATCH -o %x-%j.out
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 2 
#SBATCH -C nvme
#SBATCH --threads-per-core=2
#SBATCH --gpus-per-node 8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vineethg@udel.edu

module load rocm/5.6.0
#./miniconda_crusher/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda init bash
conda activate tml3

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

#for BLOCKS in 2 4 6 8
#do  
#    echo $BLOCKS
#    srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$head_node_ip:29400 train_cINN_distributed_toy8.py --coupling_blocks $BLOCKS
#    #torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_cINN_distributed_toy8.py --coupling_blocks $BLOCKS
#    #torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_cINN_distributed_toy8.py --coupling_blocks $BLOCKS
#    #torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_cINN_distributed_toy8.py --coupling_blocks $BLOCKS
#done


srun -N $SLURM_JOB_NUM_NODES -n $SLURM_GPUS_PER_NODE torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_JOB_NUM_NODES --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$head_node_ip:29400 multi_node_train_cINN_distributed_toy8.py

