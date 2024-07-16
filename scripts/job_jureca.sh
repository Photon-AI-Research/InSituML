#!/bin/bash
#SBTACH -c 4
#SBATCH -p dc-gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --ntasks=1 
#SBATCH -A training2406
#SBATCH -t 2:0:0 

CINN=/p/home/jusers/kelling1/jureca/git/InSituML/main/ModelHelpers/cINN
ROOT=/p/scratch/training2406/team_hechtlab_kelling/

. $ROOT/env/profile
. $ROOT/env/insituml/bin/activate

export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT="1$( echo -n $SLURM_JOBID | tail -c 4 )"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo HOSTNAME $HOSTNAME
nvidia-smi

export BATCH_SIZE=4
echo BATCH_SIZE $BATCH_SIZE
srun python $CINN/ac_jr_fp_ks_openpmd-streaming-continual-learning.py --io_config $CINN/io_config_jureca.py --model_config $CINN/model_config.py --runner srun
