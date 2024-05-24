#!/bin/bash -l

#SBATCH -A hlab
#SBATCH --partition=hlab
#SBATCH -t 47:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

export INSITUML="${INSITUML:-.}"

. $INSITUML/share/env/ddp_tested_hemera_env.sh

export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT="1$( echo -n $SLURM_JOBID | tail -c 4 )"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo HOSTNAME $HOSTNAME
nvidia-smi

# Usage: sbatch create_metrics.sh [arg1] [arg2] [arg3] [arg4]

# particle_pathpattern = "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/24-nodes_full-picongpu-data/04-01_1013/simOutput/openPMD/simData_%T.bp5",
# radiation_pathpattern = "/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/24-nodes_full-picongpu-data/04-01_1013/simOutput/radiationOpenPMD/e_radAmplitudes_%T.bp5"

# Assign arguments to variables with default values
model_filepath_pattern=${1:-"/bigdata/hplsim/scratch/kelling/chamfers/slurm-6923925/{}"}
load_model_checkpoint=${2:-"model_24211"}
particle_pathpattern=${3:-"/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1645/simOutput/openPMD/simData_%T.bp5"}
radiation_pathpattern=${4:-"/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1645/simOutput/streamedRadiation/ts_{}.npy"}

mpirun -n 1 python $INSITUML/tools/calculate_metrics.py --model_filepath_pattern $model_filepath_pattern --load_model_checkpoint $load_model_checkpoint --particle_pathpattern $particle_pathpattern --radiation_pathpattern $radiation_pathpattern