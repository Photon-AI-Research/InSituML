#!/bin/sh
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --job-name=ddp_run
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --account=casus
#SBATCH --partition=casus
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2

export WORLD_SIZE=2
export MASTER_PORT=12340

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=localhost
echo "MASTER_ADDR="$MASTER_ADDR

module load python gcc git gcc/12.2.0 cuda/12.1 openmpi/4.1.5-cuda121-gdr
source /home/checkr99/.new_env3.10/bin/activate

mpirun -n 2 python ac_test_example_training_buffer.py slurm
