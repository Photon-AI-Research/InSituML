#!/bin/sh
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --job-name=ddp_run
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=casus
#SBATCH --partition=casus
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4

export WORLD_SIZE=8
export MASTER_PORT=12340

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ddp_tested_hemera_env.sh
source /home/checkr99/.new_env3.10/bin/activate

mpirun -n 8 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py --runner mpirun --type_streamer offline
