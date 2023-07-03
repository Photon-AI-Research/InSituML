#!/bin/bash -l

#SBATCH -A hlab
#SBATCH --partition=hlab
#SBATCH -t 47:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -o cINN_%j.out

module load python

source /home/willma32/ml1/bin/activate
cd /home/willma32/insitu_particles/InSituML/main/ModelHelpers/cINN

python train_cINN.py