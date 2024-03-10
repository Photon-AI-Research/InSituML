#!/bin/bash -l

#SBATCH -A hlab
#SBATCH --partition=hlab
#SBATCH -t 47:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -o maf_%j.out

## tell slurm to append to the output file if it exists so restarts do not
## override previous output
#SBATCH --open-mode=append


module load python/3.10.4
module load cuda/12.1
module load gcc

# source /home/rustam75/InSituML_test/examples/virtual_env/bin/activate
source /home/rustam75/ml_env/virtualenv/bin/activate

cd /home/rustam75/InSituML/main/ModelHelpers/cINN


TERM ()
{
	echo "shell received SIGTERM"
	scontrol requeue $SLURM_JOBID
	exit 0
}
# register signal handler
trap "TERM" SIGTERM

# python train_MAF_AE_khi_box_ex.py 
# python train_MAF_VAE_khi_box_ex.py
# python train_INN_AE_khi_box_ex.py
# python train_INN_VAE_khi_box_ex.py
#python train_INN_VAE_khi_box_ex-ank.py
#python train_VAE_khi_box.py
#python train_VAE_khi_box_new.py
# python train_INN_VAE_khi_box_ex-full_pretrain.py
# python train_INN_AE_khi_box_ex-ae_pretrain.py
# python -m wandb agent jeyhun/khi_public/pvt9nfb1
# python train_INN_AE_khi_box_ex-ae_pretrain-addloss.py
# python -m wandb agent jeyhun/khi_public_final/kpqz0f58
python -m wandb agent jeyhun/khi_public_final/o78s0ocr


