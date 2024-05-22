#!/bin/bash
#SBTACH -c 4
##SBATCH -p gpu
#SBATCH -p hlab
#SBATCH -A hlab
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --ntasks=1 
#SBATCH -t 2:0:0 

# This jobscript can be used as a quick tets for overall functionality on hemera:
# * NOTE: Ensure, that the environment file (. $INSITUML/share/env/ddp_tested_hemera_env.sh) loads a working environment (modules + python venv with local inSituML install.
# * Submit from repository root: `sbatch scripts/job_hemera.sh`.
# * Observe the loss printed in slurm-*/loss_0.dat, should look like:
#    ```
#    # batch_index	time	total_loss	loss_AE	loss_IM	loss_ae_reconst	kl_loss	l_fit	l_latent	l_rev
#    0	70350	2.184997081756592	1.1215089559555054	1.0634880065917969	1.0105606317520142	0.110948346555233	10.213825225830078	450.2242431640625	603.0498657226562
#    1	158	3.1079459190368652	2.024231433868408	1.083714485168457	1.8160319328308105	0.2081994116306305	29.154804229736328	450.2120666503906	604.3475952148438
#    ...
#    ```

export INSITUML="${INSITUML:-.}"

. $INSITUML/share/env/ddp_tested_hemera_env.sh

export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT="1$( echo -n $SLURM_JOBID | tail -c 4 )"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo HOSTNAME $HOSTNAME
nvidia-smi

BATCH_SIZE="${INSITUML:-4}"
mpiexec bash scripts/ompi_CUDA_VISIBLE_DEVICES_wrapper.sh python $INSITUML/tools/openpmd-streaming-continual-learning.py --io_config $INSITUML/share/configs/io_config_hemera.py  --model_config $INSITUML/share/configs/model_config.py
# mpiexec python $INSITUML/tools/openpmd-streaming-continual-learning.py --io_config $INSITUML/share/configs/io_config_hemera.py  --model_config $INSITUML/share/configs/model_config.py --type_streamer offline
