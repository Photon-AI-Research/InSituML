#!/bin/bash
#SBATCH -p dc-gpu
#SBATCH -c 32
#SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH --ntasks=4 
#SBATCH -A training2406
#SBATCH -t 1:0:0 

export PARAM_LIST_FILE="params.dat"
export OBJ_LIST_FILE="objective.dat"

SLURM_NTASKS=4

export CINN=/p/home/jusers/kelling1/jureca/git/InSituML/main/ModelHelpers/cINN
export ROOT=/p/scratch/training2406/team_hechtlab_kelling/

. $ROOT/env/profile
. $ROOT/env/insituml/bin/activate

echo HOSTNAME $HOSTNAME
nvidia-smi

JOB_OFFSET=$(( $SLURM_ARRAY_TASK_ID * $SLURM_NTASKS ))

for p in $( seq 1 $SLURM_NTASKS ); do

	export TASK_ID=$(( $p - 1 ))
	export CUDA_VISIBLE_DEVICES=$TASK_ID
	export WORLD_SIZE=1
	export MASTER_PORT="1$( echo -n $SLURM_JOBID | tail -c 3 )$TASK_ID"
	master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
	export MASTER_ADDR=$master_addr

	echo $CUDA_VISIBLE_DEVICES $MASTER_ADDR $MASTER_PORT

	if [ -f $PARAM_LIST_FILE ]; then
		LINE_NUM=$(( $JOB_OFFSET + p ))
		if [ $( wc -l $PARAM_LIST_FILE ) -gt $LINE_NUM ]; then
			break
		fi
		set $( head -n $LINE_NUM $PARAM_LIST_FILE | tail -n 1 )

		export LAM_PREDICT=$1
		export LAM_LATENT=$2
		export LAM_REV=$3
		export LAM_KL=$4
		export LAM_RECON=$5
		export LR_REST=$6
		export LR_AE=$7
	fi

	srun -n 1 python $CINN/ac_jr_fp_ks_openpmd-streaming-continual-learning.py --io_config $CINN/io_config_jureca.py --model_config $CINN/model_config.py --runner srun >& slurm-$SLURM_JOBID-$SLURM_ARRAY_TASK_ID-${TASK_ID}.out &

done

wait
