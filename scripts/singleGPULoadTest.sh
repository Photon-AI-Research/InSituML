#!/bin/bash/

# source env or call from working env

export WORLD_SIZE=1
export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

for BATCH_SIZE in 2 4 8 12 16 20 24; do
	export BATCH_SIZE
	for NUM_PART_GPU in 50000 100000 150000; do
		export NUM_PART_GPU
		echo $BATCH_SIZE | tr '\n' '\t'
		echo $NUM_PART_GPU | tr '\n' '\t'
		OUT=singleGPULoadTest_${BATCH_SIZE}_${NUM_PART_GPU}.log
		mpirun -n 1 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py --io_config singleGPULoadTest_io_config.py > $OUT
		grep "Total elapsed time:" $OUT | cut -d ' ' -f 4
	done
done
