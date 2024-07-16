#!/bin/bash
#SBATCH -p dc-gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --ntasks=1 
#SBATCH -A training2406
#SBATCH -t 0:30:0 

PARAM_LIST_FILE="params.dat"
OBJ_LIST_FILE="objective.dat"

avg()
{
	if [ -z "$1" ]; then
		n=1
	else
		n=$1
	fi
	awk "function isnum(x){return(x==x+0)} { if(isnum(\$$n)) { sum+=\$$n; sumsq+=\$$n*\$$n ; n+=1;} } END { print sum/n, sum, n, sqrt(sumsq/n - sum*sum/n/n) }"
}


if [ -f "objective.dat" ]; then
	OBJ_JOBID=$( cat $OBJ_LIST_FILE )
	for i in $( ls slurm-$OBJ_JOBID* ); do
		echo $i
	done

	for i in slurm-$OBJ_JOBID*/loss_0.dat ; do
		tail -n 20 $i | cut -f 
	done
fi
