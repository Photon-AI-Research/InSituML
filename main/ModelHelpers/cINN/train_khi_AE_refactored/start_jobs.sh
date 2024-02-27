property=("positions" "momentum" "force" "all")
learning_rate=(1e-2 1e-5)
z_dim=(5 10 15)

for property in ${property[@]}
do
    mkdir $property
    cd $property
    lr=1e-2
    for lr in ${learning_rate[@]}
    do
        z_dim=5
        for z_dim in ${z_dim[@]}
        do

        mkdir "learning_rate"$lr"_z_dim"$z_dim
        cd "learning_rate"$lr"_z_dim"$z_dim

        jobname=$property"Training"

        echo "#!/bin/sh">>job.sh
        echo "#SBATCH -o %j.out">>job.sh
        echo "#SBATCH -e %j.err">>job.sh
        echo "#SBATCH --job-name="$jobname>>job.sh
        echo "#SBATCH --ntasks-per-node=1">>job.sh
        echo "#SBATCH --cpus-per-task=12">>job.sh
        echo "#SBATCH --account=casus">>job.sh
        echo "#SBATCH --partition=casus">>job.sh
        echo "#SBATCH --gres=gpu:1">>job.sh
        echo "#SBATCH --time=12:00:00">>job.sh

        echo "module load python cuda gcc">>job.sh
        echo "source /home/checkr99/.new_env3.10/bin/activate">>job.sh
        echo "">>job.sh


        CALL_SEQUENCE="python3 /home/checkr99/InSituML/main/ModelHelpers/cINN/train_khi_AE_refactored/main.py "
        CALL_SEQUENCE+=" --property_ "$property
        CALL_SEQUENCE+=" --learning_rate "$lr
        CALL_SEQUENCE+=" --z_dim "$z_dim

        echo $CALL_SEQUENCE>>job.sh

        #sbatch job.sh

        cd ..
       done
    done
    cd ..
done
