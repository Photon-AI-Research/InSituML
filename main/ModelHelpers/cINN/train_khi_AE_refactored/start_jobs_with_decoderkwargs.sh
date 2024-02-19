property=("positions")

learning_rate=(1e-4)
z_dim=(128 200 400)
project_keyword="position_runs_long_decoder2"

count=0

for property in ${property[@]}
do
    mkdir $property"6"
    cd $property"6"
    lr=2e-3
    for lr in ${learning_rate[@]}
    do
        z_dim=128
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
        if [ $count -le 9 ]
        then
          echo "#SBATCH --partition=gpu">>job.sh
        else
          echo "#SBATCH --partition=casus">>job.sh
        fi
        ##echo "#SBATCH --partition=gpu">>job.sh
        echo "#SBATCH --gres=gpu:1">>job.sh
        echo "#SBATCH --time=48:00:00">>job.sh

        echo "module load python cuda gcc">>job.sh
        echo "source /home/checkr99/.new_env3.10/bin/activate">>job.sh
        echo "">>job.sh

        CALL_SEQUENCE="python3 /home/checkr99/InSituML/main/ModelHelpers/cINN/train_khi_AE_refactored/main.py "
        CALL_SEQUENCE+=" --property_ "$property
        CALL_SEQUENCE+=" --learning_rate "$lr
        CALL_SEQUENCE+=" --z_dim "$z_dim
        CALL_SEQUENCE+=" --lossfunction chamfersloss_o "
        CALL_SEQUENCE+=" --project_kw "$project_keyword
        CALL_SEQUENCE+=" --use_encoding_in_decoder True "
        CALL_SEQUENCE+=" --particles_to_sample 150000 "
        CALL_SEQUENCE+=" --decoder_kwargs \"{'layer_config':'[128,256,512]','add_batch_normalisation':True}\" "
        CALL_SEQUENCE+=" --encoder_kwargs \"{'ae_config':'non_deterministic'}\" "

        echo $CALL_SEQUENCE>>job.sh

        sbatch job.sh
        sleep 0.1
        ((count++))

        cd ..
       done
    done
    cd ..
done
