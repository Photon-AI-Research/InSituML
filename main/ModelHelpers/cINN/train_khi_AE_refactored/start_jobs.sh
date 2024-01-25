property=("positions" "momentum" "force" "all")
property=("positions" "momentum")

for property in ${property[@]}
do

mkdir $property
cd $property

jobname="Training"$property

echo "#!/bin/sh">>job.sh
echo "#SBATCH -o %j.out">>job.sh
echo "#SBATCH -e %j.err">>job.sh
echo "#SBATCH --job-name="$jobname>>job.sh
echo "#SBATCH --ntasks-per-node=1">>job.sh
echo "#SBATCH --cpus-per-task=1">>job.sh
echo "#SBATCH --account=casus">>job.sh
echo "#SBATCH --partition=casus">>job.sh
echo "#SBATCH --gres=gpu:1">>job.sh
echo "#SBATCH --time=12:00:00">>job.sh

echo "module load python cuda gcc">>job.sh
echo "source /home/checkr99/.new_env3.10/bin/activate">>job.sh
echo "">>job.sh


CALL_SEQUENCE="python3 /home/checkr99/InSituML/main/ModelHelpers/cINN/train_khi_AE_refactored/main.py "
CALL_SEQUENCE+=" --property_ "$property

echo $CALL_SEQUENCE>>job.sh

#sbatch job.sh

cd ..
done
