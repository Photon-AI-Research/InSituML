# Run training with torch in an interactive job on **crusher**
# Allocate interactive resources with
#   export WORLD_SIZE=1; salloc --time=08:00:00 --nodes=1 --ntasks=$WORLD_SIZE --gres=gpu:$WORLD_SIZE --cpus-per-task=7 --ntasks-per-gpu=1 --mem-per-gpu=64000 -p batch -A csc380 -C nvme
# execute script with
#   bash crusher_run_interactive.sh

# Load openPMD/ADIOS2/PyTorch environment

OPENPMD_ENVIRONMENT="/lustre/orion/csc380/proj-shared/openpmd_environment"
INSITUML_DIR="$HOME//src/InSituML/main/ModelHelpers/cINN"
WORK_DIR="/lustre/orion/csc380/proj-shared/ksteinig/2024-03_Training-from-Stream/job_temp/trainingOutput"

source ${OPENPMD_ENVIRONMENT}/env.sh

srun bash -c 'echo "$SLURM_PROCID: $SLURM_LOCALID: $SLURM_GPUS_ON_NODE, $SLURM_GPU_BIND, $SLURM_SHARDS_ON_NODE"' | tee out.txt

mkdir -p ${WORK_DIR}

cd ${WORK_DIR}

export MIOPEN_USER_DB_PATH="/mnt/bb/$USER"; export MIOPEN_DISABLE_CACHE=1
export MASTER_PORT=12340; export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
unset LEARN_R
export LEARN_R_AE=2
unset MIN_TB
srun -l python ${INSITUML_DIR}/ac_jr_fp_ks_openpmd-streaming-continual-learning.py --io_config ${INSITUML_DIR}/io_config_frontier_offline.py --model_config ${INSITUML_DIR}/model_config_lr-001.py 2>err.txt | tee -a out.txt

