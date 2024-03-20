In order to train the model do:

1. obtain a gpu (node)
   both: `getDevice`
   frontier: `NUM_GPUS=1; salloc --time=10:00 --nodes=1 --ntasks=$NUM_GPUS --gres=gpu:$NUM_GPUS --cpus-per-task=7 --ntasks-per-gpu=1 --gpu-bind=closest --mem-per-gpu=64000 -p batch -q debug -A csc380`
   hemera: `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`

2. load openPMD environment:
   fontier: `CURRENT_DIR=$PWD; cd /lustre/orion/csc380/proj-shared/openpmd_environment; source env.sh; cd $CURRENT_DIR`
   hemera: `CURRENT_DIR=$PWD; cd /bigdata/hplsim/scratch/poesch58/InSituML_env; source env.sh; cd $CURRENT_DIR`

3. Change to directory `InSituML/main/ModelHelpers/cINN` and adjust path to data and path to pre-trained model in `ac_jr_fp_ks_openpmd-streaming-continual-learning.py` to:
   frontier path to data: `/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset`
   frontier path to model: `/autofs/nccs-svm1_home1/ksteinig/src/InSituML/main/ModelHelpers/cINN/trained_models/{}/best_model_` (not world readable, have your own git clone!)
   hemera: `/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/`

4. run training in an interactive job by continual learning with stream loader (on single gpu):

   frontier:
   ```bash
   $ cd /lustre/orion/csc380/proj-shared/ksteinig/2024-03_Training-from-Stream/job_temp
   $ export MIOPEN_USER_DB_PATH="$(pwd)"; export MIOPEN_DISABLE_CACHE=1
   $ srun -n 1 python ~/src/InSituML/main/ModelHelpers/cINN/ac_jr_fp_ks_openpmd-streaming-continual-learning.py
   ```
   hemera: `mpirun -n 1 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py`

