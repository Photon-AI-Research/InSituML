In order to train the model do:

1. load openPMD environment:
   fontier: `source /lustre/orion/csc380/proj-shared/openpmd_environment/env.sh`
   hemera: `source /bigdata/hplsim/scratch/poesch58/InSituML_env/env.sh`

2. obtain a gpu (node)
   both: `getDevice`
   frontier: `srun  --time=1:00:00 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=7 --ntasks-per-gpu=1 --gpu-bind=closest --mem-per-gpu=64000 -p batch -A csc380 --pty bash`
   hemera: `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`

3. Change to directory `InSituML/main/ModelHelpers/cINN` and adjust path to data in `ac_jr_fp_ks_openpmd-streaming-continual-learning.py` to:
   frontier: `/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset`
   hemera: `/bigdata/hplsim/scratch/poesch58/InSituML_env/pic_run/`

4. run training by continual learning with stream loader (on single gpu):
   frontier: `srun -n 1 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py`
   hemera: `mpirun -n 1 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py`

