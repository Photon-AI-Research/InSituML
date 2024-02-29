In order to train the model do:

1. load openPMD environment:
   fontier: `source /lustre/orion/csc380/proj-shared/openpmd_environment/env.sh`
   hemera: `source /bigdata/hplsim/scratch/poesch58/InSituML_env/env.sh`

2. obtain a gpu (node)
   both: `getDevice`
   frontier: `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`
   hemera: `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`

3. run training by continual learning with stream loader (on single gpu):
   hemera: `mpirun -n 1 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py`
   frontier: `srun -n 1 python ac_jr_fp_ks_openpmd-streaming-continual-learning.py`

