In order to train the model do:

1. obtain a gpu (node)
   both: `getDevice`
   frontier: `NUM_GPUS=1; salloc --time=10:00 --nodes=1 --ntasks=$NUM_GPUS --gres=gpu:$NUM_GPUS --cpus-per-task=7 --ntasks-per-gpu=1 --gpu-bind=closest --mem-per-gpu=64000 -p batch -q debug -A csc380`
   hemera: `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`

2. load openPMD environment:
   fontier: `CURRENT_DIR=$PWD; cd /lustre/orion/csc380/proj-shared/openpmd_environment; source env.sh; cd $CURRENT_DIR`
   hemera: `CURRENT_DIR=$PWD; cd /bigdata/hplsim/scratch/poesch58/InSituML_env; source env.sh; cd $CURRENT_DIR`
   * To create a new environment on hemera:
      ```bash
	  . ddp_tested_hemera_env.sh
	  export openPMD_USE_MPI=ON
	  pip install -r requirements_hemera.txt
	  ```

3. Change to directory `InSituML/main/ModelHelpers/cINN` and
   * adjust path to data in `io_config.py` (`pathpattern1` and `pathpattern2` (already there, but commented-out)) to
	 frontier path to data: `/lustre/orion/csc380/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset`
   * `streamin_config` to `None` (to train from file),
   * path to pre-trained model in `io_config.py` to should not need to be adjusted.

4. run training in an interactive job by continual learning with stream loader (on single gpu):

   frontier:
   ```bash
   $ cd /lustre/orion/csc380/proj-shared/ksteinig/2024-03_Training-from-Stream/job_temp
   $ export MIOPEN_USER_DB_PATH="$(pwd)"; export MIOPEN_DISABLE_CACHE=1
   $ srun python ~/src/InSituML/main/ModelHelpers/cINN/ac_jr_fp_ks_openpmd-streaming-continual-learning.py
   ```
   Add `--type_streamer offline` to work from file with any number of GPUs.
   This will send a random permutations of all data to all ranks.  All data is
   send to all ranks, i.e. the number of epochs/work increase with number GPUs.
   Can be controlled with in `io_config.py` in `streamLoader_config` with the
   `num_epochs` parameter, which may be fractional.

   hemera:
   ```bash
   export WORLD_SIZE=<number of global torch ranks>
   export MASTER_PORT=12340
   master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   export MASTER_ADDR=$master_addr
   mpirun -n <torch ranks per node> python ac_jr_fp_ks_openpmd-streaming-continual-learning.py --io_config=io_config_hemera.py --type_streamer=offline`
   ```
   `--type_streamer` may be `streaming` or `offline`.
