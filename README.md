In order to train the model do:


1. Obtain a gpu (node)
   ADAPT: `WORLD_SIZE` (=1 works always), `--time=02:00:00` for `WORLD_SIZE=1`, project ID in `-A` option.
   ATTENTION: Slurm option `--gpu-bind=closest` only works when requesting a full node`. When using less GPUs per node, dropping the option helps to avoid NCCL errors!
   - frontier: `export WORLD_SIZE=8; salloc --time=00:20:00 --nodes=1 --ntasks=$WORLD_SIZE --gres=gpu:$WORLD_SIZE --cpus-per-task=7 --ntasks-per-gpu=1 --mem-per-gpu=64000 -p batch -q debug -A csc621 -C nvme`
  - hemera: `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`

2. Load openPMD environment:
   ADAPT: Path to openPMD environment.
   - frontier: `source /lustre/orion/csc621/proj-shared/openpmd_environment/env.sh`
   - hemera: create a new environment
      ```bash
	  . $INSITUML/share/env/ddp_tested_hemera_env.sh
	  export openPMD_USE_MPI=ON
	  pip install -r requirements.txt
	  ```

3. Adjust path to offline PIConGPU data in `$INSITUML/share/configs/io_config.py` (`pathpattern1` and `pathpattern2` (already there, but commented-out)) to
      - frontier path to data with 08GPUs: `/lustre/orion/csc621/world-shared/ksteinig/008_KHI_withRad_randomInit_8gpus/simOutput`
      - frontier path to data with 16GPUs: `/lustre/orion/csc621/world-shared/ksteinig/016_KHI_withRad_randomInit_16gpus/simOutput`
      - frontier path to data with 32GPUs: `/lustre/orion/csc621/world-shared/ksteinig/002_KHI_withRad_randomInit_data-subset`
   * `streaming_config` to `None` (to train from file),
   * path to pre-trained model in `$INSITUML/share/configs/io_config.py` should not need to be adjusted.
   * **There is also `$INSITUML/share/configs/io_config_frontier_offline.py`** which has these settings,
     see below.

4. Run training in an interactive job by continual, distributed learning with stream loader.
   Number of GPUs for training depends on `WORLD_SIZE` in step 1. above:
   ADAPT: Path to InSituML repo

   - frontier:
   ```bash
   $ cd /lustre/orion/csc621/proj-shared/ksteinig/2024-03_Training-from-Stream/job_temp # change directory to arbitrary temporary directory
   $ export MIOPEN_USER_DB_PATH="/mnt/bb/$USER"; export MIOPEN_DISABLE_CACHE=1
   $ export MASTER_PORT=12340; export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   $ srun python $INSITUML/tools/openpmd-streaming-continual-learning.py --io_config $INSITUML/share/configs/io_config_frontier_offline.py 2>err.txt | tee out.txt
   ```
   Add `--type_streamer 'streaming'` to work from file but using the `StreamingLoader` instead of `RandomLoader`.
   This will aid testing the streaming, continual, distributed learning workflow without requiring a PIConGPU simulation producing the data at the same time.
   `RandomLoader` will send a random permutations of all data to all ranks.
   All data is send to all ranks, i.e. the number of epochs/work increase with number GPUs.
   Can be controlled in `io_config.py` with the `num_epochs` parameter in `streamLoader_config`, which may be fractional.

   - hemera: use `$INSITUML/scripts/job_hemera.sh`, or
   - hemera:
   ```bash
   export WORLD_SIZE=<number of global torch ranks>
   export MASTER_PORT=12340
   master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   export MASTER_ADDR=$master_addr
   cd $INSITUML
   mpirun -n <torch ranks per node> python tools/openpmd-streaming-continual-learning.py --io_config=share/configs/io_config_hemera.py --type_streamer=offline
   ```
   `--type_streamer` may be `streaming` or `offline`.

## Parameters for `openpmd-streaming-continual-learning.py`
|arg | description | values |
| --- | --- | --- |
|`--io_config`| IO-related config (producer/training buffer/model paths) | e.g. `$INSITUML/share/configs/io_config.py` (default), `$INSITUML/share/configs/io_config_frontier_offline.py`, `$INSITUML/share/configs/io_config_hemera.py` |
|`--model_config` | model hyper parameters | `$INSITUML/share/configs/model_config.py` |
