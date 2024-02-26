In order to train the model do:

1. obtain a gpu node on hemera
   `srun --time=10:00:00 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 --mem=128G --partition=gpu --pty bash`

2. `bash`

3. `module load git gcc/12.2.0 cuda/12.1; conda activate pytorch-cuda`

4. `ipython -i train_MAF_khi_radiation.py` for the existing training by Jeyhun & Jeffrey
    or 
    `ipython -i ks_main.py` for the refactored version using threading by Klaus & Jeyhun & Jeffrey

OR

4. `python train_MAF_khi_radiation.py 2>err.txt | tee out.txt`

