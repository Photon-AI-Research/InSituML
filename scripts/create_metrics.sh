#!/bin/bash -l

#SBATCH -A hlab
#SBATCH --partition=hlab
#SBATCH -t 47:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# base_path="/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/03-30_learning-rate-scaling-with-ranks_chamfersdistance_fix-gpu-volume/96-nodes_lr-0.0005_min-tb-16"
base_path="/bigdata/hplsim/scratch/kelling/chamfers/slurm-6923925"
# base_path="/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/04-01_rerun-independent-AE-scaling_chamfersdistance_fix-gpu-volume_scaling/8-nodes_lr-0.0001_min-tb-4_lrAE-20/04-01_1840"
# base_path="trained_models"
#/bigdata/hplsim/scratch/kelling/chamfers/slurm-6923925

# the plot directory path
plot_directory_path="/bigdata/hplsim/aipp/SC24_PIConGPU-Continual-Learning/metrics_and_plots/plots_slurm-6923925/$SLURM_JOB_ID"

# create the directory after the job starts
mkdir -p $plot_directory_path

# Load modules
module load python gcc git gcc/12.2.0 cuda/12.1 openmpi/4.1.5-cuda121-gdr
module load zlib/1.2.11 libfabric/1.17.0 c-blosc2 hdf5-parallel/1.12.0-omp415-cuda121 adios2/2.9.2-cuda121-blosc2 libpng/1.6.39

# Activate Python environment
source /home/rustam75/ml_env/ml_stream/bin/activate

# Change to the desired directory
cd /home/rustam75/InSituML/main/ModelHelpers/cINN/

# python calculate_metrics.py --sim "24-nodes_full-picongpu-data" --N_samples 5 --eval_timesteps 900 950 1000 --generate_best_box_plot True --generate_plots True --plot_directory_path "$plot_directory_path" --model_filepath_pattern "${base_path}/{}" --load_model_checkpoint "model_24000"

python calculate_metrics.py --sim "014" --N_samples 5 --eval_timesteps 900 950 1000 --generate_best_box_plot True --generate_plots True --plot_directory_path "$plot_directory_path" --model_filepath_pattern "${base_path}/{}" --load_model_checkpoint "model_24000"
