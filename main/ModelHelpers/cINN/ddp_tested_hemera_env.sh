module load python gcc git gcc/12.2.0 cuda/12.1 openmpi/4.1.5-cuda121-gdr
module load zlib/1.2.11   libfabric/1.17.0  c-blosc2  hdf5-parallel/1.12.0-omp415-cuda121  adios2/2.9.2-cuda121-blosc2 libpng/1.6.39

source /bigdata/hplsim/aipp/ankush/.new_env3.10/bin/activate

export PMIX_MCA_gds=hash
