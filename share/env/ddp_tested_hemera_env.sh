module load gcc/12.2.0 cuda/12.1 openmpi/4.1.5-cuda121-gdr ucx/1.14.0-gdr \
	libfabric/1.17.0 cmake/3.26.1  git/2.37.1 python/3.12.2 boost/1.85.0 \
	zlib/1.2.11 c-blosc2/2.31.1 libpng/1.6.39 pngwriter/0.7.0 \
	hdf5-parallel/1.12.0-omp415-cuda121 adios2/2.9.2-cuda121-blosc2-py3122
	# openpmd/0.15.2-cuda121-blosc2-py3122
# for (re-)instaling openpmd-api
export openPMD_USE_MPI=ON
source /home/kelling/checkout/insitumlNp2Torch26Env/bin/activate
export PMIX_MCA_gds=hash
