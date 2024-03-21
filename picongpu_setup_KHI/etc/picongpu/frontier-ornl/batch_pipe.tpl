#!/usr/bin/env bash
# Copyright 2013-2023 Axel Huebl, Richard Pausch, Rene Widera, Sergei Bastrakov, Klaus Steinger, Franz Poeschel
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#


# PIConGPU batch script for crusher's SLURM batch system

#SBATCH --account=!TBG_nameProject
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes_adjusted
### #SBATCH --gpu-bind=closest
#SBATCH --chdir=!TBG_dstPath
# The following two parameters are needed to avoid batch system hangups
# when trying to run two sub-jobs asynchronously.
#SBATCH -S 0
#SBATCH --exclusive # This one might not be strictly necessarily, haven't tested
#SBATCH --network=single_node_vni,job_vni
#SBATCH -C nvme

###################
# Optional params #
###################
##SBATCH -C nvme
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress

######################
# Unnecessary params #
######################
#SBATCH --partition=!TBG_queue

#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##
.TBG_queue="batch"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${PROJID:-""}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numDevicesForPIConGPUPerNode=4

# number of CPU cores to block per GPU
# we have 8 CPU cores per GPU (64cores/8gpus ~ 8cores)
# If we take only seven of them, we have 8 leftover CPUs for one further
# CPU-only task
#
.TBG_coresPerGPU=7
.TBG_coresPerPipeInstance=7

# Keep data transport MPI for now.
# "fabric" works as well with the latest ADIOS2 master, but there might still be issues.
# If MPI is not available and fabric is available, ADIOS2 will fall back to that automatically anyway.
.TBG_DataTransport=fabric

# Assign one OpenMP thread per available core per GPU (=task)
export OMP_NUM_THREADS=!TBG_coresPerGPU

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numDevicesForPIConGPUPerNode ] ; then echo $TBG_numDevicesForPIConGPUPerNode; else echo $TBG_tasks; fi)

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

# oversubscribe the node allocation by N per thousand
# The default can be overwritten by setting the environment variable PIC_NODE_OVERSUBSCRIPTION_PT
.TBG_node_oversubscription_pt=${PIC_NODE_OVERSUBSCRIPTION_PT:-2}

# adjust number of nodes for fault tolerance adjustments
.TBG_nodes_adjusted=$((!TBG_nodes * (1000 + !TBG_node_oversubscription_pt) / 1000))
.TBG_tasks_adjusted=$((!TBG_nodes_adjusted * !TBG_numDevicesForPIConGPUPerNode))

## end calculations ##

echo 'Start job with !TBG_nodes_adjusted nodes. Required are !TBG_nodes nodes.'

if (( !TBG_nodes < 2 )); then
    cat << EOF >&2
Frontier does not reliably initialize its networking capabilities for
single-node jobs. For a reliable streaming implementation, please adjust the
.cfg file such that the job uses at least two nodes.
EOF
    exit 1
fi

cat << EOF >&2
WARNING: Scalable streaming with ADIOS2 SST (currently) requires the MPI-based
implementation of SST. Due to changes in the networking capabilities
of Frontier, the CMake configuration of ADIOS2 does (currently) not detect its
availability and requires manual intervention:

  diff --git a/cmake/DetectOptions.cmake b/cmake/DetectOptions.cmake
  -    if (ADIOS2_HAVE_MPI_CLIENT_SERVER)
  -      set(ADIOS2_SST_HAVE_MPI TRUE)
  -    endif()
  +    set(ADIOS2_SST_HAVE_MPI TRUE)

EOF

cd !TBG_dstPath

set -o pipefail

export MODULES_NO_OUTPUT=1
source !TBG_profile
if [ $? -ne 0 ] ; then
    echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
    exit 1
fi
unset MODULES_NO_OUTPUT

compressed_env="${PREFIX}.tar.gz"

# Create a folder for putting our binaries inside the NVMe on every node.
srun -N !TBG_nodes_adjusted --ntasks-per-node=1 mkdir -p "/mnt/bb/$USER/sync_bins"
# Use sbcast to put the launch binaries and their libraries to node-local storage.
# Note that this puts only the Python binary itself, but not its runtime dependency to node-local storage.
# That seems to be the lesser problem anyway.
for binary in "!TBG_dstPath/input/bin/"{picongpu,cuda_memtest,cuda_memtest.sh,mpiInfo} "$(which python)" "$compressed_env"; do
    dest="/mnt/bb/$USER/sync_bins/${binary##*/}"
    sbcast "$binary" "$dest"
    if [ ! "$?" == "0" ]; then
        # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
        # your application may pick up partially complete shared library files, which would give you confusing errors.
        echo "SBCAST failed for $binary!"
        exit 1
    fi
done

srun -N !TBG_nodes_adjusted --ntasks-per-node=1 tar -xzf "/mnt/bb/$USER/sync_bins/local.tar.gz" --directory "/mnt/bb/$USER/"
export LD_LIBRARY_PATH="/mnt/bb/$USER/local/lib:/mnt/bb/$USER/local/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
export PATH="/mnt/bb/$USER/sync_bins/:$PATH"
for python_version in "/mnt/bb/$USER/local/"lib{64,}/python*; do
    if [[ ! -d "$python_version" ]]; then
        continue
    fi
    export PYTHONPATH="$python_version/site-packages:$PYTHONPATH"
done

verify_synced() {
    ls "/mnt/bb/$USER/local/"* -lisah
    ls "/mnt/bb/$USER/sync_bins/"* -lisah
    echo
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "PYTHONPATH=$PYTHONPATH"

    which picongpu
    ldd `which picongpu`
    srun python -c 'import openpmd_api; print("OPENPMD PATH:", openpmd_api)'
}
verify_synced | sed 's|^|>\t|'

# Remove duplicates from the LD_LIBRARY_PATH to speed up application launches
# at large scale.
# Additionally remove parallel filesystems from the LD_LIBRARY_PATH
clean_duplicates_stable()
{
    # read lines from stdin
    local already_seen=()
    while read line; do
        if [[ -n "$(echo "$line" | grep -E '^(/ccs/home|/lustre/orion)')" ]]; then
            continue
        fi
        local skip=0
        for possible_duplicate in "${already_seen[@]}"; do
            if [[ "$line" = "$possible_duplicate" ]]; then
                skip=1
                break
            fi
        done
        if [[ "$skip" = 0 ]]; then
            echo "$line"
            already_seen+=("$line")
        fi
    done
}

# remove parallel filesystems from PYTHONPATH
export PYTHONPATH="$(echo "$PYTHONPATH" | sed -E 's_:(/lustre/orion|/ccs/home)[^:]*__g')"
export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr : '\n' | clean_duplicates_stable | tr '\n' :)"
echo -e "LD_LIBRARY_PATH after cleaning duplicate entries:\n>\t$(echo "$LD_LIBRARY_PATH" | sed -E 's|:|\n>\t|g')"

echo "BEGIN DATE: $(date)"

# set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# number of broken nodes
n_broken_nodes=0

# return code of cuda_memcheck
node_check_err=1

if [ -f /mnt/bb/$USER/sync_bins/cuda_memtest ] && [ !TBG_numDevicesForPIConGPUPerNode -eq !TBG_mpiTasksPerNode ] ; then
    run_cuda_memtest=1
else
    run_cuda_memtest=0
fi

.TBG_numHostedDevicesPerNode=8

# test if cuda_memtest binary is available and we have the node exclusive
if [ $run_cuda_memtest -eq 1 ] ; then
    touch bad_nodes.txt
    n_tasks=$((!TBG_nodes_adjusted * !TBG_numHostedDevicesPerNode))
    for((i=0; ($n_tasks >= !TBG_tasks) && ($node_check_err != 0); ++i)) ; do
        n_tasks_last_check=$n_tasks
        mkdir "cuda_memtest_$i"
        cd "cuda_memtest_$i"
        # Run cuda_memtest (HIP version) to check GPU's health
        echo "GPU memtest started with $n_tasks tasks. Required are !TBG_tasks tasks."
        test $n_broken_nodes -ne 0 && exclude_nodes="-x../bad_nodes.txt"
        # do not bind to any GPU, else we can not use the local MPI rank to select a GPU
        # - test always all except the broken nodes
        # - catch error to avoid that the batch script stops processing in case an error happened
        node_check_err=$(srun -n $n_tasks --nodes=$((n_tasks / !TBG_numHostedDevicesPerNode)) $exclude_nodes -K1 --gpu-bind=none /mnt/bb/$USER/sync_bins/cuda_memtest.sh && echo 0 || echo 1)
        cd ..
        ls -1 "cuda_memtest_$i" | sed -n -e 's/cuda_memtest_\([^_]*\)_.*/\1/p' | sort -u >> ./bad_nodes.txt
        n_broken_nodes=$(cat ./bad_nodes.txt | sort -u | wc -l)
        n_tasks=$(((!TBG_nodes_adjusted - n_broken_nodes) * !TBG_numHostedDevicesPerNode))
        # if cuda_memtest not passed and we have no broken nodes something else went wrong
        if [ $node_check_err -ne 0 ] ; then
            if [ $n_tasks_last_check -eq $n_tasks ] ; then
                echo "cuda_memtest: Number of broken nodes has not increased but for unknown reasons cuda_memtest reported errors." >&2
                break
            fi
            if [ $n_broken_nodes -eq 0 ] ; then
                echo "cuda_memtest: unknown error" >&2
                break
            else
                echo "cuda_memtest: "$n_broken_nodes" broken node(s) detected!. The test will be repeated with healthy nodes only." >&2
            fi
        fi
    done
    echo "GPU memtest with $n_tasks tasks finished with error code $node_check_err."
else
    echo "Note: GPU memory test was skipped as no binary 'cuda_memtest' available or compute node is not exclusively allocated. This does not affect PIConGPU, starting it now" >&2
fi

# Note: chunk distribution strategies are not yet mainlined in openPMD
# This env variable is hence currently a no-op, except if you use
# this branch/PR: https://github.com/openPMD/openPMD-api/pull/824
# Otherwise, the current distribution strategy in openpmd-pipe is to simply
# subdivide each dataset into n equal-sized hyperslabs.
# As a consequence, communication will not necessarily happen within a node.
# If however using that branch, the following environment variable advises the
# openpmd-pipe script to load only data chunks originating from the same
# host(name), using a bin-packing approach within hosts (irrelevant here since
# there is only one reader per host), and failing with a runtime error if not
# all chunks could be distributed this way (either because of bad scheduling or
# because the writer did not send hostname information). Instead of "fail", a
# secondary fallback strategy could be specified here
# (e.g. hostname_binpacking_binpacking).

# Implicit node-level aggregation via streaming is independent from distribution
# strategy that is used, since the deciding factor is that there is only one
# instance of openpmd-pipe writing data per node. It does not matter that this
# data does not necessarily stem from the same node.
export OPENPMD_CHUNK_DISTRIBUTION=hostname_roundrobin_discard

export MPICH_OFI_NIC_POLICY=NUMA # The default

# Increase the launch timeout to 10 minutes.
# Default is 3 minutes.
export PMI_MMAP_SYNC_WAIT_TIME=1200

# Create a little launch script to set some rank-specific environment variables
cat << EOF > ./tmp.sh
#!/usr/bin/env bash

# https://docs.olcf.ornl.gov/_images/Frontier_Node_Diagram.jpg
# Each PIConGPU instance runs on one L3 cache group
devices=(cxi2 cxi1 cxi3 cxi0)
index_within_node=\$(( PMI_RANK % 4 ))
export FABRIC_IFACE="\${devices[\$index_within_node]}"

"\$@"
EOF

chmod +x ./tmp.sh
sbcast ./tmp.sh /mnt/bb/$USER/sync_bins/launch.sh
rm ./tmp.sh

TORCH_SCRATCH="/mnt/bb/$USER"

mkdir -p $TORCH_SCRATCH

## Say torch to write to scratch directory
export MIOPEN_USER_DB_PATH="$TORCH_SCRATCH"
export MIOPEN_DISABLE_CACHE=1

insituml=/autofs/nccs-svm1_home1/fpoeschel/git-repos/InSituML

oldpwd="$(pwd)"
pushd "${insituml%/*}"
tar -czf "$oldpwd/insituml.tar.gz" "${insituml##*/}"
popd
sbcast insituml.tar.gz "/mnt/bb/$USER/insituml.tar.gz"
srun -N !TBG_nodes_adjusted --ntasks-per-node=1 tar -xzf "/mnt/bb/$USER/insituml.tar.gz" --directory "/mnt/bb/$USER/"

if [ $node_check_err -eq 0 ] || [ $run_cuda_memtest -eq 0 ] ; then
    ##################
    ## Run InSituML ##
    ##################
    echo "Start InSituML"
    test $n_broken_nodes -ne 0 && exclude_nodes="-x./bad_nodes.txt"

    echo '!TBG_inconfig_pipe' | tee inconfig.json

    mkdir -p openPMD

    export MPICH_OFI_CXI_PID_BASE=0
    # L3 cache groups 1 3 5 7
    #mask="0x101010101010101"
    #mask="0xff,0xff00,0xff0000,0xff000000"
# Do not forget to add the srun option
#    --cpu-bind=verbose,mask_cpu:$mask \
# once we know how the binding is done

    srun -l                                       \
      --ntasks !TBG_tasks                         \
      --nodes !TBG_nodes                          \
      $exclude_nodes                              \
      --ntasks-per-node=4                         \
      --gpus-per-node=4                           \
      --cpus-per-task=!TBG_coresPerPipeInstance   \
      --network=single_node_vni,job_vni           \
      /mnt/bb/$USER/sync_bins/launch.sh           \
      /mnt/bb/$USER/sync_bins/python              \
        "/mnt/bb/$USER/InSituML/main/ModelHelpers/cINN/ac_jr_fp_ks_openpmd-streaming-continual-learning.py" \
        > ../training.out 2> ../training.err              &

    sleep 1


    ##################
    ## Run PIConGPU ##
    ##################
    echo "Start PIConGPU."
    # Need threaded MPI for SST on ECP systems with MPI backend of ADIOS2 SST.
    # This might or might not be needed with libfabric-based SST which will be available with ADIOS2 v2.10.
    export PIC_USE_THREADED_MPI=MPI_THREAD_MULTIPLE
    export MPICH_OFI_CXI_PID_BASE=$((MPICH_OFI_CXI_PID_BASE+1))
    # L3 cache groups 2 4 6 8
    #mask=0xfe,0xfe00,0xfe0000,0xfe000000,0xfe00000000,0xfe0000000000,0xfe000000000000,0xfe00000000000000
    #mask="0xff00000000,0xff0000000000,0xff000000000000,0xff00000000000000"
# Do not forget to add the srun option
#    --cpu-bind=verbose,mask_cpu:$mask \
# once we know how the binding is done

    srun -l                                 \
      --overlap                             \
      --ntasks !TBG_tasks                   \
      --nodes=!TBG_nodes                    \
      $exclude_nodes                        \
      --ntasks-per-node=!TBG_devicesPerNode \
      --gpus-per-node=!TBG_devicesPerNode   \
      --cpus-per-task=!TBG_coresPerGPU      \
      --network=single_node_vni,job_vni     \
      -K1                                   \
      /mnt/bb/$USER/sync_bins/launch.sh     \
      /mnt/bb/$USER/sync_bins/picongpu      \
        !TBG_author                         \
        !TBG_programParams                  \
        > ../pic.out 2> ../pic.err          &

    wait
else
    echo "Job stopped because of previous issues."
    echo "Job stopped because of previous issues." >&2
fi

echo "END DATE: $(date)"
