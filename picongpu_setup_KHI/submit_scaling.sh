# Submission script  of PIConGPU+Streaming+ML scaling runs
# execute by
#   bash submit_scaling.sh | tee -a submit_scaling_$(date '+%F_%H%M%S').log

INSITUML_PIC_DIR="/ccs/home/kelling/checkout/InSituML/picongpu_setup_KHI"
PIC_BUILD_DIR="/ccs/home/kelling/checkout/frontier_env/build/pic_build"

cd $INSITUML_PIC_DIR

source insituml_picongpu.profile

cd $PIC_BUILD_DIR

## Need
# export MIN_TB_FROM_UNCHANGED_NOW_BF=96
# export LEARNING_RATE
# export LEARNING_RATE_AE

export BATCH_SIZE=4

# for jobSize in {8,24,48,96}; do # for test: {8,}; do
for jobSize in {8,24,}; do # for test: {8,}; do
    for minTB in {24,}; do
        for learningRate in "1e-06"; do
            for learningRateAE in {20,}; do
                echo ""; echo "========== job size: ${jobSize} | min tb: ${minTB} | learning rate: ${learningRate} | learning rate AE mult: ${learningRateAE} =========="; echo ""
                export LEARN_R=${learningRate}
                export LEARN_R_AE=${learningRateAE}
                export MIN_TB=${minTB}
                tbg -s \
                    -t $PIC_BUILD_DIR/etc/picongpu/frontier-ornl/batch_pipe.tpl \
                    -c $PIC_BUILD_DIR/etc/picongpu/${jobSize}-nodes_streaming_bench.cfg \
                    /lustre/orion/csc621/proj-shared/kelling/runsFromScratch_2/${jobSize}-nodes_lr-${learningRate}_min-tb-${minTB}_lrAE-${learningRateAE}_bs-${BATCH_SIZE}
            done
        done
    done
done

