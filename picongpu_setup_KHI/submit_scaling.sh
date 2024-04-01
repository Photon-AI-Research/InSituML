cd ~/src/InSituML/picongpu_setup_KHI

source insituml_picongpu.profile

## Need
# export MIN_TB_FROM_UNCHANGED_NOW_BF
# export LEARNING_RATE

for jobSize in {24,}; do # for test: {8,}; do
    for minTB in {16, 32}; do
        for learningRate in "0.0001"; do
            for learningRateAE in 10 15 20 25 30 40 60 80 100; do
                echo ""; echo "========== job size: ${jobSize} | min tb: ${minTB} | learning rate: ${learningRate} | learning rate AE mult: ${learningRateAE} =========="; echo ""
                export LEARN_R=${learningRate}
                export LEARN_R_AE=${learningRateAE}
                export MIN_TB=${minTB}
                tbg -s \
                    -t etc/picongpu/frontier-ornl/batch_pipe.tpl \
                    -c etc/picongpu/${jobSize}-nodes_streaming_bench.cfg \
                    /lustre/orion/csc380/proj-shared/ksteinig/2024-03_Training-from-Stream_chamfersdistance_fix-gpu-volume_scaling/${jobSize}-nodes_lr-${learningRate}_min-tb-${minTB}_lrAE-${learningRateAE} | tee submit_scaling.log
            done
        done
    done
done

