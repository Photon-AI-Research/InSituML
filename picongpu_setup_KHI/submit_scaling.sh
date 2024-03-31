cd ~/src/InSituML/picongpu_setup_KHI

source insituml_picongpu.profile

## Need
# export MIN_TB_FROM_UNCHANGED_NOW_BF
# export LEARNING_RATE

for jobSize in {8,24,48,72,96}; do # for test: {8,}; do
    for minTB in {4,8,16}; do
        for learningRate in "0.001" "0.0005" "0.0001"; do
            echo ""; echo "========== job size: ${jobSize} | min tb: ${minTB} | learning rate: ${learningRate} =========="; echo ""
            export LEARN_R=${learningRate}
            export MIN_TB=${minTB}
            tbg -s \
                -t etc/picongpu/frontier-ornl/batch_pipe.tpl \
                -c etc/picongpu/${jobSize}-nodes_streaming_bench.cfg \
                /lustre/orion/csc380/proj-shared/ksteinig/2024-03_Training-from-Stream_chamfersdistance_fix-gpu-volume_scaling/${jobSize}-nodes_lr-${learningRate}_min-tb-${minTB} | tee submit_scaling.log
        done
    done
done

