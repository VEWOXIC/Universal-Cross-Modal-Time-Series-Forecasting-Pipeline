for output_len in 12 144 288 576 
# for output_len in 576 
do

    echo "Evaluating Bear ParallelEnvTarget: input_len=288, output_len=${output_len}"
    python -u criterias.py \
        --model ParallelEnvTarget \
        --data Bear_room \
        --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
        --version latest \
        --checkpoint_base ./checkpoints \
        --task TSF \
        --input_len 288 \
        --output_len "${output_len}" \
        --batch_size 256 \
        --shuffle_environment \
        --shuffle_repeats 3 \
        --shuffle_seed 2026 \
        --device 0 | tee -a "logs/ParallelEnvTarget-Bear-test-${output_len}.log"
done
