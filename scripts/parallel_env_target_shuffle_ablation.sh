

for output_len in 12 144 288 576
do
    result_path="results/ParallelEnvTarget/zero_environment/input_288_output_${output_len}.json"
    log_path="logs/ParallelEnvTarget-ZeroEnv-288-${output_len}.log"
    echo "Evaluating shuffled environment: input_len=288, output_len=${output_len}"
    python -u criterias.py \
        --model ParallelEnvTarget \
        --data Bear_room \
        --data_config "data_configs/Bear_room/Bear-EnvWorld.yaml" \
        --task TSF \
        --input_len "288" \
        --output_len "${output_len}" \
        --batch_size "256" \
        --version "latest" \
        --checkpoint_base "./checkpoints/" \
        --device "0" \
        --zero_environment \
        --ablation_output "${result_path}" | tee "${log_path}"
done
