for output_len in 12 144 288 576 
do
    echo "Training ParallelEnvTarget: input_len=288, output_len=${output_len}"
    python -u run.py \
        --model ParallelEnvTarget \
        --model_config model_configs/general/ParallelEnvTarget.yaml \
        --data Bear_room \
        --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
        --input_len 288 \
        --output_len "${output_len}" \
        --batch_size 256 \
        --patience 5 \
        --train_epochs 20 \
        --zero_environment_train \
        --gpu 0 | tee -a "logs/ParallelEnvTarget-Bear-${output_len}.log"
done