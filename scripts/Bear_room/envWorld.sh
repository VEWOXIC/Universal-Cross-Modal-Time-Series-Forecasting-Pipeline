# for output_len in 12 144 288 576
# do
#     python -u run.py \
#         --model EnvWorld \
#         --model_config model_configs/general/EnvWorld.yaml \
#         --data Bear_room \
#         --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
#         --input_len 288 \
#         --output_len "${output_len}" \
#         --batch_size 256 \
#         --train_epochs 50 \
#         --patience 5 \
#         --loss mse \
#         --gpu 0 | tee -a "logs/EnvWorld-Bear-${output_len}-test.log"
# done

for output_len in 12 144 288 576 
do

    echo "Evaluating Bear EnvWorld: input_len=288, output_len=${output_len}"
    python -u criterias.py \
        --model EnvWorld \
        --data Bear_room \
        --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
        --version latest \
        --checkpoint_base ./checkpoints \
        --task TSF \
        --input_len 288 \
        --output_len "${output_len}" \
        --batch_size 256 \
        --device 0 | tee -a "logs/EnvWorld-Bear-test-${output_len}.log"
done