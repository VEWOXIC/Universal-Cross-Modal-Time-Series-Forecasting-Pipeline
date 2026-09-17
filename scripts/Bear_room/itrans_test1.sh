
# for output_len in 12 144 288 576 
# do
#     echo "Training Bear iTransformer: target='Zone Temperature', input_len=288, output_len=${output_len}"
#     python -u run_lightning.py \
#         --model iTransformer \
#         --model_config model_configs/general/iTransformer.yaml \
#         --data Bear_room \
#         --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
#         --input_len 288 \
#         --output_len "${output_len}" \
#         --batch_size 128 \
#         --device 0  | tee -a "logs/iTrans-Bear-multivariate-${output_len}.log"
# done

for output_len in 12 144 288 576
do
    echo "Evaluating Bear iTransformer: input_len=288, output_len=${output_len}"
    python -u criterias_lightning.py \
        --data Bear_room \
        --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
        --baseline_model iTransformer \
        --task TSF \
        --version 'latest' \
        --checkpoint_base ./checkpoints \
        --input_len 288 \
        --output_len ${output_len} \
        --batch_size 256 \
        --device 0 | tee -a "logs/iTransformer-Bear-test-${output_len}.log"
done