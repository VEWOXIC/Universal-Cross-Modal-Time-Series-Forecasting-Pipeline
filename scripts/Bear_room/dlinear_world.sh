
# for output_len in 12 144 288 576 
# do
#     echo "Training Bear DLinear: target='Zone Temperature', input_len=288, output_len=${output_len}"
#     python -u run.py \
#         --model DLinear \
#         --model_config model_configs/general/DLinear.yaml \
#         --data Bear_room \
#         --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
#         --input_len 288 \
#         --output_len "${output_len}" \
#         --batch_size 256 \
#         --gpu 0  | tee -a "logs/DLinear-Bear-multivariant-${output_len}.log"
# done

for output_len in 12 144 288 576 
do

    echo "Evaluating Bear DLinear: input_len=288, output_len=${output_len}"
    python -u criterias.py \
        --model DLinear \
        --data Bear_room \
        --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
        --version latest \
        --checkpoint_base ./checkpoints \
        --task TSF \
        --input_len 288 \
        --output_len "${output_len}" \
        --batch_size 256 \
        --device 0 | tee -a "logs/DLinear-Bear-test-${output_len}.log"
done