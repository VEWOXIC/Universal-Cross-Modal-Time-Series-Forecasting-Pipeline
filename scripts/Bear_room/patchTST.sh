# for output_len in 12 144 288 576
# do
# python -u run_lightning.py \
#     --model 'PatchTST' \
#     --model_config 'model_configs/general/PatchTST/PatchTST-Bear.yaml' \
#     --data Bear_room \
#     --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
#     --input_len 288 \
#     --output_len $output_len \
#     --batch_size 128 \
#     --patience 5 \
#     --train_epochs 50 \
#     --devices '0' | tee -a "logs/PatchTST-Bear-multivariant-${output_len}.log"
# done

for output_len in 12 144 288 576
do
    echo "Evaluating Bear PatchTST: input_len=288, output_len=${output_len}"
    python -u criterias_lightning.py \
        --data Bear_room \
        --data_config data_configs/Bear_room/Bear-EnvWorld.yaml \
        --baseline_model PatchTST \
        --task TSF \
        --version 'latest' \
        --checkpoint_base ./checkpoints \
        --input_len 288 \
        --output_len ${output_len} \
        --batch_size 256 \
        --device 0 | tee -a "logs/PatchTST-Bear-test-${output_len}.log"
done