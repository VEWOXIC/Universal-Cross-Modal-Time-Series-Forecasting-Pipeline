for output_len in 288  # 12 144 288 576
do
python -u criterias.py \
    --model 'GPT4MTS' \
    --data Bear_room \
    --input_len 288 \
    --output_len $output_len \
    --version "08-30-1112" \
    --data_config './data_configs/Bear_room/fullBear_hetero_ablation_RPLLM.yaml' \
    --batch_size 512 \
    --device "4" \
    --channel_wise True \
    --task "TGTSF" | tee -a ./logs/channelwiseablation.log
done