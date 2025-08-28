for output_len in 12 144 288 576
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_RPLLM.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --hf_offline True \
    --gpu 7 \
    --batch_size 256 | tee -a ./logs/RPLLM5.log
done