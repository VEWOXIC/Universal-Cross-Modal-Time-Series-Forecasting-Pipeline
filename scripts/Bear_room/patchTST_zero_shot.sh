for output_len in 12 144 288 576
do
python -u criterias_lightning.py \
    --baseline_model 'PatchTST' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_zero_shot.yaml' \
    --input_len 288 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/zero_shot.log

done