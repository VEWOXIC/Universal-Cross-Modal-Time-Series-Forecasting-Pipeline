for output_len in 288 864 1440 2016
do
python -u run_lightning.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/Trans.log

done