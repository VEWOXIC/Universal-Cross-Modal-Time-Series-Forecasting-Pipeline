for output_len in 288 864 1440 2016
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_TGTSF.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 256 \
    --devices '0,1,4' | tee -a ./logs/IATSF.log

done