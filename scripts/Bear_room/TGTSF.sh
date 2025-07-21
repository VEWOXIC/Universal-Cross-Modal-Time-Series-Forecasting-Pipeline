for output_len in 168 120 72 24
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Bear_room \
    --data_config './data_configs/Bear_room/fullBear_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 256 \
    --devices '0,3' | tee -a ./logs/IATSF.log

done