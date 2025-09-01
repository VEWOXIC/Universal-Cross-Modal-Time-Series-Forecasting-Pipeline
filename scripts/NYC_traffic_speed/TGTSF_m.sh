for output_len in 288 2016 4032 8640
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF.yaml' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 256 \
    --patience 5 \
    --train_epochs 50 \
    --devices '0,1,4' | tee -a ./logs/IATSF.log

done