for output_len in 24 168 336 720
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Bear \
    --data_config './data_configs/Bear/fullBear_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 64 \
    --devices '0,1,4' | tee -a ./logs/IATSF.log

done