for output_len in 168 # 336 720 24
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF/TGTSF-CAISO.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --patience 10 \
    --train_epochs 50 \
    --devices '3' | tee -a ./logs/IATSF.log

done