for output_len in 24 168 336 720
do
python -u run_lightning.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST/PatchTST-CAISO.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 256 \
    --devices '0' | tee -a ./logs/trans.log

done