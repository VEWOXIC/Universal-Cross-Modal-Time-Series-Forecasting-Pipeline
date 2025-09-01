for output_len in 96 672 1344 2880
do
python -u run_lightning.py \
    --model 'TGTSF' \
    --model_config 'model_configs/general/TGTSF.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF.yaml' \
    --input_len 1440 \
    --output_len $output_len \
    --batch_size 256 \
    --patience 5 \
    --train_epochs 50 \
    --devices '0,1,4' | tee -a ./logs/IATSF.log

done