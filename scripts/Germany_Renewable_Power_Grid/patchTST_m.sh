for output_len in 96 672 1344 2880
do
python -u run_lightning.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG.yaml' \
    --input_len 1440 \
    --output_len $output_len \
    --batch_size 512 | tee -a ./logs/Trans.log

done