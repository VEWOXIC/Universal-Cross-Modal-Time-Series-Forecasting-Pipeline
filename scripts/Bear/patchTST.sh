for output_len in 24 168 336 720
do
python -u run_lightning.py \
    --model 'PatchTST' \
    --model_config 'model_configs/general/PatchTST.yaml' \
    --data Bear \
    --data_config './data_configs/Bear/fullBear_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 | tee -a ./logs/Trans.log

done