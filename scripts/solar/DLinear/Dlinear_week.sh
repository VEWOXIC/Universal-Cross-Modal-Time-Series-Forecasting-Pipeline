python run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data solar \
    --data_config './data_configs/fullsolar.yaml' \
    --input_len 672 \
    --output_len 168 \
    --batch_size 1024 | tee -a ./logs/solar/DLinear_week.log

