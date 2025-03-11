if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/traffic" ]; then
    mkdir ./logs/traffic
fi

python run.py \
    --model 'DLinear' \
    --model_config 'model_configs/general/DLinear.yaml' \
    --data solar \
    --data_config './data_configs/fulltraffic.yaml' \
    --input_len 168 \
    --output_len 24 \
    --batch_size 1024 | tee -a ./logs/traffic/DLinear_day.log
    