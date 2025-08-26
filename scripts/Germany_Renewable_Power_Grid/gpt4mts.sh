for output_len in 24 # 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --hf_mirror True \
    --gpu 1 \
    --batch_size 128 # | tee -a ./logs/Linear.log
    
done