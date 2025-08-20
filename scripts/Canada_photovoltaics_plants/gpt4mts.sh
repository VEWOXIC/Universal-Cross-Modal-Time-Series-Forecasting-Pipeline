for output_len in 24 # 168 336 720
do
python -u run.py \
    --model 'GPT4MTS' \
    --model_config 'model_configs/general/GPT4MTS.yaml' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task 'TGTSF' \
    --hf_mirror True \
    --gpu 1 \
    --batch_size 128 # | tee -a ./logs/Linear.log
    
done