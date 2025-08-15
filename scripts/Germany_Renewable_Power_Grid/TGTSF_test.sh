for output_len in 24 168 336 720
do
python -u criterias_lightning.py \
    --data 'Germany_Renewable_Power_Grid' \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_TGTSF_H.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --device "cuda:3" | tee -a ./logs/test_IATSF.log
done