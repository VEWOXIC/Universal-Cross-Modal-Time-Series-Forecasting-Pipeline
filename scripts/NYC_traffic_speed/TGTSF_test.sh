for output_len in 720
do
python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF_H.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --device "4" | tee -a ./logs/test_IATSF.log
done