for output_len in 288 2016  # 4032 8640
do
python -u criterias_lightning.py \
    --data 'NYC_traffic_speed' \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_hetero_TGTSF.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 4320 \
    --output_len $output_len \
    --batch_size 512 \
    --device "0" | tee -a ./logs/test_m_NYC.log
done