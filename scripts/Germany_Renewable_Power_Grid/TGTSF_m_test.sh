for output_len in 96 672
do
python -u criterias_lightning.py \
    --data 'Germany_Renewable_Power_Grid' \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 1440 \
    --output_len $output_len \
    --batch_size 512 \
    --device "0" | tee -a ./logs/test_m_germany.log
done