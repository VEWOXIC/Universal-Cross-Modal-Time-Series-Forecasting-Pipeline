for output_len in 144 1008
do
python -u criterias_lightning.py \
    --data 'Jena_Atmospheric_Physics' \
    --data_config './data_configs/California_ISO/fullCAISO_hetero_TGTSF.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 2160 \
    --output_len $output_len \
    --batch_size 512 \
    --device "0" | tee -a ./logs/test_m_jena.log
done