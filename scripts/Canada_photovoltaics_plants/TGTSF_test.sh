for output_len in 24 168 336 720
do
python -u criterias_lightning.py \
    --data 'Canada_photovoltaics_plants' \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hetero_TGTSF.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --device "3" | tee -a ./logs/test_IATSF.log
done