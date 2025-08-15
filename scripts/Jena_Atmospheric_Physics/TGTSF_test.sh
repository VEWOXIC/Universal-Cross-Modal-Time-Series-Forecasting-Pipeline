for output_len in 24 168 336 720
do
python -u criterias_lightning.py \
    --data 'Jena_Atmospheric_Physics' \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_hetero_TGTSF_H.yaml' \
    --baseline_model 'TGTSF' \
    --task 'TGTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len $output_len \
    --batch_size 128 \
    --device "cuda:3" | tee -a ./logs/test_IATSF.log
done