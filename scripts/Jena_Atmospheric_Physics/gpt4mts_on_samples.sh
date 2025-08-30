python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'GPT4MTS' \
    --task 'MTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "5" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" | tee -a ./logs/test_RPLLM_on_samples.log

python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'GPT4MTS' \
    --task 'MTSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "5" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" | tee -a ./logs/test_RPLLM_on_samples.log