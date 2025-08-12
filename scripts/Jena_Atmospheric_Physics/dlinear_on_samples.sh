python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" | tee -a ./logs/test_linear_on_samples.log

python -u criterias.py \
    --data 'Jena_Atmospheric_Physics' \
    --model 'DLinear' \
    --task 'TSF' \
    --version 'latest' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --device "cuda:2" \
    --evaluate_mode "all_samples" \
    --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" | tee -a ./logs/test_linear_on_samples.log