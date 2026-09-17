python -u run_llm.py \
    --model Time-R1 \
    --model_config 'model_configs/LLM/UniModal/Time-R1.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --hf_mirror True \
    --eval_mode local \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" # | tee -a ./logs/Time-R1.log

python -u run_llm.py \
    --model Time-R1 \
    --model_config 'model_configs/LLM/UniModal/Time-R1.yaml' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_hetero_LLM.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --hf_mirror True \
    --eval_mode local \
    --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" # | tee -a ./logs/Time-R1.log
