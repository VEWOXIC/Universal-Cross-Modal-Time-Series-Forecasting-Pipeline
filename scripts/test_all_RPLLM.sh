for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'GPT4TS' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/GPT4TS.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'GPT4TS' \
    --data Germany_Renewable_Power_Grid \
    --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/GPT4TS.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'GPT4TS' \
    --data Jena_Atmospheric_Physics \
    --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/GPT4TS.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'GPT4TS' \
    --data NYC_traffic_speed \
    --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/GPT4TS.log

done