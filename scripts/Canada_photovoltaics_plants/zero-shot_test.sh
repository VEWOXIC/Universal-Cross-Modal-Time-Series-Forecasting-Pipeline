for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'PatchTST' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias_lightning.py \
    --baseline_model 'PatchTST' \
    --data NYC_traffic_speed \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'iTransformer' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias_lightning.py \
    --baseline_model 'iTransformer' \
    --data NYC_traffic_speed \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done


for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'DLinear' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'DLinear' \
    --data NYC_traffic_speed \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'FITS' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'FITS' \
    --data NYC_traffic_speed \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'GPT4TS' \
    --data Canada_photovoltaics_plants \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done

for output_len in 24 168 #336 720
do
python -u criterias.py \
    --model 'GPT4TS' \
    --data NYC_traffic_speed \
    --data_config './data_configs/Canada_photovoltaics_plants/fullCPP_hid.yaml' \
    --input_len 360 \
    --output_len $output_len \
    --task "TSF" | tee -a ./logs/CCPzerotest.log

done