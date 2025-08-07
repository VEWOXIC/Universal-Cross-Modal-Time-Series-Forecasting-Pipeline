# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data Canada_photovoltaics_plants \
#     --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data Canada_photovoltaics_plants \
#     --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Sundial' \
#     --model_config 'model_configs/FM/Sundial.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log





# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data Canada_photovoltaics_plants \
#     --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data Canada_photovoltaics_plants \
#     --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'TimeMoE' \
#     --model_config 'model_configs/FM/TimeMoE.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log




# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data Canada_photovoltaics_plants \
#     --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data Canada_photovoltaics_plants \
#     --data_config './data_configs/Canada_photovoltaics_plants/fullCPP.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Canada_photovoltaics_plants_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data Germany_Renewable_Power_Grid \
#     --data_config './data_configs/Germany_Renewable_Power_Grid/fullGRPG_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Germany_Renewable_Power_Grid_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data Jena_Atmospheric_Physics \
#     --data_config './data_configs/Jena_Atmospheric_Physics/fullJAP_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/Jena_Atmospheric_Physics_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --input_len 360 \
#     --output_len 24 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_day.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log

# python -u run_fm.py \
#     --model 'Chronos' \
#     --model_config 'model_configs/FM/Chronos.yaml' \
#     --data NYC_traffic_speed \
#     --data_config './data_configs/NYC_traffic_speed/fullNYCTS_H.yaml' \
#     --input_len 360 \
#     --output_len 168 \
#     --batch_size 1 \
#     --filtered_samples "sample_indexes/NYC_traffic_speed_sample_week.json" \
#     --gpu 1 | tee -a ./logs/test_FM_on_samples.log





python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'Chronos' \
    --model_config 'model_configs/FM/Chronos.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'TimeMoE' \
    --model_config 'model_configs/FM/TimeMoE.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 24 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_day.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log

python -u run_fm.py \
    --model 'Sundial' \
    --model_config 'model_configs/FM/Sundial.yaml' \
    --data California_ISO \
    --data_config './data_configs/California_ISO/fullCAISO_H.yaml' \
    --input_len 360 \
    --output_len 168 \
    --batch_size 1 \
    --filtered_samples "sample_indexes/California_ISO_sample_week.json" \
    --gpu 3 | tee -a ./logs/test_FM_on_samples.log