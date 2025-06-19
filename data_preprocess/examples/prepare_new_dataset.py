"""
Example script demonstrating how to prepare and add a new dataset to the time series forecasting framework.

This script:
1. Demonstrates how to process raw time series data into the expected format
2. Creates the necessary configuration files
3. Shows how to integrate the dataset with the framework
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_synthetic_data(num_stations=3, days=365, save_dir="example_data"):
    """
    Generate synthetic time series data for demonstration purposes.
    
    Args:
        num_stations: Number of different time series (stations) to generate
        days: Number of days of data to generate
        save_dir: Directory to save the generated data
    
    Returns:
        save_path: Path where the data was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamps
    start_date = datetime(2021, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(24 * days)]
    
    # Dictionary to store metadata
    id_info = {}
    
    # Generate data for each station
    for station_id in range(1, num_stations + 1):
        # Create a seasonal pattern with different characteristics for each station
        seasonal_component = 10 + 5 * np.sin(2 * np.pi * np.arange(24 * days) / (24 * 365) + station_id)
        
        # Daily pattern
        daily_component = 3 * np.sin(2 * np.pi * np.arange(24 * days) / 24 + station_id * 0.5)
        
        # Random noise
        noise = np.random.normal(0, 1, 24 * days)
        
        # Trend
        trend = 0.01 * station_id * np.arange(24 * days)
        
        # Combine components
        values = seasonal_component + daily_component + noise + trend
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': timestamps,
            'value': values
        })
        
        # Some missing values to simulate real data
        missing_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'value'] = np.nan
        
        # Simulate a sensor downtime period
        downtime_start = start_date + timedelta(days=150 + station_id * 10)
        downtime_end = downtime_start + timedelta(days=5)
        df.loc[(df['date'] >= downtime_start) & (df['date'] <= downtime_end), 'value'] = np.nan
        
        # Save to parquet file
        file_path = os.path.join(save_dir, f'id_{station_id}.parquet')
        df.to_parquet(file_path)
        
        # Add metadata
        id_info[f'{station_id}'] = {
            'description': f'Synthetic station {station_id}',
            'sensor_downtime': {
                '1': {
                    'time': [downtime_start.strftime('%Y-%m-%d %H:%M:%S'), 
                             downtime_end.strftime('%Y-%m-%d %H:%M:%S')]
                }
            }
        }
        
        # Create some visualizations
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['value'])
        plt.title(f'Synthetic Time Series - Station {station_id}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'station_{station_id}_plot.png'))
        plt.close()
    
    # Save id_info.json
    with open(os.path.join(save_dir, 'id_info.json'), 'w') as f:
        json.dump(id_info, f, indent=2)
    
    print(f"Generated synthetic data for {num_stations} stations in: {save_dir}")
    print(f"Each station has {days} days of hourly data")
    
    # Also generate heterogeneous data (e.g., weather forecasts)
    generate_heterogeneous_data(save_dir, start_date, days)
    
    return save_dir

def generate_heterogeneous_data(save_dir, start_date, days):
    """Generate synthetic heterogeneous data (e.g., weather forecasts)"""
    hetero_dir = os.path.join(save_dir, 'hetero')
    os.makedirs(hetero_dir, exist_ok=True)
    
    # Generate daily weather forecasts
    for day in range(days):
        date = start_date + timedelta(days=day)
        data = {}
        
        # For each day, generate a weather forecast
        for hour in range(24):
            timestamp = date + timedelta(hours=hour)
            timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')
            
            # Synthetic weather data
            data[timestamp_str] = {
                "temperature": round(15 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2), 1),
                "humidity": round(60 + 20 * np.sin(2 * np.pi * hour / 24 + np.pi) + np.random.normal(0, 5), 1),
                "precipitation": max(0, np.random.normal(0, 0.5)),
                "forecast_time": timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Save to JSON file
        file_path = os.path.join(hetero_dir, f'weather_forecast_{date.strftime("%Y%m%d")}.json')
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Create static information
    static_info = {
        'general_info': 'This dataset contains synthetic time series data with weather forecasts as heterogeneous data.',
        'downtime_prompt': 'The sensor was down for maintenance or due to technical issues.',
        'channel_info': {}
    }
    
    # Add channel information
    for station_id in range(1, 4):  # Assuming 3 stations
        static_info['channel_info'][f'{station_id}'] = f'Station {station_id} measures synthetic values at hourly intervals.'
    
    # Save static information
    with open(os.path.join(hetero_dir, 'static_info.json'), 'w') as f:
        json.dump(static_info, f, indent=2)
    
    print(f"Generated heterogeneous data in: {hetero_dir}")

def create_data_config(data_dir, use_hetero=True):
    """
    Create data configuration YAML file.
    
    Args:
        data_dir: Directory containing the dataset
        use_hetero: Whether to include heterogeneous data
    
    Returns:
        config_path: Path to the created configuration file
    """
    # Basic configuration
    config = f"""# Synthetic dataset configuration
root_path: {data_dir}
spliter: timestamp
split_info:
  - '2021-09-01'
  - '2021-12-01'
timestamp_col: date
target: 
  - value
id_info: id_info.json
id: all
formatter: 'id_{{i}}.parquet'
sampling_rate: 1h
"""
    
    # Add heterogeneous data configuration if requested
    if use_hetero:
        hetero_config = f"""hetero_info:
  sampling_rate: 1day
  root_path: {os.path.join(data_dir, 'hetero')}
  formatter: weather_forecast_????.json
  matching: single
  input_format: json
  static_path: static_info.json
"""
        config += hetero_config
    
    # Create directory if it doesn't exist
    os.makedirs('data_configs', exist_ok=True)
    
    # Determine file name based on whether heterogeneous data is included
    config_name = 'synthetic_hetero.yaml' if use_hetero else 'synthetic.yaml'
    config_path = os.path.join('data_configs', config_name)
    
    # Write configuration file
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"Created data configuration at: {config_path}")
    return config_path

def main():
    """Main function demonstrating the complete workflow."""
    print("=== Example: Preparing a New Dataset for the Framework ===\n")
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic data...")
    data_dir = generate_synthetic_data(num_stations=3, days=365, save_dir="example_data")
    
    # Step 2: Create data configuration
    print("\nStep 2: Creating data configuration...")
    config_path = create_data_config(data_dir, use_hetero=True)
    
    # Step 3: Show how to use the new dataset
    print("\n=== How to Use the New Dataset ===")
    
    # With a standard forecasting model (DLinear)
    print("\nTo train with a standard model:")
    print(f"python run.py --model DLinear --data_config {config_path} --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 24")
    
    # With a text-guided model that can use heterogeneous data
    print("\nTo train with a text-guided model (using heterogeneous data):")
    print(f"python run.py --model TGTSF --data_config {config_path} --model_config model_configs/general/TGTSF.yaml --input_len 96 --output_len 24")
    
    # With PyTorch Lightning
    print("\nTo train with PyTorch Lightning:")
    print(f"python run_lightning.py --model DLinear --data_config {config_path} --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 24")
    
    print("\n=== Dataset Structure ===")
    print(f"- Main time series data: {data_dir}/id_*.parquet")
    print(f"- Metadata: {data_dir}/id_info.json")
    print(f"- Heterogeneous data: {data_dir}/hetero/weather_forecast_*.json")
    print(f"- Static information: {data_dir}/hetero/static_info.json")
    
    print("\n=== Next Steps ===")
    print("1. For real datasets, process your data into the same format")
    print("2. Create the appropriate data configuration file in data_configs/")
    print("3. Adjust the model configuration if needed")
    print("4. Train models using the commands above")

if __name__ == "__main__":
    main() 