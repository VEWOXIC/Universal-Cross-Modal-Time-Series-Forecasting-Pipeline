# Universal Cross-Modal Time Series Forecasting Pipeline

A comprehensive, flexible and commonly-used DLinear-like framework for time series forecasting with support for both PyTorch and PyTorch Lightning. Easy for time series forecasting model development and comparison.

This framework supports various traditional time series models, text-embedding based cross-modal forecasting as well as language model-based reasoning approaches.

## Table of Contents

- [Universal Cross-Modal Time Series Forecasting Pipeline](#universal-cross-modal-time-series-forecasting-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Start with a predefined task:](#start-with-a-predefined-task)
    - [Develop with PyTorch:](#develop-with-pytorch)
    - [Training a model with PyTorch Lightning:](#training-a-model-with-pytorch-lightning)
  - [Features](#features)
  - [Support Datasets](#support-datasets)
  - [Pipeline Components](#pipeline-components)
    - [Data Flow](#data-flow)
    - [Training Process](#training-process)
    - [PyTorch vs Lightning](#pytorch-vs-lightning)
  - [Configuration](#configuration)
    - [Model Configuration](#model-configuration)
    - [Data Configuration](#data-configuration)
    - [Command-line Arguments](#command-line-arguments)
      - [Common Arguments](#common-arguments)
      - [Training Arguments](#training-arguments)
      - [GPU Arguments](#gpu-arguments)
      - [Data Loading Arguments](#data-loading-arguments)
      - [Lightning-Specific Arguments](#lightning-specific-arguments)
    - [Ahead task definition](#ahead-task-definition)
  - [Heterogeneous Data Support](#heterogeneous-data-support)
    - [Overview](#overview)
    - [Data Preparation](#data-preparation)
    - [Heterogeneous Data Configuration](#heterogeneous-data-configuration)
    - [Time Alignment Methods](#time-alignment-methods)
    - [Input Formats](#input-formats)
    - [Memory Efficiency](#memory-efficiency)
    - [Usage in Models](#usage-in-models)
  - [Extending the Framework](#extending-the-framework)
    - [Adding a New Model](#adding-a-new-model)
    - [Adding a New Dataset](#adding-a-new-dataset)
  - [Advanced Usage](#advanced-usage)
    - [Heterogeneous Data Handling](#heterogeneous-data-handling)
    - [Multi-GPU Training](#multi-gpu-training)
    - [Checkpoint Management](#checkpoint-management)

## Architecture Overview

The framework consists of several key components:

```
.
├── data_provider/         # Data loading and preparation
├── models/                # Model definitions
├── exp/                   # Experiment handling
├── utils/                 # Utility functions
├── layers/                # Model building blocks
├── data_configs/          # Dataset configurations 
├── model_configs/         # Model configurations
├── run.py                 # Traditional PyTorch training entry point
└── run_lightning.py       # PyTorch Lightning training entry point
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd time-series-forecasting
   ```

2. Install the required packages:
   

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start


### Start with a predefined task:

```bash
bash scripts/solar/DLinear/DLinear_day.sh
```

### Develop with PyTorch:

Use the `run.py` script to train a model with PyTorch. This script allows you to specify the model, data configuration, and other parameters. It is exactly the same pipeline as the previous DLinear implementation. Easy to adapt and debug. 

```bash
python run.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 96
```

### Training a model with PyTorch Lightning:

After the development, you may want to use multi-GPU training and other features provided by PyTorch Lightning. The `run_lightning.py` script is the entry point for training with Lightning. Just replace the `run.py` with `run_lightning.py` in the command line and add `--use_multi_gpu` and `--devices` arguments to enable multi-GPU training.

```bash
python run_lightning.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 96
```

```bash
python run_lightning.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 96 --use_multi_gpu --devices 0,1,2,3
```

## Features

- **DLinear-like Pipeline**: Familiar, easy-to-use and adaptable pipeline for various time series forecasting task.
- **Flexible Model/Dataset Support**: Use yaml configuration files to define models and datasets. Easier to manage and extend.
- **Ready for Multimodal Time Series Task**: Multi-modal time series analysis is the next big thing in time series forecasting. This framework is ready for with both embedding-based and text-based method. 
- **Dual Pipeline**: Supports both PyTorch and PyTorch Lightning for training. Pytorch for easy debugging and development, and Lightning for efficient multi-GPU training and experiment tracking.
- **Unified & Simple Task Definition**: Task defination method with `--ahead` argument for real-time aligned task definition. The ahead task is automatically aligns with the sampling rate of the dataset. E.g. if the ahead is 1 day, the prediction horizon is 24 for hourly data and 24*60=1440 for minutely sampled data.
- **Customizable Datapipeline**: Easy to customize the ahead task, dataset splitter and so on. Just modify the corresponding yaml file or the code in `data_provider/data_helper.py`.

## Support Datasets

Apart from the original time series dataset for TSF, e.g. ETT. We also support the following multimodal datasets:

- [WIATS: Weather Intervention-Aware Time Series Benchmark](https://huggingface.co/collections/VEWOXIC/wiats-weather-intervention-aware-time-series-benchmark-6805e446a4dd84280a40a699)


## Pipeline Components

### Data Flow

1. **Data Configuration**: Specified in YAML files in `data_configs/`.
2. **Data Provider**: `data_provider/data_factory.py` creates datasets and dataloaders.
3. **Dataset Class**: `data_provider/data_loader.py` contains the dataset classes.
4. **DataModule**: For Lightning, `data_provider/lightning_data_module.py` manages data.

The data flow follows this path:
1. Data configuration is loaded from YAML.
2. `Data_Provider` class initializes datasets based on the config.
3. Datasets load data from files specified in the configuration.
4. DataLoaders prepare batches for model training.

### Training Process

1. **Experiment Class**: `exp/exp_universal.py` (PyTorch) or `exp/exp_lightning.py` (Lightning).
2. **Model Initialization**: Models are initialized from the `models/` directory.
3. **Training Loop**: Handled by the experiment class or Lightning Trainer.
4. **Checkpoint Management**: Saves model checkpoints and metrics.

### PyTorch vs Lightning

The framework provides two training pipelines:

1. **PyTorch Pipeline**:
   - Manually implemented training loop in `exp/exp_universal.py`
   - Provides granular control over training details
   - Entry point: `run.py`

2. **PyTorch Lightning Pipeline**:
   - Uses Lightning's structured approach in `exp/exp_lightning.py`
   - Simplified multi-GPU training
   - Better experiment tracking
   - More efficient code organization
   - Entry point: `run_lightning.py`

## Configuration

### Model Configuration

Model configurations are specified in YAML files in the `model_configs/` directory. These are passed to the model's `__init__` method, some of them are also used to initialize the dataset. Here's an example for DLinear:

```yaml
model: DLinear
individual: False
enc_in: 1
task: TSF
```

Common model configuration parameters:
- `model`: Model name (must match a model file in `models/`)
- `individual`: Whether to use individual parameters for each time series
- `enc_in`: Number of input channels
- `task`: Task type (TSF = Time Series Forecasting, TGTSF = Text-Guided Time Series Forecasting), Reasoning = LLM Reasoning task. This is to indicate the type of task the model is used and help dataloader to determine the type of data included in a batch. Or use a comma split string of ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel'] to override as custom input. Customize tasktype in `data_provider/data_loader.py Universal_Dataset.__input_format_parser__` if needed.
- `hetero_align_stride`: if this is set to True, the dataloader will align the stride of the heterogeneous data with the time series patching for less memory usage
- Define any other parameters in the model's `__init__` method.

More complex models like TGTSF have additional parameters:

```yaml
model: TGTSF
individual: False
enc_in: 1
e_layers: 3
cross_layers: 3
self_layers: 3
mixer_self_layers: 3
n_heads: 4
d_model: 256
text_dim: 256
dropout: 0.3
patch_len: 16
stride: 8
hetero_align_stride: True # if this is set to True, the dataloader will align the stride of the heterogeneous data with the time series patching for less memory usage
revin: True
task: TGTSF # 
time_zone: UTC # [optional] add this if your data have timezone information, change to the timezone of your data
```

### Data Configuration

Data configurations are specified in YAML files in the `data_configs/` directory:

Basic configuration (without heterogeneous data):
```yaml
root_path: /path/to/data
spliter: timestamp
split_info:
  - '2021-01-01'
  - '2022-01-01'
timestamp_col: date
target: 
  - kWh
id_info: id_info.json
id: all
formatter: 'id_{i}.parquet' # i for the index in id_info.json
sampling_rate: 1h
base_T: 24
```

Configuration with heterogeneous data:
```yaml
root_path: /path/to/data
spliter: timestamp
split_info:
  - '2021-01-01'
  - '2022-01-01'
timestamp_col: date
target: 
  - kWh
id_info: id_info.json
id: all
formatter: 'id_{i}.parquet'
sampling_rate: 1h
base_T: 24
hetero_info:
  sampling_rate: 1day
  root_path: /path/to/hetero/data
  formatter: weather_forecast_????.json # use regex to match the file name
  matching: single
  input_format: json
  static_path: static_info.json
```

Common data configuration parameters:
- `root_path`: Path to the data directory
- `spliter`: Data splitting method (`timestamp` or `ratio`) or define your own spliter in `data_provider/data_helper.py`
- `split_info`: Split points for train/validation/test
- `timestamp_col`: Column name for timestamps
- `target`: Target column(s) for forecasting
- `id_info`: JSON file containing metadata
- `id`: IDs to use for training (or `all`)
- `formatter`: Format string for data file names
- `sampling_rate`: Time series sampling rate
- `base_T`: Base periodicity for time series

### Command-line Arguments

#### Common Arguments

- `--model`: Model name (e.g., DLinear, TGTSF)
- `--model_config`: Path to model configuration file
- `--data_config`: Path to data configuration file
- `--input_len`: Input sequence length
- `--output_len`: Output sequence length (prediction horizon)
- `--ahead`: Shorthand for day/week/month ahead forecasting
- `--batch_size`: Batch size for training

#### Training Arguments

- `--train_epochs`: Number of training epochs
- `--learning_rate`: Initial learning rate
- `--loss`: Loss function (mse, l1)
- `--lradj`: Learning rate adjustment strategy
- `--patience`: Early stopping patience

#### GPU Arguments

- `--use_gpu`: Whether to use GPU
- `--gpu`: GPU device ID
- `--use_multi_gpu`: Whether to use multiple GPUs
- `--devices`: GPU device IDs for multi-GPU training

#### Data Loading Arguments

- `--scale`: Whether to scale the data
- `--disable_buffer`: Disable data buffer for memory efficiency
- `--preload_hetero`: Preload heterogeneous data
- `--num_workers`: Number of dataloader workers
- `--prefetch_factor`: Prefetch factor for dataloader

#### Lightning-Specific Arguments

- `--precision`: Training precision ('32', '16', or 'bf16')
- `--gradient_clip_val`: Gradient clipping value

### Ahead task definition

The framework supports ahead task definition, which is a shorthand for the prediction horizon. The ahead task is automatically aligns with the sampling rate of the dataset. E.g. if the ahead is 1 day, the prediction horizon is 24 for hourly data and 24*60=1440 for minutely sampled data.

```bash
python run.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --ahead day
```

You can add your own ahead task definition in the `utils/task.py` file. The predefined ahead task is as below:

| Ahead Task | Prediction Horizon | Lookback Window |
| ---------- | ------------------ | --------------- |
| day        | 1 day              | 7 day           |
| week       | 7 day              | 30 day          |
| month      | 30 day             | 60 day          |

## Heterogeneous Data Support

### Overview

The framework provides robust support for heterogeneous data integration, allowing you to enrich time series forecasting with additional contextual information (e.g., weather forecasts, textual data, or any other external information). This is particularly valuable for models that can leverage multi-modal data, such as text-guided time series forecasting.

The heterogeneous data integration works by:
1. Loading heterogeneous data through a dedicated loader (`Heterogeneous_Dataset`)
2. Creating partial functions that link time series timestamps to corresponding heterogeneous data
3. Passing these functions to the main dataset class (`Universal_Dataset`)
4. Dynamically fetching heterogeneous data based on timestamps during training

### Data Preparation

Heterogeneous data should be organized with the following components:

1. **Dynamic data**: Time-varying heterogeneous information (e.g., weather forecasts)
   - Supported formats: JSON, CSV, pre-computed embeddings (PKL)
   - Files should be named according to a pattern (e.g., `weather_forecast_20210101.json`)
   - Each file should contain timestamps as keys and data as values

2. **Static data**: Information that remains constant (e.g., metadata)
   - Stored in a single JSON file (e.g., `static_info.json`)
   - Contains three main components:
     - `general_info`: General description of the dataset
     - `downtime_prompt`: Information about sensor downtime periods
     - `channel_info`: Information about specific channels/stations

3. **Downtime information**: Periods when sensors were not working
   - Stored in the `id_info.json` file
   - Contains time ranges for each station/channel when data was not collected



### Heterogeneous Data Configuration

To enable heterogeneous data, add a `hetero_info` section to your data config file:

```yaml
hetero_info:
  sampling_rate: 1day           # Sampling rate of heterogeneous data
  root_path: /path/to/hetero    # Path to heterogeneous data (None to use main data path)
  formatter: weather_????.json   # Filename pattern for heterogeneous data files
  matching: single              # Time alignment method (nearest, forward, backward, single)
  input_format: json            # Format of heterogeneous data (json, dict, csv, embedding)
  static_path: static_info.json # Path to static information file
```

### Time Alignment Methods

The framework supports several methods for aligning time series timestamps with heterogeneous data:

- **nearest**: Find the closest timestamp in heterogeneous data
- **forward**: Use the next available timestamp in heterogeneous data
- **backward**: Use the previous available timestamp in heterogeneous data
- **single**: Use the last matching timestamp and deduplicate (most memory-efficient) recommended for LLM tasks

This alignment is handled by the `time_matcher` method in the `Heterogeneous_Dataset` class.

### Input Formats

Heterogeneous data can be provided to the model in several formats:

- **json**: Data is loaded from JSON files and returned as a JSON string
- **dict**: Data is loaded and returned as Python dictionaries
- **csv**: Data is loaded and returned as CSV strings
- **embedding**: Pre-computed embeddings are loaded from pickle files (for efficiency)

The embedding format is particularly useful for large-scale deployments where computing embeddings on-the-fly would be expensive.

### Memory Efficiency

For large heterogeneous datasets, the framework provides several options to manage memory usage:

1. **Lazy loading**: By default, heterogeneous data is fetched on-demand during training
2. **Preloading**: Set `--preload_hetero` to load all heterogeneous data into memory for faster access
3. **Stride alignment**: Set `hetero_align_stride: True` in the model config to match model stride with heterogeneous data
4. **Single matching**: Use `matching: single` to deduplicate heterogeneous data

The framework's implementation ensures efficient retrieval via:
- Vectorized timestamp matching operations
- Interval-based downtime checking
- Partial functions to avoid redundant computations

### Usage in Models

Models can access heterogeneous data through additional parameters in the forward method:

```python
def forward(self, x, historical_events=None, news=None, dataset_description=None, channel_description=None):
    # x: Time series data [Batch, Input length, Channel]
    # historical_events: Historical heterogeneous data
    # news: Future heterogeneous data (prediction period)
    # dataset_description: General dataset information
    # channel_description: Channel-specific information
    
    # Model logic using both time series and heterogeneous data
    ...
```

The experiment will automatically detect which parameters the model accepts and provide the corresponding data.

## Extending the Framework

### Adding a New Model

1. **Create a model file** in the `models/` directory (e.g., `models/NewModel.py`):

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # Extract configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        # Define your model architecture here
        self.layers = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            # Add more layers as needed
        )
    
    def forward(self, x, **kwargs):
        """
        x: Input data [Batch, Input length, Channel]
        Returns: Output prediction [Batch, Output length, Channel]
        """
        # Implement the forward pass
        
        # Example implementation:
        batch_size, seq_len, channels = x.shape
        x = x.permute(0, 2, 1)  # [Batch, Channel, Input length]
        output = self.layers(x)  # [Batch, Channel, Output length]
        output = output.permute(0, 2, 1)  # [Batch, Output length, Channel]
        
        return output
    
    # Optional: implement move_to_device method for models that require 
    # specific device handling (especially with heterogeneous data)
    def move_to_device(self, seq_x, seq_y, x_time, y_time, 
                    x_hetero, y_hetero, hetero_x_time, hetero_y_time, 
                    hetero_general, hetero_channel, device):
        # Move necessary tensors to device
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)
        
        # For models that use heterogeneous data:
        # hetero_channel = hetero_channel.float().to(device)
        # y_hetero = y_hetero.float().to(device)
        
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
```

2. **Create a model configuration file** in `model_configs/` (e.g., `model_configs/general/NewModel.yaml`):

```yaml
model: NewModel
individual: False
enc_in: 1
# Add any additional parameters your model needs
task: TSF  # or TGTSF if it uses heterogeneous data
```

3. **Update models/__init__.py** if needed (usually not necessary as models are loaded dynamically)

### Adding a New Dataset

1. **Prepare your dataset files** in the format expected by the data loader (typically CSV or Parquet).

2. **Create a data configuration file** in `data_configs/` (e.g., `data_configs/new_dataset.yaml`):

```yaml
root_path: /path/to/data
spliter: timestamp  # or ratio
split_info:
  - '2022-01-01'
  - '2022-07-01'
timestamp_col: timestamp
target: 
  - value
id_info: id_info.json
id: all
formatter: 'id_{i}.parquet'
sampling_rate: 1h
base_T: 24
```

3. **Create an id_info.json file** that describes the dataset:

```json
{
  "station1": {
    "description": "Description of station 1",
    "sensor_downtime": {...}
  },
  "station2": {
    "description": "Description of station 2",
    "sensor_downtime": {...}
  }
}
```

4. **For heterogeneous data**, prepare additional data files and update the configuration.

## Advanced Usage

### Heterogeneous Data Handling

The framework supports heterogeneous data integration (e.g., text data, auxiliary information) through the `hetero_info` configuration in data config files.

When dealing with large heterogeneous data, consider:
- Set `--disable_buffer` to avoid loading all data into memory
- Adjust `--prefetch_factor` and `--num_workers` for efficient data loading

### Multi-GPU Training

Use Lightning for efficient multi-GPU training:

```bash
python run_lightning.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --use_multi_gpu --devices 0,1,2,3
```

For PyTorch, multi-GPU is also supported but less optimized:

```bash
python run.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --use_multi_gpu --devices 0,1,2,3
```

### Checkpoint Management

Checkpoints are saved in `./checkpoints/{setting_name}/`, including:
- `checkpoint.pth`: Best model based on validation loss
- `args.json`: Command-line arguments used for training
- TensorBoard logs (for Lightning): `./checkpoints/tb_logs/{setting_name}/`

