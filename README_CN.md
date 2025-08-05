好的，这是您提供的 README.md 文件的中文翻译。翻译保留了原始的 Markdown 格式、代码块和命令，以便于阅读和使用。

---

# 通用跨模态时间序列预测框架

<div align="center">

[<img src="https://devin.ai/assets/deepwiki-badge.png" alt="在 DeepWiki.com 上提问" height="20"/>](https://deepwiki.com/VEWOXIC/Universal-Cross-Modal-Time-Series-Forecasting-Pipeline)
[<img alt="Static Badge" src="https://img.shields.io/badge/Check_Tutorial-aaa?logo=https%3A%2F%2Fcode2tutorial.com%2F_next%2Fimage%3Furl%3D%252Ficon.png%26w%3D32%26q%3D75&label=Code2Tutorial&link=https%3A%2F%2Fcode2tutorial.com%2Ftutorial%2Fb08f8f15-cd02-475e-a4ee-17e0b775ae81%2Findex.md">](https://code2tutorial.com/tutorial/b08f8f15-cd02-475e-a4ee-17e0b775ae81/index.md)

</div>

一个全面、灵活且常用的类 DLinear 时间序列预测框架，同时支持 PyTorch 和 PyTorch Lightning。便于时间序列预测模型的开发与比较。

该框架支持多种传统时间序列模型、基于文本嵌入的跨模态预测以及基于语言模型的推理方法。

数据集: https://huggingface.co/collections/VEWOXIC/wiats-weather-intervention-aware-time-series-benchmark-6805e446a4dd84280a40a699

## 目录

- [通用跨模态时间序列预测框架](#通用跨模态时间序列预测框架)
  - [目录](#目录)
  - [架构概览](#架构概览)
  - [安装](#安装)
  - [快速入门](#快速入门)
    - [从预定义任务开始：](#从预定义任务开始)
    - [使用 PyTorch 开发：](#使用-pytorch-开发)
    - [使用 PyTorch Lightning 训练模型：](#使用-pytorch-lightning-训练模型)
  - [特性](#特性)
  - [支持的数据集](#支持的数据集)
  - [管线组件](#管线组件)
    - [数据流](#数据流)
    - [训练流程](#训练流程)
    - [PyTorch vs Lightning](#pytorch-vs-lightning)
  - [配置](#配置)
    - [模型配置](#模型配置)
    - [数据配置](#数据配置)
    - [命令行参数](#命令行参数)
      - [通用参数](#通用参数)
      - [训练参数](#训练参数)
      - [GPU 参数](#gpu-参数)
      - [数据加载参数](#数据加载参数)
      - [Lightning 特定参数](#lightning-特定参数)
    - [Ahead 任务定义](#ahead-任务定义)
  - [异构数据支持](#异构数据支持)
    - [概览](#概览)
    - [数据准备](#数据准备)
    - [异构数据配置](#异构数据配置)
    - [时间对齐方法](#时间对齐方法)
    - [输入格式](#输入格式)
    - [内存效率](#内存效率)
    - [在模型中使用](#在模型中使用)
  - [扩展框架](#扩展框架)
    - [添加新模型](#添加新模型)
    - [添加新数据集](#添加新数据集)
  - [高级用法](#高级用法)
    - [异构数据处理](#异构数据处理)
    - [多 GPU 训练](#多-gpu-训练)
    - [检查点管理](#检查点管理)
    - [警告⚠](#警告)

## 架构概览

该框架由以下几个关键组件构成：

```
.
├── data_provider/         # 数据加载与准备
├── models/                # 模型定义
├── exp/                   # 实验处理
├── utils/                 # 工具函数
├── layers/                # 模型构建模块
├── data_configs/          # 数据集配置
├── model_configs/         # 模型配置
├── run.py                 # 传统 PyTorch 训练入口点
└── run_lightning.py       # PyTorch Lightning 训练入口点
```

## 安装

1.  克隆仓库：
    ```bash
    git clone <repository-url>
    cd time-series-forecasting
    ```

2.  安装所需包：

    ```bash
    pip install -r requirements.txt
    ```

## 快速入门

### 从预定义任务开始：

```bash
bash scripts/solar/DLinear/DLinear_day.sh
```

### 使用 PyTorch 开发：

使用 `run.py` 脚本通过 PyTorch 训练模型。该脚本允许您指定模型、数据配置和其他参数。它与之前的 DLinear 实现完全相同，易于调整和调试。

```bash
python run.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 96
```

### 使用 PyTorch Lightning 训练模型：

开发完成后，您可能希望使用 PyTorch Lightning 提供的多 GPU 训练和其他功能。`run_lightning.py` 脚本是使用 Lightning 进行训练的入口点。只需在命令行中将 `run.py` 替换为 `run_lightning.py`，并添加 `--use_multi_gpu` 和 `--devices` 参数即可启用多 GPU 训练。

```bash
python run_lightning.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 96
```

```bash
python run_lightning.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --input_len 96 --output_len 96 --use_multi_gpu --devices 0,1,2,3
```

## 特性

-   **类 DLinear 管线**: 熟悉、易用且适应性强的管线，适用于各种时间序列预测任务。
-   **灵活的模型/数据集支持**: 使用 yaml 配置文件定义模型和数据集，更易于管理和扩展。
-   **为多模态时间序列任务做好准备**: 多模态时间序列分析是时间序列预测的下一个重要方向。该框架已准备好支持基于嵌入和基于文本的方法。
-   **双训练管线**: 同时支持 PyTorch 和 PyTorch Lightning 进行训练。PyTorch 便于调试和开发，而 Lightning 则支持高效的多 GPU 训练和实验跟踪。
-   **统一且简单的任务定义**: 使用 `--ahead` 参数进行实时对齐的任务定义方法。ahead 任务会自动与数据集的采样率对齐。例如，如果 ahead 为 1 天，对于小时级数据，预测范围是 24；对于分钟级采样数据，则是 24*60=1440。
-   **可定制的数据管线**: 轻松定制 ahead 任务、数据集划分器等。只需修改相应的 yaml 文件或 `data_provider/data_helper.py` 中的代码即可。

## 支持的数据集

除了用于 TSF 的原始时间序列数据集（如 ETT）外，我们还支持以下多模态数据集：

-   [WIATS: Weather Intervention-Aware Time Series Benchmark](https://huggingface.co/collections/VEWOXIC/wiats-weather-intervention-aware-time-series-benchmark-6805e446a4dd84280a40a699)

## 管线组件

### 数据流

1.  **数据配置**: 在 `data_configs/` 目录下的 YAML 文件中指定。
2.  **数据提供者**: `data_provider/data_factory.py` 创建数据集和数据加载器。
3.  **数据集类**: `data_provider/data_loader.py` 包含数据集类。
4.  **DataModule**: 对于 Lightning，`data_provider/lightning_data_module.py` 负责管理数据。

数据流遵循以下路径：
1.  从 YAML 文件加载数据配置。
2.  `Data_Provider` 类根据配置初始化数据集。
3.  数据集从配置中指定的文件加载数据。
4.  DataLoaders 为模型训练准备批次数据。

### 训练流程

1.  **实验类**: `exp/exp_universal.py` (PyTorch) 或 `exp/exp_lightning.py` (Lightning)。
2.  **模型初始化**: 模型从 `models/` 目录中初始化。
3.  **训练循环**: 由实验类或 Lightning Trainer 处理。
4.  **检查点管理**: 保存模型检查点和指标。

### PyTorch vs Lightning

该框架提供两种训练管线：

1.  **PyTorch 管线**:
    -   在 `exp/exp_universal.py` 中手动实现训练循环
    -   提供对训练细节的精细控制
    -   入口点: `run.py`

2.  **PyTorch Lightning 管线**:
    -   在 `exp/exp_lightning.py` 中使用 Lightning 的结构化方法
    -   简化的多 GPU 训练
    -   更好的实验跟踪
    -   更高效的代码组织
    -   入口点: `run_lightning.py`

## 配置

### 模型配置

模型配置在 `model_configs/` 目录的 YAML 文件中指定。这些参数会传递给模型的 `__init__` 方法，其中一些也用于初始化数据集。以下是 DLinear 的一个示例：

```yaml
model: DLinear
individual: False
enc_in: 1
task: TSF
```

通用模型配置参数：
-   `model`: 模型名称（必须与 `models/` 中的模型文件匹配）
-   `individual`: 是否为每个时间序列使用独立的参数
-   `enc_in`: 输入通道数
-   `task`: 任务类型（TSF = 时间序列预测，TGTSF = 文本引导的时间序列预测，Reasoning = LLM 推理任务）。这用于指示模型使用的任务类型，并帮助数据加载器确定批次中包含的数据类型。也可以使用逗号分隔的字符串（如 `['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']`）来覆盖为自定义输入。如果需要，可以在 `data_provider/data_loader.py` 的 `Universal_Dataset.__input_format_parser__` 中自定义任务类型。
-   `hetero_align_stride`: 如果设置为 True，数据加载器将对齐异构数据的步幅与时间序列的补丁（patching）以减少内存使用。
-   在模型的 `__init__` 方法中定义任何其他参数。

更复杂的模型（如 TGTSF）有额外的参数：

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
hetero_align_stride: True # 如果设置为 True，数据加载器将对齐异构数据的步幅与时间序列的补丁（patching）以减少内存使用
revin: True
task: TGTSF
time_zone: UTC # [可选] 如果您的数据有时区信息，请添加此项，并更改为您的数据所在的时区
```

### 数据配置

数据配置在 `data_configs/` 目录的 YAML 文件中指定：

基础配置（不含异构数据）：
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
formatter: 'id_{i}.parquet' # i 代表 id_info.json 中的索引
sampling_rate: 1h
base_T: 24
```

包含异构数据的配置：
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
  formatter: weather_forecast_????.json # 使用正则表达式匹配文件名
  matching: single
  input_format: json
  static_path: static_info.json
```

通用数据配置参数：
-   `root_path`: 数据目录的路径
-   `spliter`: 数据划分方法（`timestamp` 或 `ratio`），或在 `data_provider/data_helper.py` 中定义自己的划分器
-   `split_info`: 训练/验证/测试集的划分点
-   `timestamp_col`: 时间戳列的名称
-   `target`: 用于预测的目标列
-   `id_info`: 包含元数据的 JSON 文件
-   `id`: 用于训练的 ID（或 `all`）
-   `formatter`: 数据文件名的格式化字符串
-   `sampling_rate`: 时间序列的采样率
-   `base_T`: 时间序列的基础周期性

### 命令行参数

#### 通用参数

-   `--model`: 模型名称（例如，DLinear, TGTSF）
-   `--model_config`: 模型配置文件的路径
-   `--data_config`: 数据配置文件的路径
-   `--input_len`: 输入序列长度
-   `--output_len`: 输出序列长度（预测范围）
-   `--ahead`: 用于天/周/月级别预测的简写
-   `--batch_size`: 训练的批次大小

#### 训练参数

-   `--train_epochs`: 训练轮数
-   `--learning_rate`: 初始学习率
-   `--loss`: 损失函数（mse, l1）
-   `--lradj`: 学习率调整策略
-   `--patience`: 早停的耐心值

#### GPU 参数

-   `--use_gpu`: 是否使用 GPU
-   `--gpu`: GPU 设备 ID
-   `--use_multi_gpu`: 是否使用多个 GPU
-   `--devices`: 用于多 GPU 训练的设备 ID

#### 数据加载参数

-   `--scale`: 是否缩放数据
-   `--disable_buffer`: 禁用数据缓冲区以提高内存效率
-   `--preload_hetero`: 预加载异构数据
-   `--num_workers`: 数据加载器的工作进程数
-   `--prefetch_factor`: 数据加载器的预取因子

#### Lightning 特定参数

-   `--precision`: 训练精度（'32', '16', 或 'bf16'）
-   `--gradient_clip_val`: 梯度裁剪值

### Ahead 任务定义

该框架支持 ahead 任务定义，这是预测范围的简写。Ahead 任务会自动与数据集的采样率对齐。例如，如果 ahead 是 `day`，对于小时级数据，预测范围是 24；对于分钟级采样数据，则是 24*60=1440。

```bash
python run.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --ahead day
```

您可以在 `utils/task.py` 文件中添加自己的 ahead 任务定义。预定义的 ahead 任务如下：

| Ahead 任务 | 预测范围 | 回顾窗口 |
| ---------- | -------- | -------- |
| day        | 1天      | 7天      |
| week       | 7天      | 30天     |
| month      | 30天     | 60天     |

## 异构数据支持

### 概览

该框架为异构数据集成提供了强大的支持，允许您通过额外的上下文信息（例如，天气预报、文本数据或任何其他外部信息）来丰富时间序列预测。这对于能够利用多模态数据的模型（如文本引导的时间序列预测）尤其有价值。

异构数据集成的工作方式如下：
1.  通过专门的加载器（`Heterogeneous_Dataset`）加载异构数据
2.  创建将时间序列时间戳链接到相应异构数据的偏函数（partial functions）
3.  将这些函数传递给主数据集类（`Universal_Dataset`）
4.  在训练期间根据时间戳动态获取异构数据

### 数据准备

异构数据应按以下组件组织：

1.  **动态数据**: 时变的异构信息（例如，天气预报）
    -   支持的格式：JSON、CSV、预计算的嵌入（PKL）
    -   文件应根据模式命名（例如，`weather_forecast_20210101.json`）
    -   每个文件应包含时间戳作为键，数据作为值

2.  **静态数据**: 保持不变的信息（例如，元数据）
    -   存储在单个 JSON 文件中（例如，`static_info.json`）
    -   包含三个主要部分：
        -   `general_info`: 数据集的通用描述
        -   `downtime_prompt`: 关于传感器停机时段的信息
        -   `channel_info`: 关于特定通道/站点的信

3.  **停机信息**: 传感器不工作的时段
    -   存储在 `id_info.json` 文件中
    -   包含每个站点/通道未收集数据的时间范围

### 异构数据配置

要启用异构数据，请在您的数据配置文件中添加一个 `hetero_info` 部分：

```yaml
hetero_info:
  sampling_rate: 1day           # 异构数据的采样率
  root_path: /path/to/hetero    # 异构数据的路径 (None 表示使用主数据路径)
  formatter: weather_????.json   # 异构数据文件的文件名模式
  matching: single              # 时间对齐方法 (nearest, forward, backward, single)
  input_format: json            # 异构数据的格式 (json, dict, csv, embedding)
  static_path: static_info.json # 静态信息文件的路径
```

### 时间对齐方法

该框架支持多种方法来对齐时间序列时间戳与异构数据：

-   **nearest**: 在异构数据中查找最近的时间戳
-   **forward**: 使用异构数据中的下一个可用时间戳
-   **backward**: 使用异构数据中的上一个可用时间戳
-   **single**: 使用最后一个匹配的时间戳并去重（最节省内存），推荐用于 LLM 任务

此对齐由 `Heterogeneous_Dataset` 类中的 `time_matcher` 方法处理。

### 输入格式

异构数据可以以多种格式提供给模型：

-   **json**: 数据从 JSON 文件加载并作为 JSON 字符串返回
-   **dict**: 数据被加载并作为 Python 字典返回
-   **csv**: 数据被加载并作为 CSV 字符串返回
-   **embedding**: 从 pickle 文件加载预计算的嵌入（以提高效率）

嵌入格式对于大规模部署特别有用，因为在这些场景中，动态计算嵌入的成本会很高。

### 内存效率

对于大型异构数据集，该框架提供了几个选项来管理内存使用：

1.  **懒加载**: 默认情况下，异构数据在训练期间按需获取
2.  **预加载**: 设置 `--preload_hetero` 将所有异构数据加载到内存中以加快访问速度
3.  **步幅对齐**: 在模型配置中设置 `hetero_align_stride: True` 以使模型步幅与异构数据匹配
4.  **单一匹配**: 使用 `matching: single` 来对异构数据进行去重

该框架的实现通过以下方式确保高效检索：
-   向量化的时间戳匹配操作
-   基于区间的停机检查
-   使用偏函数避免冗余计算

### 在模型中使用

模型可以通过 `forward` 方法中的附加参数访问异构数据：

```python
def forward(self, x, historical_events=None, news=None, dataset_description=None, channel_description=None):
    # x: 时间序列数据 [批次, 输入长度, 通道]
    # historical_events: 历史异构数据
    # news: 未来异构数据 (预测期)
    # dataset_description: 通用数据集信息
    # channel_description: 特定于通道的信息

    # 使用时间序列和异构数据的模型逻辑
    ...
```

实验将自动检测模型接受哪些参数，并提供相应的数据。

## 扩展框架

### 添加新模型

1.  在 `models/` 目录中创建一个模型文件（例如，`models/NewModel.py`）：

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 提取配置参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        # 在这里定义你的模型架构
        self.layers = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            # 根据需要添加更多层
        )

    def forward(self, x, **kwargs):
        """
        x: 输入数据 [批次, 输入长度, 通道]
        返回: 输出预测 [批次, 输出长度, 通道]
        """
        # 实现前向传播
        
        # 示例实现：
        batch_size, seq_len, channels = x.shape
        x = x.permute(0, 2, 1)  # [批次, 通道, 输入长度]
        output = self.layers(x)  # [批次, 通道, 输出长度]
        output = output.permute(0, 2, 1)  # [批次, 输出长度, 通道]

        return output
    
    # 可选：为需要特定设备处理的模型实现 move_to_device 方法
    # (尤其是在处理异构数据时)
    def move_to_device(self, seq_x, seq_y, x_time, y_time,
                    x_hetero, y_hetero, hetero_x_time, hetero_y_time,
                    hetero_general, hetero_channel, device):
        # 将必要的张量移动到设备
        seq_x = seq_x.float().to(device)
        seq_y = seq_y.float().to(device)

        # 对于使用异构数据的模型：
        # hetero_channel = hetero_channel.float().to(device)
        # y_hetero = y_hetero.float().to(device)

        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
```

2.  在 `model_configs/` 目录中创建一个模型配置文件（例如，`model_configs/general/NewModel.yaml`）：

```yaml
model: NewModel
individual: False
enc_in: 1
# 添加你的模型需要的任何其他参数
task: TSF  # 如果使用异构数据，则为 TGTSF
```

3.  如果需要，更新 `models/__init__.py`（通常不需要，因为模型是动态加载的）

### 添加新数据集

1.  按照数据加载器期望的格式（通常是 CSV 或 Parquet）准备您的数据集文件。

2.  在 `data_configs/` 目录中创建一个数据配置文件（例如，`data_configs/new_dataset.yaml`）：

```yaml
root_path: /path/to/data
spliter: timestamp  # 或 ratio
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

3.  创建一个 `id_info.json` 文件来描述数据集：

```json
{
  "station1": {
    "description": "站点1的描述",
    "sensor_downtime": {...}
  },
  "station2": {
    "description": "站点2的描述",
    "sensor_downtime": {...}
  }
}
```

4.  对于异构数据，准备额外的数据文件并更新配置。

## 高级用法

### 异构数据处理

该框架通过数据配置文件中的 `hetero_info` 配置支持异构数据集成（例如，文本数据、辅助信息）。

在处理大型异构数据时，请考虑：
-   设置 `--disable_buffer` 以避免将所有数据加载到内存中
-   调整 `--prefetch_factor` 和 `--num_workers` 以实现高效的数据加载

### 多 GPU 训练

使用 Lightning 进行高效的多 GPU 训练：

```bash
python run_lightning.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --use_multi_gpu --devices 0,1,2,3
```

对于 PyTorch，也支持多 GPU，但优化程度较低：

```bash
python run.py --model DLinear --data_config data_configs/fullsolar.yaml --model_config model_configs/general/DLinear.yaml --use_multi_gpu --devices 0,1,2,3
```

### 检查点管理

检查点保存在 `./checkpoints/{setting_name}/` 中，包括：
-   `checkpoint.pth`: 基于验证损失的最佳模型
-   `args.json`: 用于训练的命令行参数
-   TensorBoard 日志（对于 Lightning）: `./checkpoints/tb_logs/{setting_name}/`

### 警告⚠
- 不要用 pytorch lightning 跑 FITS 这个模型，loss 会爆炸，初步猜测可能是复数计算的通信优化仍存在问题。