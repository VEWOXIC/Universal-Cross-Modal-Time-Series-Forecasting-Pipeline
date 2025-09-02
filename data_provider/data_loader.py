import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import warnings
from .data_helper import timestamp_spliter, ratio_spliter, data_buffer
import multiprocessing as mp
from time import time
from tqdm import tqdm
import json
from datetime import datetime
from functools import partial
import glob
import joblib, torch

warnings.filterwarnings('ignore')

class Universal_Dataset(Dataset):
    """
    PyTorch Dataset for universal time series forecasting with cross-modal data support.
    
    This dataset class handles both traditional time series data and heterogeneous cross-modal
    information (text, events, etc.) for enhanced forecasting. It supports multiple task types
    including standard time series forecasting (TSF), text-guided forecasting (TGTSF), and 
    multi-modal forecasting (MTSF).
    
    Args:
        root_path (str): Root directory path containing data files
        flag (str): Dataset split identifier ('train', 'val', 'test')
        data_path (str): Relative path to the specific data file (e.g., 'ETTh1.csv')
        seq_len (int): Length of input sequences for modeling
        pred_len (int): Length of prediction/forecast sequences  
        spliter (callable): Function to split data into train/val/test sets
        timestamp_col (str): Name of the timestamp column in the data
        target (str): Target column name(s) for prediction. Use 'all' for all columns
        scale (bool): Whether to apply z-score normalization to the data
        data_buffer (data_buffer, optional): Buffer object for caching loaded data
        hetero_data_getter (callable, optional): Function to retrieve heterogeneous data
        preload_hetero (bool): Whether to preload all heterogeneous data into memory
        hetero_stride (int): Stride for heterogeneous data alignment
        task (str, optional): Task type ('TSF', 'TGTSF', 'MTSF', 'Reasoning')
        custom_input (str, optional): Custom input specification overriding task defaults
        timezone (str, optional): Timezone for timestamp conversion
        downsample (int, optional): Downsampling factor for data reduction
    
    Attributes:
        data: Processed time series data as numpy array
        timestamp: Processed timestamps as numpy array
        scaler: StandardScaler for data normalization
        custom_input: List of input components to include in batches
    
    Example:
        ```python
        dataset = Universal_Dataset(
            root_path='./data',
            data_path='stock_data.csv',
            flag='train',
            seq_len=96,
            pred_len=24,
            target='close_price',
            scale=True,
            task='TSF'
        )
        ```
    """
    def __init__(self, root_path, flag='train', data_path='ETTh1.csv',
                 seq_len=24, pred_len=24, spliter=ratio_spliter, timestamp_col='date',
                 target='OT', scale=True, data_buffer=None, hetero_data_getter=None, preload_hetero=False, hetero_stride=1, task=None, custom_input=None, timezone=None, downsample=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        # init
        self.spliter = spliter
        self.set_type = flag

        self.target = target
        self.scale = scale
        self.data_buffer = data_buffer

        self.timestamp_col = timestamp_col

        self.root_path = root_path
        self.data_path = data_path

        self.hetero_data_getter = (lambda x: x) if hetero_data_getter is None else hetero_data_getter # return the timestamp
        self.timezone = timezone
        self.downsample = downsample

        self.__read_data__()
        self.preload_hetero = preload_hetero
        self.hetero_stride = hetero_stride

        if self.preload_hetero:
            self.__preload_hetero__()

        self.task = task
        self.custom_input = custom_input


        self.__input_format_parser__()

    def __input_format_parser__(self):
        """
        Parses and configures the input format based on task type or custom specification.
        
        Different tasks require different input components:
        - TSF: Basic time series forecasting (seq_x, seq_y, x_time, y_time)
        - TGTSF: Text-guided forecasting (adds future heterogeneous data)
        - MTSF: Multi-modal forecasting (adds historical heterogeneous data)
        - Reasoning/all: Full multi-modal input with all components
        
        Sets self.custom_input to list of required input component names.
        """
        if self.custom_input is not None:
            print('[ warning ] Custom input set, overriding task defined input as: {}'.format(self.custom_input))
            self.custom_input = self.custom_input.strip().split(',')
            # assert all the input is in the list of ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            assert all([i in ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel'] for i in self.custom_input]), "Custom input should be a comma split string of ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']"
        else:
            if self.task is None:
                print('[ warning ] No task defined, using default all input')
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            elif self.task == 'TSF':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time']
            elif self.task == 'TGTSF':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            elif self.task == 'MTSF':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_general', 'hetero_channel']
            elif self.task == 'Reasoning' or 'all':
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_x_time', 'x_hetero', 'hetero_y_time', 'y_hetero', 'hetero_general', 'hetero_channel']
            else:
                raise NotImplementedError('Task not supported, please use custom input to override')
            

    # @profile
    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.data_buffer is None:
            if self.data_path.endswith('.csv'):
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            elif self.data_path.endswith('.parquet'):
                df_raw = pd.read_parquet(os.path.join(self.root_path, self.data_path))
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')
        elif isinstance(self.data_buffer, data_buffer):
            df_raw = self.data_buffer(os.path.join(self.root_path, self.data_path))

        # convert the timestamp to datetime
        df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col])

        # Check if timestamp_col has timezone
        if df_raw[self.timestamp_col][0].tz is not None:
            if self.timezone is not None:
                print('[ info ] The timestamp column has timezone, converting to {}'.format(self.timezone))
                df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col], utc=True).dt.tz_convert(self.timezone).dt.tz_localize(None)
            else:
                print('[ info ] The timestamp column has timezone, forcing UTC')
                df_raw[self.timestamp_col] = pd.to_datetime(df_raw[self.timestamp_col], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)

        # apply the spliter
        train_data, val_data, test_data = self.spliter(df=df_raw)

        if self.set_type == 'train':
            self.data = train_data
        elif self.set_type == 'val':
            self.data = val_data
        elif self.set_type == 'test':
            self.data = test_data

        print(f"[ info ] Length of {self.set_type}: {self.data.shape[0]}")

        # convert the self.timestamp_col to yyyymmddHHMMSS int
        self.data[self.timestamp_col] = self.data[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S')
        # convert to int
        self.data[self.timestamp_col] = self.data[self.timestamp_col].astype(np.int64)
        

        self.timestamp = self.data[self.timestamp_col].values.copy()
        if self.target == 'all':
            self.data = self.data.drop(columns=[self.timestamp_col])
            self.data = self.data.values.astype(np.float32).copy()
            train_data = train_data.drop(columns=[self.timestamp_col])
            train_data = train_data.values.astype(np.float32).copy()
        else:
            self.data = self.data[self.target].values.astype(np.float32).copy()
            train_data = train_data[self.target].values.astype(np.float32).copy()

        if self.scale:
            self.scaler.fit(train_data)
            print(f"[ info ] mean and std (on train) of {self.data_path}: mean {self.scaler.mean_}, std {self.scaler.var_}")
            self.data = self.scaler.transform(self.data).astype(np.float32).copy()

        if self.downsample is not None:

            self.data = self.data[::self.downsample]
            self.timestamp = self.timestamp[::self.downsample]


    def __preload_hetero__(self):
        """
        Preloads all heterogeneous data into memory for efficient batch processing.
        
        When enabled, this method loads the complete heterogeneous dataset at initialization
        time rather than loading data on-demand during training. This improves training
        speed at the cost of increased memory usage.
        """
        if self.preload_hetero:
            print('[ info ] Preloading the full heterogeneous data')
            _ = time()
            self.hetero_time, self.hetero_general, self.hetero_channel, self.full_hetero = self.hetero_data_getter(self.timestamp)
            print('[ info ] Preload the full heterogeneous data successfully, cost time: {:.2f}s'.format(time() - _))
            del self.hetero_data_getter
    def __getitem__(self, index):
        """
        Retrieves a single data sample with all associated modalities.
        
        Constructs a complete training/inference sample containing time series data,
        timestamps, and corresponding heterogeneous cross-modal information. Handles
        both preloaded and on-demand heterogeneous data loading based on configuration.
        
        Args:
            index (int): Sample index in the dataset
        
        Returns:
            tuple: Complete data sample containing:
                - seq_x: Input time series sequence (seq_len, features)
                - seq_y: Target time series sequence (pred_len, features)  
                - x_time, y_time: Corresponding timestamps
                - x_hetero, y_hetero: Heterogeneous data (text, events, etc.)
                - hetero_x_time, hetero_y_time: Heterogeneous data timestamps
                - hetero_general: General heterogeneous information
                - hetero_channel: Channel-specific heterogeneous information
        """
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        x_time = self.timestamp[s_begin:s_end]
        y_time = self.timestamp[r_begin:r_end]

        x_hetero = np.zeros((1), dtype=np.float32)
        y_hetero = np.zeros((1), dtype=np.float32)
        hetero_x_time = np.zeros((1), dtype=np.float32)
        hetero_y_time = np.zeros((1), dtype=np.float32)
        hetero_general = np.zeros((1), dtype=np.float32)
        hetero_channel = np.zeros((1), dtype=np.float32)

        if self.preload_hetero:
            hetero_general = self.hetero_general
            hetero_channel = self.hetero_channel

            # dynamically load hetero to reduce preprocess time
            if 'x_hetero' in self.custom_input:
                hetero_x_time = self.hetero_time[s_begin:s_end:self.hetero_stride]
                x_hetero = self.full_hetero[s_begin:s_end:self.hetero_stride]
            if 'y_hetero' in self.custom_input:
                hetero_y_time = self.hetero_time[r_begin:r_end:self.hetero_stride]
                y_hetero = self.full_hetero[r_begin:r_end:self.hetero_stride]

            
        else:
            if 'x_hetero' in self.custom_input:
                x_hetero = self.hetero_data_getter(x_time[::self.hetero_stride])
                hetero_x_time = x_hetero[0]
                hetero_general = x_hetero[1]
                hetero_channel = x_hetero[2]
                x_hetero = x_hetero[3]

            if 'y_hetero' in self.custom_input:
                y_hetero = self.hetero_data_getter(y_time[::self.hetero_stride])
                hetero_y_time = y_hetero[0]
                hetero_general = y_hetero[1]
                hetero_channel = y_hetero[2]
                y_hetero = y_hetero[3]
        # still return everything for compatibility, but unwanted set as 0 for efficiency
        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

    def __len__(self):
        """
        Returns the total number of valid samples in the dataset.
        
        Calculates the number of complete sequences that can be generated
        given the sequence length and prediction length constraints.
        
        Returns:
            int: Number of valid data samples
        """
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Reverses the normalization transformation applied to data.
        
        Converts normalized data back to original scale using the fitted scaler.
        Essential for interpreting model predictions in their original units.
        
        Args:
            data (np.ndarray): Normalized data to transform back
        
        Returns:
            np.ndarray: Data in original scale
        """
        return self.scaler.inverse_transform(data)



class Heterogeneous_Dataset(Dataset):
    """
    Specialized dataset for managing heterogeneous cross-modal data sources.
    
    This class handles diverse data types including text (news, events), images, 
    and other modalities that complement time series forecasting. It provides
    efficient data loading, temporal alignment, and embedding generation for
    cross-modal forecasting tasks.
    
    Key Features:
        - Support for multiple heterogeneous data formats (JSON, text, images)
        - Temporal alignment between time series and heterogeneous data
        - Positional embeddings for temporal relationships
        - Memory-efficient data loading and caching
        - Flexible matching strategies (nearest, interpolation, etc.)
    
    Args:
        root_path (str): Root directory for heterogeneous data files
        formatter (str): File naming pattern for data files
        id_info (dict): Mapping of dataset IDs to metadata
        static_path (str, optional): Path to static/constant heterogeneous data
        matching (str): Strategy for temporal alignment ('nearest', 'interpolate')
        output_format (str): Format for heterogeneous data output ('json', 'text')
        timezone (str, optional): Timezone for timestamp alignment
        noise (float): Noise level for data augmentation
        hetero_type (str): Type of heterogeneous data handling strategy
        id_list (list, optional): Specific IDs to process
        postemb (str, optional): Positional embedding configuration
        postemb_model (str, optional): Model for generating positional embeddings
        postemb_max_len (int, optional): Maximum sequence length for embeddings
        postemb_d (int, optional): Dimensionality of positional embeddings
        postemb_batch_size (int): Batch size for embedding generation
        postemb_handle_downtime (str, optional): Strategy for handling data gaps
        device (str): Computing device ('cpu' or cuda device id)
    
    Example:
        ```python
        hetero_dataset = Heterogeneous_Dataset(
            root_path='./data/news',
            formatter='news_{i}.json',
            id_info=dataset_ids,
            matching='nearest',
            output_format='json'
        )
        ```
    """
    def __init__(self, root_path, formatter, id_info, static_path=None, matching='nearest', output_format='json', timezone=None, noise = 0.0, hetero_type='all_for_one', id_list=None, postemb=None, postemb_model=None, postemb_max_len=None, postemb_d=None, postemb_batch_size=200, postemb_handle_downtime=None, device='cpu'):
        super().__init__()

        self.hetero_type = hetero_type
        self.root_path = root_path
        self.formatter = formatter
        self.id_list = id_list
        self.device = torch.device('cpu') if device == 'cpu' else torch.device(f'cuda:{device}')
        self.postemb = postemb
        self.postemb_model = postemb_model
        self.postemb_max_len = int(postemb_max_len) if postemb_max_len is not None else postemb_max_len
        self.postemb_d = int(postemb_d) if postemb_d is not None else postemb_d
        self.postemb_batch_size = int(postemb_batch_size) if postemb_batch_size is not None else postemb_batch_size
        self.postemb_handle_downtime = postemb_handle_downtime

        self.tokenizer = AutoTokenizer.from_pretrained(self.postemb_model) if postemb is not None else None
        self.model = AutoModel.from_pretrained(self.postemb_model).to(self.device) if postemb is not None else None

        self.id_info = id_info
        self.static_path = static_path
        assert matching in ['nearest', 'forward', 'backward', 'single'], "The matching method should be one of ['nearest', 'forward', 'backward', 'single']"
        self.matching = matching
        assert output_format in ['dict','json', 'csv', 'embedding'], "The output format should be one of ['dict','json', 'csv', 'embedding']"
        self.output_format = output_format
        self.timezone = timezone
        
        if self.output_format == 'embedding':
            assert self.formatter is not None, "The embedding formatter should be provided if the output format is embedding"
            self.load_embedding(id_list=self.id_list)
        else:
            self.load_data()
        self.noise = noise

    def __addnoise__(self, x):
        x = x * (1 - self.noise) + np.random.randn(*x.shape) * self.noise
        # normalize
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def convert_plain_text_to_embeddings(self, text):
        tokenizer = self.tokenizer
        model = self.model
        model.eval()

        encoded = tokenizer(text,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt')

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # [CLS]
            text_embedding = outputs.last_hidden_state[:, 0, :].to('cpu')

        return text_embedding[0]

    def convert_df_text_to_embeddings(self, df):
        tokenizer = self.tokenizer
        model = self.model
        model.eval()

        df_time = df[['time']]
        df_merged = df.drop('time', axis=1).astype(str).apply(''.join, axis=1)

        batch_size = self.postemb_batch_size
        ls_embeddings = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing post embedding batches", unit="batch"):
            batch_texts = df_merged.iloc[i:i+batch_size].tolist()

            encoded = tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt')

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                # [CLS]
                batch_embeddings = outputs.last_hidden_state[:, 0, :].to('cpu')
                ls_embeddings.extend(batch_embeddings)

        df_time['embeddings'] = ls_embeddings

        return df_time

    def load_data(self):
        self.dynamic_data = {}
        self.dynamic_embed = {}
        if self.hetero_type == 'all_for_one':
            if self.formatter.endswith('.json'):
                file_paths = glob.glob(os.path.join(self.root_path, self.formatter))
                for file_path in file_paths:
                    json_data = json.load(open(file_path))
                    self.dynamic_data.update(json_data)

                self.dynamic_data = pd.DataFrame.from_dict(self.dynamic_data, orient='index')
                self.dynamic_data.index = pd.to_datetime(self.dynamic_data.index)
                # sort the index
                self.dynamic_data.sort_index(inplace=True)
                self.dynamic_data['time'] = self.dynamic_data.index
                self.dynamic_data['time'] = self.dynamic_data['time'].dt.strftime('%Y%m%d%H%M%S')
            elif self.formatter.endswith('.csv'):
                file_paths = glob.glob(os.path.join(self.root_path, self.formatter))
                df_list = []
                for file_path in file_paths:
                    df_list.append(pd.read_csv(file_path))
                self.dynamic_data = pd.concat(df_list)
                self.dynamic_data['time'] = pd.to_datetime(self.dynamic_data['time'])
                self.dynamic_data.set_index('time', inplace=True)
                self.dynamic_data.sort_index(inplace=True)
                # self.dynamic_data['time'] = self.dynamic_data.index.strftime('%Y%m%d%H%M%S') # This line is now redundant
            
            print('[ info ] Successfully load the dynamic data from {}'.format(self.formatter))

            if self.postemb is not None:
                self.dynamic_embed = self.convert_df_text_to_embeddings(self.dynamic_data)
                print('[ info ] Successfully convert text to embeddings after loading the textual data')

        elif self.hetero_type == 'each_subset':
            print(f'[ info ] Found {len(self.id_list)} subset IDs. Starting to load data for each...')
            for id in self.id_list:
                file_path = os.path.join(self.root_path, str(id), self.formatter)
                
                if not os.path.exists(file_path):
                    print(f'[ Warning ] Data file not found for id: {id} at path: {file_path}. Skipping.')
                    continue

                if self.formatter.endswith('.json'):
                    json_data = json.load(open(file_path))
                    df = pd.DataFrame.from_dict(json_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    df['time'] = df.index.strftime('%Y%m%d%H%M%S')

                elif self.formatter.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)

                # Time zone processing for each subset
                if df.index.tz is not None:
                    if self.timezone is not None:
                        df.index = df.index.tz_convert(self.timezone).tz_localize(None)
                    else:
                        df.index = df.index.tz_convert('UTC').tz_localize(None)
                
                self.dynamic_data[id] = df
                print(f'[ info ] Successfully loaded data for id: {id}')
                
                if self.postemb is not None:
                    df = self.convert_df_text_to_embeddings(df)
                    self.dynamic_embed[id] = df
                    print(f'[ info ] Successfully convert text to embeddings after loading the textual data for id: {id}')
        
        else:
            raise NotImplementedError('Only all_for_one and each_subset hetero type are supported, implement more if needed')

        if self.static_path is None:
            print('[ Warning ] No static data is provided, use default static data!')
            self.static_data = {
                'downtime_prompt': 'The sensor is down for unknown reasons.',
                'general_info': 'The general information of the sensor',
                'channel_info': {k: 'The information of the channel {}'.format(k) for k in self.id_info.keys()}
            }
        else:
            self.static_data = json.load(open(os.path.join(self.root_path, self.static_path)))
            print('[ info ] Successfully load the static data from {}'.format(self.static_path))
    
    def load_embedding(self, id_list=None):
        if self.hetero_type == 'all_for_one':
            self.embeddings = {}
            # if self.formatter.endswith('.npz'):
            #     file_paths = glob.glob(os.path.join(self.root_path, self.formatter))
            #     for file_path in file_paths:
            #         npz_data = np.load(file_path)
            #         self.embeddings.update(npz_data)
            #     self.static_data = np.load(os.path.join(self.root_path, self.static_path))
            if self.formatter.endswith('.pkl'):
                file_paths = glob.glob(os.path.join(self.root_path, self.formatter))
                for file_path in file_paths:
                    pkl_data = joblib.load(file_path)
                    self.embeddings.update(pkl_data)
                print('[ info ] Successfully load the dynamic data embedding from {}'.format(self.formatter))

            else:
                raise NotImplementedError('Only .pkl data are supported, implement more if needed')
            
            # fake dynamic data just for timestamp matching
            self.dynamic_data = pd.DataFrame.from_dict({k: 0 for k in self.embeddings.keys()}, orient='index')
            self.dynamic_data['time'] = self.dynamic_data.index
            self.dynamic_data.index = pd.to_datetime(self.dynamic_data.index)
            # sort the index
            # check if the index have timezone
            if self.dynamic_data.index.tz is not None:
                if self.timezone is not None:
                    print('[ info ] The index has timezone, converting to {}'.format(self.timezone))
                    self.dynamic_data.index = self.dynamic_data.index.tz_convert(self.timezone).tz_localize(None)
                else:
                    print('[ Warning ] The index has timezone, forcing UTC')
                    self.dynamic_data.index = self.dynamic_data.index.tz_convert('UTC').tz_localize(None)
                # print('[ info ] The index has timezone, converting to naive datetime, if need to keep timezone, please implement alignment using UDT')
                # self.dynamic_data.index = self.dynamic_data.index.tz_convert('Europe/Berlin').tz_localize(None)
            self.dynamic_data.sort_index(inplace=True)

        elif self.hetero_type == 'each_subset':

            self.embeddings = {}
            self.dynamic_data = {}

            print(f'[ info ] Found {len(id_list)} subset IDs. Starting to load embeddings for each...')

            for id in id_list:
                # /root_path/{id}/{formatter}. e.g. /data/subset_A/embeddings.pkl

                file_path = os.path.join(self.root_path, str(id), self.formatter)
                
                if not os.path.exists(file_path):
                    print(f'[ Warning ] Embedding file not found for id: {id} at path: {file_path}. Skipping.')
                    continue

                if self.formatter.endswith('.pkl'):
                    # Load the embeddings for each subset
                    id_specific_embeddings = joblib.load(file_path)
                    self.embeddings[id] = id_specific_embeddings
                    
                    # DataFrame for time matching for each subset
                    df = pd.DataFrame.from_dict({k: 0 for k in id_specific_embeddings.keys()}, orient='index')
                    df['time'] = df.index
                    df.index = pd.to_datetime(df.index)
                    
                    # Time zone processing for each subset
                    if df.index.tz is not None:
                        if self.timezone is not None:
                            df.index = df.index.tz_convert(self.timezone).tz_localize(None)
                        else:
                            df.index = df.index.tz_convert('UTC').tz_localize(None)
                    
                    df.sort_index(inplace=True)
                    self.dynamic_data[id] = df
                    print(f'[ info ] Successfully loaded embeddings for id: {id}')

                else:
                    raise NotImplementedError('Only .pkl data are supported for this structure.')

        else:
            raise NotImplementedError('Only all_for_one and each_subset hetero type are supported, implement more if needed')
    
        # Global static data
        self.static_data = joblib.load(os.path.join(self.root_path, self.static_path))
        print('[ info ] Successfully load the static data from {}'.format(self.static_path))

    def init_hetero_data(self, id):
        down_time = self.id_info[id]['sensor_downtime']
        down_time = [down_time[k]['time'] for k in down_time.keys()]
        down_time = [[pd.to_datetime(t[0]), pd.to_datetime(t[1])] for t in down_time]

        # check if all the downtime have timezone
        if any([t[0].tz is not None for t in down_time]):
            if self.timezone is not None:
                print('[ info ] The downtime has timezone, converting to {}'.format(self.timezone))
                down_time = [[t[0].tz_convert(self.timezone).tz_localize(None), t[1].tz_convert(self.timezone).tz_localize(None)] for t in down_time]
            else:
                print('[ Warning ] The downtime has timezone, forcing UTC')
                down_time = [[t[0].tz_convert('UTC').tz_localize(None), t[1].tz_convert('UTC').tz_localize(None)] for t in down_time]
            # print('[ info ] The downtime has timezone, converting to naive datetime, if need to keep timezone, please implement alignment using UDT')
            # down_time = [[t[0].tz_localize(None), t[1].tz_localize(None)] for t in down_time]

        general_info = self.static_data['general_info']
        channel_info = self.static_data['channel_info'][id]

        # channel_info = channel_info.reshape(1, 256) if channel_info.shape == (256,) else channel_info
        
        downtime_prompt = self.static_data['downtime_prompt']
        # Convert downtime ranges to IntervalIndex using from_arrays
        start_times = [t[0] for t in down_time]
        end_times = [t[1] for t in down_time]
        downtime_ranges = pd.IntervalIndex.from_arrays(start_times, end_times)

        return partial(self.get_hetero_data, downtime_ranges, general_info, channel_info, downtime_prompt, id)
            

    def time_matcher(self, timestamps, id=None):
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(timestamps.astype(str))

        id_specific_df = self.dynamic_data[id] if self.hetero_type == 'each_subset' else self.dynamic_data

        # Match times using vectorized operations on the correct DataFrame
        matched_indices = id_specific_df.index.searchsorted(timestamps)
        if self.matching == 'nearest':
            prev_indices = np.maximum(matched_indices - 1, 0)
            next_indices = np.minimum(matched_indices, len(id_specific_df.index) - 1)
            prev_deltas = (timestamps - id_specific_df.index[prev_indices]).total_seconds()
            next_deltas = (id_specific_df.index[next_indices] - timestamps).total_seconds()
            matched_indices = np.where(prev_deltas <= next_deltas, prev_indices, next_indices)
        elif self.matching == 'forward':
            matched_indices = np.minimum(matched_indices, len(id_specific_df.index) - 1)
        elif self.matching in ['backward', 'single']:
            matched_indices = np.maximum(matched_indices - 1, 0)

        matched_times = id_specific_df.index[matched_indices]

        if self.matching == 'single':
            _, unique_indices = np.unique(matched_times, return_index=True)
            matched_times = matched_times[unique_indices]
            timestamps = timestamps[unique_indices]

        return matched_times

    def downtime_checker(self, timestamps, downtime_ranges):
        # try:
            

            # Check downtime using vectorized operations
        is_downtime = np.array([any(downtime_ranges.contains(ts)) for ts in timestamps])
        # except TypeError:
        #     print(timestamps, type(timestamps), downtime_ranges, type(downtime_ranges))

        return is_downtime

    # @profile
    def get_hetero_data(self, downtime_ranges, general_info, channel_info, downtime_prompt, id, timestamp):

        if self.hetero_type == 'all_for_one':
            # Match times
            matched_times = self.time_matcher(timestamp)

            # Check downtime
            if len(downtime_ranges) == 0:
                is_downtime = np.zeros(len(matched_times), dtype=bool)
            else:  
                is_downtime = self.downtime_checker(matched_times, downtime_ranges)

            if self.output_format == 'embedding':
                matched_dynamic = self.dynamic_data.loc[matched_times]['time'].values
                output_dynamic_ = np.array([self.embeddings[time] for time in matched_dynamic], dtype=np.float32)
                downtime_data_ = np.array([downtime_prompt if is_down else np.zeros((1, downtime_prompt.shape[-1])) 
                                        for is_down in is_downtime], dtype=np.float32)
                # output_dynamic = np.concatenate([output_dynamic_, downtime_data_], axis=1)
                output_dynamic = np.empty((len(matched_dynamic), output_dynamic_.shape[1] + downtime_data_.shape[1], downtime_prompt.shape[-1]), dtype=np.float32)
                output_dynamic[:, :output_dynamic_.shape[1],:] = output_dynamic_
                output_dynamic[:, output_dynamic_.shape[1]:,:] = downtime_data_

                if self.noise > 0:
                    output_dynamic = self.__addnoise__(output_dynamic)

            else:
                if self.postemb is not None:
                    matched_dynamic = self.dynamic_data.loc[matched_times].copy()
                    matched_embed = self.dynamic_embed.loc[matched_times].copy() # .to_dict(orient='records')
                    # handling downtime
                    if self.postemb_handle_downtime is not None:
                        downtime_indices = np.where(is_downtime)[0]
                        for i in downtime_indices:
                            row_data = matched_dynamic.iloc[i].drop('time')
                            concatenated_row = ' '.join(str(x) for x in row_data.values) + " " + downtime_prompt
                            embedding_add_downtime = self.convert_plain_text_to_embeddings(concatenated_row)
                            matched_embed.iloc[i, matched_embed.columns.get_loc('embeddings')] = embedding_add_downtime
                    matched_embed = matched_embed.to_dict(orient='records')
                    output_dynamic = torch.stack([matched_df['embeddings'] for matched_df in matched_embed], dim=0)
                    # handling different length
                    if output_dynamic.size(0) < self.postemb_max_len:
                        padding_output = torch.zeros(self.postemb_max_len, self.postemb_d)
                        padding_output[:output_dynamic.size(0)] = output_dynamic
                        output_dynamic = padding_output
                    elif output_dynamic.size(0) > self.postemb_max_len:
                        output_dynamic = output_dynamic[:self.postemb_max_len]
                else:
                    matched_dynamic = self.dynamic_data.loc[matched_times].copy()
                    matched_dynamic['note'] = np.where(is_downtime, downtime_prompt, '')
                    matched_dynamic = matched_dynamic.to_dict(orient='records')
                    # remove the time from the dicts
                    for record in matched_dynamic:
                        record.pop('time', None)
                    
                    if self.output_format == 'dict':
                        output_dynamic = matched_dynamic
                    elif self.output_format == 'json':
                        output_dynamic = [json.dumps(record) for record in matched_dynamic]
                    elif self.output_format == 'csv':
                        output_dynamic = matched_dynamic.to_csv(index=False)
                    else:
                        raise NotImplementedError('Output format is not implemented yet')

            matched_times = matched_times.strftime('%Y%m%d%H%M%S').tolist()
            return matched_times, general_info, channel_info, output_dynamic

        elif self.hetero_type == 'each_subset':
            # Match times using the correct id
            matched_times = self.time_matcher(timestamp, id)

            if len(downtime_ranges) == 0:
                is_downtime = np.zeros(len(matched_times), dtype=bool)
            else:  
                is_downtime = self.downtime_checker(matched_times, downtime_ranges)

            if self.output_format == 'embedding':
                # Get the matched dynamic data for the specific subset id
                matched_dynamic = self.dynamic_data[id].loc[matched_times]['time'].values
                id_specific_embeddings = self.embeddings[id]
                output_dynamic_ = np.array([id_specific_embeddings[time] for time in matched_dynamic], dtype=np.float32)

                downtime_data_ = np.array([downtime_prompt if is_down else np.zeros((1, downtime_prompt.shape[-1])) 
                                        for is_down in is_downtime], dtype=np.float32)
                output_dynamic = np.empty((len(matched_dynamic), output_dynamic_.shape[1] + downtime_data_.shape[1], downtime_prompt.shape[-1]), dtype=np.float32)
                output_dynamic[:, :output_dynamic_.shape[1],:] = output_dynamic_
                output_dynamic[:, output_dynamic_.shape[1]:,:] = downtime_data_
                if self.noise > 0:
                    output_dynamic = self.__addnoise__(output_dynamic)
            else:
                # handling postemb
                if self.postemb is not None:
                    matched_dynamic = self.dynamic_data[id].loc[matched_times].copy()
                    matched_embed = self.dynamic_embed[id].loc[matched_times].copy() # .to_dict(orient='records')
                    # handling downtime
                    if self.postemb_handle_downtime is not None:
                        downtime_indices = np.where(is_downtime)[0]
                        for i in downtime_indices:
                            row_data = matched_dynamic.iloc[i].drop('time')
                            concatenated_row = ' '.join(str(x) for x in row_data.values) + " " + downtime_prompt
                            embedding_add_downtime = self.convert_plain_text_to_embeddings(concatenated_row)
                            matched_embed.iloc[i, matched_embed.columns.get_loc('embeddings')] = embedding_add_downtime
                    matched_embed = matched_embed.to_dict(orient='records')
                    output_dynamic = torch.stack([matched_df['embeddings'] for matched_df in matched_embed], dim=0)
                    # handling different length
                    if output_dynamic.size(0) < self.postemb_max_len:
                        padding_output = torch.zeros(self.postemb_max_len, self.postemb_d)
                        padding_output[:output_dynamic.size(0)] = output_dynamic
                        output_dynamic = padding_output
                    elif output_dynamic.size(0) > self.postemb_max_len:
                        output_dynamic = output_dynamic[:self.postemb_max_len]
                else:
                    matched_dynamic = self.dynamic_data[id].loc[matched_times].copy()
                    matched_dynamic['note'] = np.where(is_downtime, downtime_prompt, '')
                    matched_dynamic = matched_dynamic.to_dict(orient='records')
                    # remove the time from the dicts
                    for record in matched_dynamic:
                        record.pop('time', None)
                    
                    if self.output_format == 'dict':
                        output_dynamic = matched_dynamic
                    elif self.output_format == 'json':
                        output_dynamic = [json.dumps(record) for record in matched_dynamic]
                    elif self.output_format == 'csv':
                        output_dynamic = matched_dynamic.to_csv(index=False)
                    else:
                        raise NotImplementedError('Output format is not implemented yet')

            matched_times = matched_times.strftime('%Y%m%d%H%M%S').tolist()
            return matched_times, general_info, channel_info, output_dynamic
        
        else:
            raise NotImplementedError('Only all_for_one and each_subset hetero type are supported, implement more if needed')
