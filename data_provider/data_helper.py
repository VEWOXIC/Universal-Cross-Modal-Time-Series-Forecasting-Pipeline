import pandas as pd
import os, torch
# import the default collate function
from torch.utils.data.dataloader import default_collate

class data_buffer():
    """
    In-memory caching system for data files to optimize I/O performance.
    
    This class provides a buffer mechanism that caches loaded data files in memory
    to avoid repeated file system operations during training. Particularly useful
    when working with multiple datasets or when data needs to be accessed repeatedly.
    
    Attributes:
        buffer (dict): Internal dictionary storing file_path -> DataFrame mappings
    
    Example:
        ```python
        buffer = data_buffer()
        df = buffer('path/to/data.csv')  # Loads and caches
        df2 = buffer('path/to/data.csv')  # Returns cached version
        buffer.clear()  # Clears all cached data
        ```
    """
    def __init__(self):
        self.buffer = {}

    def __call__(self, file_path, force_reload=False):
        """
        Loads data from file with optional caching.
        
        Args:
            file_path (str): Path to the data file (.csv or .parquet)
            force_reload (bool): If True, reloads file even if cached
        
        Returns:
            pd.DataFrame: Loaded data (copy to prevent accidental modification)
        
        Raises:
            NotImplementedError: If file format is not .csv or .parquet
        """
        if force_reload:
            print(f'Force reloading data from {file_path}')
            if file_path.endswith('.csv'):
                df_raw = pd.read_csv(file_path)
                self.buffer[file_path] = df_raw.copy()
            elif file_path.endswith('.parquet'):
                df_raw = pd.read_parquet(file_path)
                self.buffer[file_path] = df_raw.copy()
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')
            return df_raw
        
        if file_path in self.buffer.keys():
            return self.buffer[file_path].copy()
        else:
            if file_path.endswith('.csv'):
                df_raw = pd.read_csv(file_path)
                self.buffer[file_path] = df_raw.copy()
            elif file_path.endswith('.parquet'):
                df_raw = pd.read_parquet(file_path)
                self.buffer[file_path] = df_raw.copy()
            else:
                raise NotImplementedError('Only .csv and .parquet data are supported, implement more if needed')
            print(f'[ info ] Add data {file_path} to buffer')
            return df_raw
    def clear(self):
        """
        Clears all cached data from memory buffer.
        
        This method is useful for freeing memory when switching between different
        datasets or when the cached data is no longer needed.
        """
        self.buffer = {}
        print('[ info ] Buffer cleared')
        

def ratio_spliter(split=(7,1,2),seq_len=0, df=None):
    """
    Splits time series data into train/validation/test sets based on proportional ratios.
    
    This function divides the input data according to specified ratios while ensuring
    that sequence overlap is maintained between splits for proper time series modeling.
    
    Args:
        split (tuple/list/str): Split ratios in format (train, val, test) or "x:y:z"
            Default (7,1,2) means 70% train, 10% validation, 20% test
        seq_len (int): Sequence length for overlap between splits to maintain continuity
        df (pd.DataFrame): Input DataFrame to split
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
    
    Example:
        ```python
        train, val, test = ratio_spliter(split=(8,1,1), seq_len=24, df=data)
        # Results in 80% train, 10% val, 10% test with 24-point overlap
        ```
    
    Raises:
        ValueError: If split format is invalid or doesn't contain exactly 3 components
    """
    
    # check if split is string it should be in format "x:y:z"
    if isinstance(split, str):
        assert len(split.split(':')) == 3, "Split should be in format 'x:y:z'"
        split = [int(x) for x in split.split(':')]
    elif isinstance(split, tuple) or isinstance(split, list):
        assert len(split) == 3, "Split should be in format 'x:y:z', or list with length of 3"
    else:
        raise ValueError("Split should be in format 'x:y:z', or list with length of 3")
    
    # split mark
    train_split = split[0] / sum(split)
    val_split = split[1] / sum(split) + train_split    # [----train----] train_split [----val----] val_split [----test----]
    
    raw_data_len = len(df)

    train_split = int(raw_data_len * train_split)
    val_split = int(raw_data_len * val_split)
    
    train_data = df[0:train_split]
    val_data = df[train_split-seq_len:val_split]
    test_data = df[val_split-seq_len:]

    return train_data, val_data, test_data

def timestamp_spliter(split = ['2020-01-01', '2020-02-01'], seq_len=0, df=None, timestamp_col='timestamp'):
    """
    Splits time series data into train/validation/test sets based on timestamp boundaries.
    
    This function provides precise temporal splitting by using specific dates/times as
    boundaries between different data splits. Supports optional data filtering by
    providing start and end timestamps.
    
    Args:
        split (list): List of timestamp strings defining split boundaries:
            - 2 elements: [val_start, test_start] 
            - 4 elements: [data_start, val_start, test_start, data_end]
        seq_len (int): Not used in timestamp splitting (maintained for API compatibility)
        df (pd.DataFrame): Input DataFrame with timestamp column
        timestamp_col (str): Name of the timestamp column
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames
    
    Example:
        ```python
        # Simple split: train before 2020-01-01, val until 2020-02-01, test after
        train, val, test = timestamp_spliter(['2020-01-01', '2020-02-01'], df=data)
        
        # With data filtering: only use data from 2019-01-01 to 2021-01-01
        train, val, test = timestamp_spliter(
            ['2019-01-01', '2020-01-01', '2020-02-01', '2021-01-01'], 
            df=data
        )
        ```
    
    Raises:
        ValueError: If split timestamps are not provided as strings
    """
    
    if isinstance(split[0], str):
        split = [pd.to_datetime(x) for x in split]
    else:
        raise ValueError("Split should be a list of strings")
    if len(split) == 4:
        print(f'[ info ] Discarding the data before {split[0]}')
        df = df[df[timestamp_col] >= split[0]]
        print(f'[ info ] Discarding the data after {split[3]}')
        df = df[df[timestamp_col] <= split[3]]
        split = split[1:3]
    train_data = df[df[timestamp_col] < split[0]]
    val_data = df[(df[timestamp_col] >= split[0]) & (df[timestamp_col] < split[1])]
    test_data = df[df[timestamp_col] >= split[1]]
    
    return train_data, val_data, test_data

