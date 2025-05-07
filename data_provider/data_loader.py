import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from .data_helper import timestamp_spliter, ratio_spliter, data_buffer
import multiprocessing as mp
from time import time
import json
from datetime import datetime
from functools import partial
import glob
import joblib, torch

warnings.filterwarnings('ignore')

class Universal_Dataset(Dataset):
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
        self.__read_data__()
        self.preload_hetero = preload_hetero
        self.hetero_stride = hetero_stride

        if self.preload_hetero:
            self.__preload_hetero__()

        self.task = task
        self.custom_input = custom_input
        self.downsample = downsample

        self.__input_format_parser__()

    def __input_format_parser__(self):
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
                self.custom_input = ['seq_x', 'seq_y', 'x_time', 'y_time', 'hetero_y_time', 'y_hetero', 'hetero_channel']
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

        # convert the self.timestamp_col to yyyymmddHHMMSS int
        self.data[self.timestamp_col] = self.data[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S')
        # convert to int
        self.data[self.timestamp_col] = self.data[self.timestamp_col].astype(int)
        

        self.timestamp = self.data[self.timestamp_col].values.copy()
        if self.target == 'all':
            self.data = self.data.drop(columns=[self.timestamp_col])
            self.data = self.data.values.astype(np.float32).copy()
        else:
            self.data = self.data[self.target].values.astype(np.float32).copy()

        if self.scale:
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data).astype(np.float32).copy()

        if self.downsample is not None:

            self.data = self.data[::self.downsample]
            self.timestamp = self.timestamp[::self.downsample]


    def __preload_hetero__(self):
        if self.preload_hetero:
            print('[ info ] Preloading the full heterogeneous data')
            _ = time()
            self.hetero_time, self.hetero_general, self.hetero_channel, self.full_hetero = self.hetero_data_getter(self.timestamp)
            print('[ info ] Preload the full heterogeneous data successfully, cost time: {:.2f}s'.format(time() - _))
            del self.hetero_data_getter
    # @profile
    def __getitem__(self, index):
        
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
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Heterogeneous_Dataset(Dataset):
    def __init__(self, root_path, formatter, id_info, static_path=None, matching='nearest', output_format='json', timezone=None, noise = 0.0):
        super().__init__()
        self.root_path = root_path
        self.formatter = formatter

        self.id_info = id_info
        self.static_path = static_path
        assert matching in ['nearest', 'forward', 'backward', 'single'], "The matching method should be one of ['nearest', 'forward', 'backward', 'single']"
        self.matching = matching
        assert output_format in ['dict','json', 'csv', 'embedding'], "The output format should be one of ['dict','json', 'csv', 'embedding']"
        self.output_format = output_format
        self.timezone = timezone
        
        if self.output_format == 'embedding':
            assert self.formatter is not None, "The embedding formatter should be provided if the output format is embedding"
            self.load_embedding()
        else:
            self.load_data()
        self.noise = noise

    def __addnoise__(self, x):
        x = x * (1 - self.noise) + np.random.randn(*x.shape) * self.noise
        # normalize
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x

    def load_data(self):
        self.dynamic_data = {}

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
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                self.dynamic_data[df['time']] = df
            self.dynamic_data = pd.concat(self.dynamic_data.values())
            self.dynamic_data['time'] = pd.to_datetime(self.dynamic_data['time'])
            self.dynamic_data.set_index('time', inplace=True)
            self.dynamic_data['time'] = self.dynamic_data['time'].dt.strftime('%Y%m%d%H%M%S')
        
        print('[ info ] Successfully load the dynamic data from {}'.format(self.formatter))
        # TODO: static_data = {downtime_prompt: '', general_info: '', channel_info: {114514: '', 1919810: ''}}

        if self.static_path is None:
            print('[ Warning ] No static data is provided, use default static data!!!!')
            self.static_data = {
                'downtime_prompt': 'The sensor is down for unknown reasons.',
                'general_info': 'The general information of the sensor',
                'channel_info': {k: 'The information of the channel {}'.format(k) for k in self.id_info.keys()}
            }

        else:
            self.static_data = json.load(open(os.path.join(self.root_path, self.static_path)))
    def load_embedding(self):
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
            self.static_data = joblib.load(os.path.join(self.root_path, self.static_path))
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

        print('[ info ] Successfully load the dynamic data embedding from {}'.format(self.formatter))

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
        downtime_prompt = self.static_data['downtime_prompt']
        # Convert downtime ranges to IntervalIndex using from_arrays
        start_times = [t[0] for t in down_time]
        end_times = [t[1] for t in down_time]
        downtime_ranges = pd.IntervalIndex.from_arrays(start_times, end_times)

        return partial(self.get_hetero_data, downtime_ranges, general_info, channel_info, downtime_prompt)
            

    def time_matcher(self, timestamps):
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(timestamps.astype(str))

        # Match times using vectorized operations
        matched_indices = self.dynamic_data.index.searchsorted(timestamps)
        if self.matching == 'nearest':
            prev_indices = np.maximum(matched_indices - 1, 0)
            next_indices = np.minimum(matched_indices, len(self.dynamic_data.index) - 1)
            prev_deltas = (timestamps - self.dynamic_data.index[prev_indices]).total_seconds()
            next_deltas = (self.dynamic_data.index[next_indices] - timestamps).total_seconds()
            matched_indices = np.where(prev_deltas <= next_deltas, prev_indices, next_indices)
        elif self.matching == 'forward':
            matched_indices = np.minimum(matched_indices, len(self.dynamic_data.index) - 1)
        elif self.matching in ['backward', 'single']:
            matched_indices = np.maximum(matched_indices - 1, 0)

        matched_times = self.dynamic_data.index[matched_indices]

        # Handle single mode to skip repeated data
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
    def get_hetero_data(self, downtime_ranges, general_info, channel_info, downtime_prompt, timestamp):

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

            
