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
import joblib

warnings.filterwarnings('ignore')

class Universal_Dataset(Dataset):
    def __init__(self, root_path, flag='train', data_path='ETTh1.csv',
                 seq_len=24, pred_len=24, spliter=ratio_spliter, timestamp_col='date',
                 target='OT', scale=True, data_buffer=None, hetero_data_getter=None):
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
        self.__read_data__()
        self.collect_all_data()
        
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
        

        self.timestamp = self.data[self.timestamp_col].values
        self.data = self.data[self.target].values

        if self.scale:
            self.scaler.fit(train_data[self.target].values)
            self.data = self.scaler.transform(self.data)


    def collect_all_data(self):
        self.x_data = []
        self.y_data = []
        self.x_time = []
        self.y_time = []

        def process_data(i):
            s_begin = i
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            return (self.data[s_begin:s_end], self.data[r_begin:r_end], 
            self.timestamp[s_begin:s_end], self.timestamp[r_begin:r_end])
        
        for i in range(len(self.data) - self.seq_len - self.pred_len + 1):
            x_data, y_data, x_time, y_time = process_data(i)
            self.x_data.append(x_data)
            self.y_data.append(y_data)
            self.x_time.append(x_time)
            self.y_time.append(y_time)

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        seq_y = self.y_data[index]
        x_time = self.x_time[index]
        y_time = self.y_time[index]

        x_hetero = self.hetero_data_getter(x_time)
        y_hetero = self.hetero_data_getter(y_time)

        hetero_general = x_hetero[1]
        hetero_channel = x_hetero[2]
        hetero_x_time = x_hetero[0]
        hetero_y_time = y_hetero[0]
        x_hetero = x_hetero[3]
        y_hetero = y_hetero[3]


        return seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

    def __len__(self):
        return len(self.x_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Heterogeneous_Dataset(Dataset):
    def __init__(self, root_path, formatter, id_info, static_path=None, matching='nearest', output_format='json'):
        super().__init__()
        self.root_path = root_path
        self.formatter = formatter

        self.id_info = id_info
        self.static_path = static_path
        assert matching in ['nearest', 'forward', 'backward', 'single'], "The matching method should be one of ['nearest', 'forward', 'backward', 'single']"
        self.matching = matching
        assert output_format in ['dict','json', 'csv', 'embedding'], "The output format should be one of ['dict','json', 'csv', 'embedding']"
        self.output_format = output_format
        
        if self.output_format == 'embedding':
            assert self.formatter is not None, "The embedding formatter should be provided if the output format is embedding"
            self.load_embedding()
        else:
            self.load_data()

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
        self.dynamic_data.sort_index(inplace=True)

        print('[ info ] Successfully load the dynamic data embedding from {}'.format(self.formatter))
        


    # def time_matcher(self, timestamp):

    #     pos = self.dynamic_data.index.searchsorted(timestamp)
    #     if self.matching == 'nearest':
    #         if pos == 0:
    #             return self.dynamic_data.index[0]
    #         elif pos == len(self.dynamic_data):
    #             return self.dynamic_data.index[-1]
    #         else:
    #             if timestamp - self.dynamic_data.index[pos - 1] < self.dynamic_data.index[pos] - timestamp:
    #                 return self.dynamic_data.index[pos - 1]
    #             else:
    #                 return self.dynamic_data.index[pos]
    #     elif self.matching == 'forward':
    #         if pos == len(self.dynamic_data):
    #             return self.dynamic_data.index[-1]
    #         else:
    #             return self.dynamic_data.index[pos]
    #     elif self.matching == 'backward' or self.matching == 'single':
    #         if pos == 0:
    #             return self.dynamic_data.index[0]
    #         else:
    #             return self.dynamic_data.index[pos - 1]

            
    # def downtime_checker(self, timestamp, down_time):
    #     for t in down_time:
    #         if t[0] <= timestamp <= t[1]:
    #             return True
    #     return False

    def init_hetero_data(self, id):
        down_time = self.id_info[id]['sensor_downtime']
        down_time = [down_time[k]['time'] for k in down_time.keys()]
        down_time = [[pd.to_datetime(t[0]), pd.to_datetime(t[1])] for t in down_time]

        general_info = self.static_data['general_info']
        channel_info = self.static_data['channel_info'][id]
        downtime_prompt = self.static_data['downtime_prompt']

        return partial(self.get_hetero_data, down_time, general_info, channel_info, downtime_prompt)


    # def get_hetero_data(self, down_time, general_info, channel_info, downtime_prompt, timestamp):
    #     output_dynamic = []
    #     matched_times = []

    #     for t in timestamp:
    #         t = pd.to_datetime(str(t))
    #         matched_time = self.time_matcher(t)

    #         if self.matching == 'single' and matched_time in matched_times:
    #             continue # skip the repeated data in single mode

    #         matched_times.append(matched_time)
    #         if self.output_format == 'embedding':
    #             matched_time = self.dynamic_data.loc[matched_time]['time']
    #             data = self.embeddings[matched_time]
    #             if self.downtime_checker(t, down_time):
    #                 data = np.concatenate((data, np.array(downtime_prompt)), axis=0)
    #             else:
    #                 data = np.concatenate((data, np.zeros((1, data.shape[1]))), axis=0)
    #         else:
    #             data = self.dynamic_data.loc[matched_time]

    #             # check if the data is in the downtime period
    #             if self.downtime_checker(t, down_time):
    #                 data['note'] = downtime_prompt
    #             else:
    #                 data['note'] = ''



    #         if self.output_format == 'dict':
    #             output_dynamic.append(data.to_dict())
    #         elif self.output_format == 'json':
    #             output_dynamic.append(data.to_json())
    #         elif self.output_format == 'csv':
    #             output_dynamic.append(data.to_csv())
    #         elif self.output_format == 'embedding':
    #             output_dynamic.append(data)
    #         else:
    #             raise NotImplementedError('Output format is not implemented yet')
    #     # convert the matched_times to str in yyyymmddHHMMSS
    #     if self.output_format == 'embedding':
    #         matched_times = [t.strftime('%Y%m%d%H%M%S') for t in matched_times]    
    #         return matched_times, general_info, channel_info, np.stack(output_dynamic)
    #     else:
    #         matched_times = [t.strftime('%Y%m%d%H%M%S') for t in matched_times]
    #         return matched_times, general_info, channel_info, output_dynamic
            

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

        return matched_times, timestamps

    def downtime_checker(self, timestamps, down_time):
        # Convert downtime ranges to IntervalIndex
        down_time = [tuple(t) for t in down_time]
        downtime_ranges = pd.IntervalIndex.from_tuples(down_time)

        # Check downtime using vectorized operations
        is_downtime = np.array([any(downtime_ranges.contains(ts)) for ts in timestamps])

        return is_downtime

    def get_hetero_data(self, down_time, general_info, channel_info, downtime_prompt, timestamp):
        start_time = time()

        # Match times
        matched_times, timestamps = self.time_matcher(timestamp)

        # Check downtime
        is_downtime = self.downtime_checker(matched_times, down_time)

        if self.output_format == 'embedding':
            matched_dynamic = self.dynamic_data.loc[matched_times]['time'].values
            output_dynamic = np.array([self.embeddings[time] for time in matched_dynamic])
            downtime_data = np.array([downtime_prompt if is_down else np.zeros((1, downtime_prompt.shape[-1])) 
                                       for is_down in is_downtime])
            output_dynamic = np.concatenate([output_dynamic, downtime_data], axis=1)
        else:
            matched_dynamic = self.dynamic_data.loc[matched_times].copy()
            matched_dynamic['note'] = np.where(is_downtime, downtime_prompt, '')

            if self.output_format == 'dict':
                output_dynamic = matched_dynamic.to_dict(orient='records')
            elif self.output_format == 'json':
                output_dynamic = matched_dynamic.to_json(orient='records')
            elif self.output_format == 'csv':
                output_dynamic = matched_dynamic.to_csv(index=False)
            else:
                raise NotImplementedError('Output format is not implemented yet')

        matched_times = matched_times.strftime('%Y%m%d%H%M%S').tolist()
        print(f"Time taken for matching: {time() - start_time} seconds")
        return matched_times, general_info, channel_info, output_dynamic

            





    

