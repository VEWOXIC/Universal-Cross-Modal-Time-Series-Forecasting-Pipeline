import os
import time
import warnings
import json
import datetime
import traceback
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool

from exp.exp_basic import Exp_Basic
from models import model_init
from utils.tools import EarlyStopping, adjust_learning_rate
from data_provider.data_factory import Data_Provider

warnings.filterwarnings('ignore')

def log_error_to_file(log_path, info, index, error_type, message, stack_trace=""):
    """Appends a formatted error record to a specified log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""--- ERROR LOG [{timestamp}] ---
Dataset: {info}
Index: {index}
Error Type: {error_type}
Message: {message}
"""
    if stack_trace:
        log_entry += f"Stack Trace:\n{stack_trace}\n"
    log_entry += "-" * (27 + len(timestamp)) + "\n\n"
    
    with open(log_path, 'a') as f:
        f.write(log_entry)

class Experiment(Exp_Basic):
    
    def __init__(self, args):
        """Initializes the experiment, building the model and data provider."""
        self.args = args
        self.model = self._build_model()
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer))

    def _build_model(self):
        """Builds and returns the large language model based on provided arguments."""
        model = model_init(self.args.model, self.args.model_config, self.args, is_LLM=True)
        return model

    def _get_data(self, flag, return_type='loader'):
        """Retrieves data loaders or datasets for training, validation, or testing."""
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type=return_type)
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type=return_type)
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type=return_type)
        return data_loader

    def _forward_step(self, iter_data):
        """Performs a single forward pass of the model with the given data."""
        output, log = self.model(iter_data)
        return output, log

    def test(self, savepath, valiset='full'):
        """Runs the testing process on specified datasets, handles errors, and reports a summary."""
        data_sets = self.data_provider.get_test(return_type='set')
        error_log_path = os.path.join(savepath, 'error_log.txt')
        
        total_errors = 0
        total_shape_mismatch_errors = 0 
        total_processed = 0
        total_skipped = 0

        if self.args.filtered_samples is not None:
            filtered_samples = json.load(open(self.args.filtered_samples))

        if valiset != 'full':
            data_sets = {i: data_sets[i] for i in valiset.split(',')}

        for info, data_set in data_sets.items():
            error_count = 0
            shape_mismatch_count = 0 
            success_count = 0
            skipped_count = 0
            
            info_result = []
            gts = []
            preds = []
            
            info_savepath = os.path.join(savepath, info)
            if not os.path.exists(info_savepath):
                os.makedirs(info_savepath)

            if self.args.filtered_samples is not None:
                indexes = filtered_samples[info]
            else:
                indexes = list(range(0, len(data_set), self.args.sample_step))
            
            my_process_iteration = partial(
                process_iteration, 
                dataset=data_set, 
                args=self.args, 
                model=self.model, 
                info_savepath=info_savepath,
                error_log_path=error_log_path,
                info_name=info
            )
            
            def process_and_count(processed_item):
                nonlocal success_count, error_count, skipped_count, shape_mismatch_count
                status, data = processed_item

                if status == "success":
                    gts.append(data["gt"])
                    preds.append(data["pred"])
                    info_result.append(data["result"])
                    success_count += 1
                elif status == "skipped":
                    skipped_count += 1
                elif status == "error":
                    error_count += 1
                elif status == "error_shape_mismatch": 
                    shape_mismatch_count += 1

            if self.args.no_parallel:
                for index in tqdm(indexes, desc=f"Testing {info}"):
                    processed = my_process_iteration(index)
                    process_and_count(processed)
            else:
                num_gpus = torch.cuda.device_count()
                max_workers = 1 
                print(f"Number of GPUs: {num_gpus}, Max Workers: {max_workers}")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(tqdm(executor.map(my_process_iteration, indexes), desc=f"Testing {info}", total=len(indexes)))
                for processed in results:
                    process_and_count(processed)
                
            if os.path.exists(os.path.join(info_savepath, f'{info}_result.json')):
                old = json.load(open(os.path.join(info_savepath, f'{info}_result.json')))
                info_result = old + info_result
            with open(os.path.join(info_savepath, f'{info}_result.json'), 'w') as f:
                json.dump(info_result, f, indent=4)
            
            print(f"\n--- Summary for [{info}] ---")
            print(f"Successfully processed: {success_count}")
            print(f"Skipped (already exist): {skipped_count}")
            print(f"Prediction Shape Mismatch Errors: {shape_mismatch_count}")
            print(f"Other Errors: {error_count}")
            print("-" * (25 + len(info)))
            
            total_processed += success_count
            total_skipped += skipped_count
            total_errors += error_count
            total_shape_mismatch_errors += shape_mismatch_count

        print("\n--- Overall Test Summary ---")
        print(f"Total successfully processed: {total_processed}")
        print(f"Total skipped: {total_skipped}")
        print(f"Total Prediction Shape Mismatch Errors: {total_shape_mismatch_errors}")
        print(f"Total Other Errors: {total_errors}")
        print(f"Check '{error_log_path}' for detailed error reports.")
        print("--------------------------\n")

        return None

def process_iteration(index, dataset, args, model, info_savepath, error_log_path, info_name):
    """Processes a single data sample, handling model inference, error logging, and result saving."""
    try:
        data_instance = dataset[index]
        date_ = data_instance[3][0]
    
        if os.path.exists(os.path.join(info_savepath, f'{date_}_result.json')):
            return "skipped", None

        result_dict, log = model(data_instance)
        
        if result_dict is None:
            raise ValueError("Model call returned None, indicating a potential API or parsing failure.")

        result = result_dict
        gt = data_instance[1][-args.output_len:, :]

        if not isinstance(result, dict) or 'pred' not in result:
            raise ValueError(f"Model output must be a dictionary with a 'pred' key. Got: {result}")
        
        pred = [p[1] for p in result['pred']]
        pred = np.asarray(pred)
        
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        if len(pred) < args.output_len:
            error_message = f"Prediction time-step mismatch. Expected {args.output_len}, got {len(pred)}."
            print(f"\n[ERROR] Index {index} in {info_name}: {error_message}")
            log_error_to_file(error_log_path, info_name, index, "Prediction Shape Mismatch", error_message)
            return "error_shape_mismatch", error_message

        pred = pred[-args.output_len:]

        if pred.shape != gt.shape:
            error_message = f"Prediction shape mismatch. Expected {gt.shape}, but got {pred.shape}."
            print(f"\n[ERROR] Index {index} in {info_name}: {error_message}")
            log_error_to_file(error_log_path, info_name, index, "Prediction Shape Mismatch", error_message)
            return "error_shape_mismatch", error_message
        
        if 'log' not in result and log is not None:
            result['log'] = log[2:]
        
        date = result['pred'][0][0]
        
        with open(os.path.join(info_savepath, f'{date}_result.json'), 'w') as f:
            json.dump(result, f, indent=4)

        return "success", {"result": result, "gt": gt, "pred": pred}
        
    except Exception as e:
        stack_trace = traceback.format_exc()
        error_message = f"Failed to process index {index}. Reason: {e}"
        
        print("\n" + "="*50)
        print(f"Exception occurred at index {index} in {info_name}:")
        print(e)
        print(f"See '{error_log_path}' for full stack trace.")
        print("="*50 + "\n")

        log_error_to_file(error_log_path, info_name, index, "General Exception", str(e), stack_trace)
        return "error", str(e)
