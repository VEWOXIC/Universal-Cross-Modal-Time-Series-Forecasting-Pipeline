from exp.exp_basic import Exp_Basic
from models import model_init

from utils.tools import EarlyStopping, adjust_learning_rate

import numpy as np
import torch
import torch.nn as nn

import os
import time
import warnings

import json

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from data_provider.data_factory import Data_Provider
from multiprocessing import Pool



warnings.filterwarnings('ignore')

class Experiment(Exp_Basic):
    
    def __init__(self, args):
        self.args= args
        self.model = self._build_model()
        self.data_provider = Data_Provider(args, buffer=(not args.disable_buffer))
        

    def _build_model(self):
        model = model_init(self.args.model, self.args.model_config, self.args, is_LLM=True)
        return model

    def _get_data(self, flag, return_type='loader'):
        """
        Get the data for training, validation, or testing.
        """
        if flag == 'train':
            data_loader = self.data_provider.get_train(return_type=return_type)
        elif flag == 'val':
            data_loader = self.data_provider.get_val(return_type=return_type)
        elif flag == 'test':
            data_loader = self.data_provider.get_test(return_type=return_type)

        return data_loader

    def _forward_step(self, iter):
        """
        Forward step for the model.
        """
        output, log = self.model(iter)


        return output, log
    
    def test(self, savepath, valiset='full'):
        """
        Validate the model on the validation dataset.
        """
        criterion = self._select_criterion()
        data_sets = self.data_provider.get_test(return_type='set')
        total_loss = []
        if valiset == 'full':
            pass
        else:
            data_sets = {i: data_sets[i] for i in valiset.split(',')}

        for info, data_set in data_sets.items():
            info_result = []
            gts=[]
            preds=[]
            # make dir if not exist
            if not os.path.exists(os.path.join(savepath, info)):
                os.makedirs(os.path.join(savepath, info))
            info_savepath= os.path.join(savepath, info)


            indexes = list(range(0, len(data_set), self.args.sample_step))
            my_process_iteration = partial(process_iteration, dataset=data_set, args=self.args, model=self.model, info_savepath=info_savepath)

            print('[DEBUG]', data_set)
            print('[DEBUG]', self.args)
            print('[DEBUG]', self.model)
            print('[DEBUG]', info_savepath)
            print('[DEBUG]', my_process_iteration)

            if self.args.no_parallel:
                
                for index in tqdm(indexes, desc=f"Testing {info}"):
                    processed = my_process_iteration(index)
                    gts.append(processed["gt"])
                    preds.append(processed["pred"])
                    info_result.append(processed["result"])
            else:
                # get the number of gpus
                num_gpus = torch.cuda.device_count()
                max_workers = num_gpus * self.args.model_config.queue_len
                print(f"Number of GPUs: {num_gpus}, Max Workers: {max_workers}")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(tqdm(executor.map(my_process_iteration, indexes), desc=f"Testing {info}", total=len(indexes)))

                for processed in results:
                    gts.append(processed["gt"])
                    preds.append(processed["pred"])
                    info_result.append(processed["result"])

        #     info_loss = criterion(torch.tensor(preds), torch.tensor(gts))
        #     total_loss.append(info_loss.item())  # Append the loss for averaging later
        #     print(f"Test loss for {info}: {np.average(info_loss)}")
        #     # save the info_result as json in ckpt
        #     with open(os.path.join(savepath, f'{info}_result.json'), 'w') as f:
        #         json.dump(info_result, f, indent=4)
        # total_loss = np.average(total_loss)

        return None
    
def process_iteration(index, dataset, args, model, info_savepath):
    """
    Process a single iteration for testing.
    """
    iter = dataset[index]
    result, log = model(iter)

    gt = iter[1]
    gt = gt[-args.output_len:, 0]#.numpy()  # Ensure it's a numpy array

    pred = result['pred']
    pred = [p[1] for p in pred]
    pred = np.asarray(pred)  # Convert to numpy array if not already
    pred = pred[-args.output_len:]  # Extract the correct dimensions
    result['log']=log[2:]
    date = result['pred'][0][0]
    with open(os.path.join(info_savepath, f'{date}_result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    return {
        "result": result,
        "gt": gt,
        "pred": pred
    }
