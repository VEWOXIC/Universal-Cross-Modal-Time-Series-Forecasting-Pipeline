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
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed


warnings.filterwarnings('ignore')

class Experiment(Exp_Basic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)

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
    
    def test(self, savepath, valinum='full'):
        """
        Validate the model on the validation dataset.
        """
        criterion = self._select_criterion()
        data_sets = self.data_provider.get_test(return_type='set')
        total_loss = []

        for info, data_set in data_sets.items():
            info_result = []
            gts=[]
            preds=[]


            indexes = list(range(0, len(data_set), 6))
            with ProcessPoolExecutor(max_workers=self.args.model_config.max_parallel) as executor:
                results = list(tqdm(executor.map(process_iteration, [data_set]*len(indexes), indexes,[self.args] * len(indexes),
                        [self.model] * len(indexes)), desc=f"Testing {info}", total=len(indexes)))

            for processed in results:
                gts.append(processed["gt"])
                preds.append(processed["pred"])
                processed["result"]['log'] = processed["log"]
                info_result.append(processed["result"])
            info_loss = criterion(torch.tensor(preds), torch.tensor(gts))
            total_loss.append(info_loss.item())  # Append the loss for averaging later
            print(f"Test loss for {info}: {np.average(info_loss)}")
            # save the info_result as json in ckpt
            with open(os.path.join(savepath, f'{info}_result.json'), 'w') as f:
                json.dump(info_result, f, indent=4)
        total_loss = np.average(total_loss)

        return total_loss
    
def process_iteration(dataset, index, args, model):
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
    print(f"pred shape: {pred.shape}")
    print(f"gt shape: {gt.shape}")
    print(pred)
    pred = pred[-args.output_len:]  # Extract the correct dimensions

    return {
        "result": result,
        "log": log,
        "gt": gt,
        "pred": pred
    }
