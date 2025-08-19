import re
from statistics import mode

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from utils.ChatTime_prompt import getPrompt
from utils.ChatTime_tools import Discretizer, Serializer


class Model(nn.Module):
    def __init__(self, configs):

        super(Model, self).__init__()

        self.model_name = configs.model_name
        self.task = configs.task
        self.device = f"cuda:{int(configs.gpu)}" if configs.gpu is not None else "cpu"
        
        self.hist_len = configs.hist_len
        self.pred_len = configs.pred_len

        self.max_pred_len = 1000
        self.num_samples = configs.num_samples
        self.top_k = configs.top_k
        self.top_p = configs.top_p
        self.temperature = configs.temperature

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        print(f"[ info ] {self.model_name} loaded successfully on device {next(self.model.model.parameters()).device}.")

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

    def move_to_device(self, batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
        """
        Move data to device
        """
        return batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel

    def forward(self, x, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        batch_size = x.shape[0]
        series = x
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(flag="prediction", context=context, input=serialized_series)

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                min_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                max_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
            )
            samples = pipe(serialized_series)

            pred_list = []
            for sample in samples:
                serialized_prediction = sample["generated_text"].split("### Response:\n")[1]
                dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                
                if dispersed_prediction.size == 0:
                    print("Warning: Empty prediction array!")
                    return None
                
                pred = self.discretizer.inverse_discretize(dispersed_prediction)

                if len(pred) < min(remaining, self.max_pred_len):
                    pred = np.concatenate([pred, np.full(min(remaining, self.max_pred_len) - len(pred), np.NaN)])

                pred_list.append(pred[:min(remaining, self.max_pred_len)])

            prediction = np.nanmedian(pred_list, axis=0)
            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)  # [pred_len,]
        prediction = prediction.reshape(batch_size, -1)  # [batch_size, pred_len]

        # transform to tensor
        return torch.tensor(prediction, dtype=torch.float32)
    
    """
    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(flag="analysis", instruction=question, input=serialized_series)

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_pred_len,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
        )
        samples = pipe(serialized_series)

        response_list = []
        for sample in samples:
            response = sample["generated_text"].split("### Response:\n")[1].split('.')[0] + "."
            response = re.findall(r"\([abc]\)", response)[0]
            response_list.append(response)

        response = mode(response_list)

        return response
    """