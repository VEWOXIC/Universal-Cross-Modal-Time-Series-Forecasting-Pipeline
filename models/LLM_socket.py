import torch
import json, yaml, re
from utils.tools import dotdict
import pandas as pd
import numpy as np
import openai

class LLM_Socket():
    def __init__(self, configs):
        self.url = configs.base_url
        self.api_key = configs.api_key
        openai.api_key = self.api_key
        openai.base_url = self.url
        self.model = configs.model
        self.temperature = configs.temperature
        self.system_prompt = configs.system_prompt
        self.first_prompt = configs.first_prompt
        self.retry_prompt = configs.retry_prompt
        self.merge_system = configs.merge_system
        self.retry = configs.retry
        self.force_retry = configs.force_retry

        self.check_url_avaliability()

        self.load_templates()

    def to(self, device):
        return self

    def check_url_avaliability(self):
        avaliable_models = [model.id for model in openai.models.list().data]
        if self.model in avaliable_models:
            return True
        else:
            raise Exception(f"Model {self.model} is not available. Available models are {avaliable_models}, may need to check the endpoint setting...")

    def flush_messages(self):
        self.messages = []

    def load_templates(self):
        with open(self.first_prompt, 'r') as f:
            self.first_prompt = f.read()
        with open(self.retry_prompt, 'r') as f:
            self.retry_prompt = f.read()
        with open(self.system_prompt, 'r') as f:
            self.system_prompt = f.read()

        # self.messages.append({"role": "system", "content": self.system_prompt})

    def extract_result(self, result):
        pattern = r'```json(.*?)```'
        result = re.search(pattern, result, re.DOTALL)
        pred = json.loads(result.group(1))

        return pred
    
    def call_openai(self, messages):
        response = openai.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    temperature=self.temperature,
                                ).choices[0].message.content

        return response
    
    def process_instance(self, instance):
        # instance: ts_x, ts_y, tm_x, tm_y, (dy_tm_x, general, channel, [dy_x]), (dy_tm_y, general, channel, [dy_y])
        x_ts = instance[0].squeeze().tolist()
        x_timestamp = instance[2].tolist()
        x_table = [(x_timestamp[i], x_ts[i]) for i in range(len(x_ts))]

        x_dy = instance[4][-1]
        x_dy_timestamp = instance[4][0]
        x_dy_table = [(x_dy_timestamp[i], x_dy[i]) for i in range(len(x_dy))]

        channel_info = instance[4][2]
        dataset_info = instance[4][1]

        y_timestamp = instance[3].tolist()
        y_ts = instance[1].squeeze().tolist()
        y_table = [(y_timestamp[i], y_ts[i]) for i in range(len(y_ts))]

        y_dy = instance[5][-1]
        y_dy_timestamp = instance[5][0]
        y_dy_table = [(y_dy_timestamp[i], y_dy[i]) for i in range(len(y_dy))]

        return x_table, x_dy_table, channel_info, dataset_info, y_timestamp, y_dy_table, y_table

    def __call__(self, instance):
        retry = self.retry  # create a local copy to ensure self.retry remains unchanged
        # instance: ts_x, ts_y, tm_x, tm_y, (dy_tm_x, general, channel, [dy_x]), (dy_tm_y, general, channel, [dy_y])

        x_table, x_dy_table, channel_info, dataset_info, y_timestamp, y_dy_table, y_table = self.process_instance(instance)

        system_prompt=self.system_prompt.format(dataset_info=dataset_info)
        y_timestamp = [[y_timestamp[i], f'<your_prediction_value[{i}]>'] for i in range(len(y_timestamp))]
        first_prompt=self.first_prompt.format(channel_info=channel_info, x_table=x_table, x_dy_table=x_dy_table, y_dy_table=y_dy_table, y_timestamp=y_timestamp)
        retry_prompt=self.retry_prompt.format(y_timestamp=y_timestamp)



        if self.merge_system:
            first_prompt = system_prompt + first_prompt
            system_prompt = ''
        

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": first_prompt})

        if self.force_retry:
            for i in range(retry):
                result = self.call_openai(messages)
                pass
        else:
            while True:
                result = self.call_openai(messages)
                try:
                    messages.append({"role": "assistant", "content": result})
                    pred = self.extract_result(result)
                    if str(pred[0][0]) != str(y_timestamp[0][0]):
                        print(f"Mismatch: pred[0][0] = {pred[0][0]}, y_timestamp[0] = {y_timestamp[0]}")
                        raise AssertionError("Mismatch between pred[0][0] and y_timestamp[0]")
                    break
                except AssertionError:
                    if len(messages)>4: 
                        message = message[:4]
                    else:
                        messages.append({"role": "user", "content": retry_prompt})
                except:
                    if retry == 0:
                        pred = [(y_timestamp[i], None) for i in range(len(y_timestamp))]

                    else:
                        
                        messages.append({"role": "user", "content": retry_prompt})
                        retry -= 1
                        continue

        result = {'pred': pred, 
                  'x_table': x_table,
                'x_dy_table': x_dy_table,
                'y_timestamp': y_timestamp,
                'y_dy_table': y_dy_table,
                'y_table': y_table,}
        return result, messages

        
    