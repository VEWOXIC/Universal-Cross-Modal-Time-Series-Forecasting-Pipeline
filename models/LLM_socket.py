import torch
import json, yaml, re
from utils.tools import dotdict
import pandas as pd
import numpy as np
import openai
from time import sleep

class LLM_Socket():
    def __init__(self, configs):
        self.url = configs.base_url
        self.api_key = configs.api_key
        # openai.api_key = self.api_key
        # openai.base_url = self.url
        self.model = configs.model
        self.temperature = configs.temperature
        self.system_prompt = configs.system_prompt
        self.first_prompt = configs.first_prompt
        self.retry_prompt = configs.retry_prompt
        self.merge_system = configs.merge_system
        self.retry = configs.retry
        self.force_retry = configs.force_retry

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.url)

        self.check_url_avaliability()

        self.load_templates()

    def to(self, device):
        return self

    def check_url_avaliability(self):
        avaliable_models = [model.id for model in self.client.models.list().data]
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
        try:
            result = result.group(1)
            if "'" in result:
                result = result.replace("'", '')
            if "//" in result:
                # remove the comment for each line
                result = re.sub(r'//.*?\n', '\n', result)
            pred = json.loads(result)
        except json.decoder.JSONDecodeError as e:
            print(result)
            raise e
        except Exception as e:
            print(f"[Error] Unexpected Error: {e}")
            raise e
        

        return pred
    
    def call_openai(self, messages):
        # try:
        response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                timeout=1200,
                                seed=int(np.random.choice([114,514,1919,810])),
                            ).choices[0].message.content
        # except openai.APITimeoutError:
        #     # sleep for a while and retry
        #     print("API Timeout Error: Retrying...")
        #     sleep(10)
            
        #     response = self.call_openai(messages)
        # except openai.BadRequestError as e:
        #     print(f"Bad Request Error: {e}")
        #     sleep(10)
        #     response = self.call_openai(messages)

        return response
    
    def process_instance(self, instance):
        # instance: seq_x, seq_y, x_time, y_time, x_hetero, y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel
        x_ts = instance[0].squeeze().tolist()
        x_timestamp = instance[2].tolist()
        x_table = [(x_timestamp[i], x_ts[i]) for i in range(len(x_ts))]

        x_dy = instance[4]
        x_dy_timestamp = instance[6]
        x_dy_table = [(x_dy_timestamp[i], x_dy[i]) for i in range(len(x_dy))]

        channel_info = instance[-1]
        dataset_info = instance[-2]

        y_timestamp = instance[3].tolist()
        y_ts = instance[1].squeeze().tolist()
        y_table = [(y_timestamp[i], y_ts[i]) for i in range(len(y_ts))]

        y_dy = instance[5]
        y_dy_timestamp = instance[7]
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
                
                try:
                    result = self.call_openai(messages)
                    messages.append({"role": "assistant", "content": result})
                    pred = self.extract_result(result)
                    if str(pred[0][0]) != str(y_timestamp[0][0]):
                        print(f"Mismatch: pred[0][0] = {pred[0][0]}, y_timestamp[0][0] = {y_timestamp[0][0]}")
                        raise AssertionError("Mismatch between pred[0][0] and y_timestamp[0]")
                    break
                except openai.APITimeoutError or openai.BadRequestError as e:
                    if retry == 0:
                        pred = [(y_timestamp[i], -1) for i in range(len(y_timestamp))]
                        break
                    messages = messages[:2]
                    print(f"[Error] Server Error: {e}")
                    sleep(10)
                    retry -= 1
                    continue
                except AssertionError as e:
                    if retry == 0:
                        pred = [(y_timestamp[i], -1) for i in range(len(y_timestamp))]
                        break
                    if len(messages)>4: 
                        messages = messages[:4]
                    else:
                        messages.append({"role": "user", "content": retry_prompt + 'Your time stamp is not properly aligned or json format is wrong, FIX IT!'})

                    print(f"[Error] Assertion Error: {e}")
                    # sleep(10)
                    retry -= 1
                    continue

                except json.JSONDecodeError as e:
                    if retry == 0:
                        pred = [(y_timestamp[i], -1) for i in range(len(y_timestamp))]
                        break
                    if len(messages)>4: 
                        messages = messages[:4]
                    else:
                        messages.append({"role": "user", "content": retry_prompt + 'Check the format of your json output!'})
                    print(f"[Error] JSON Decode Error: {e}")
                    # sleep(10)
                    retry -= 1
                    continue
                except Exception as e:
                    if retry == 0:
                        pred = [(y_timestamp[i], -1) for i in range(len(y_timestamp))]
                        break
                    if len(messages)>4: 
                        messages = messages[:4]
                    else:
                        messages.append({"role": "user", "content": retry_prompt + 'Check your output! make sure your output is a json format!'})
                    print(f"[Error] Unexpected Error: {e}")
                    # sleep(10)
                    retry -= 1
                    continue
                

        result = {'pred': pred, 
                  'x_table': x_table,
                'x_dy_table': x_dy_table,
                'y_timestamp': y_timestamp,
                'y_dy_table': y_dy_table,
                'y_table': y_table,}
        return result, messages

        
    