import torch
import json, yaml
from utils.tools import dotdict
import pandas as pd
import numpy as np
import openai

class LLM_Socket():
    def __init__(self, configs):
        self.url = configs.url
        self.api_key = configs.api_key
        openai.api_key = self.api_key
        openai.base_url = self.url
        self.model = configs.model
        self.temperature = configs.temperature
        self.system_prompt = configs.system_prompt
        self.prompts = configs.prompts
        self.retry_prompt = configs.retry_prompt

        self.check_url_avaliability()

        self.messages = []

    def check_url_avaliability(self):
        avaliable_models = [model.id for model in openai.models.list().data]
        if self.model in avaliable_models:
            return True
        else:
            raise Exception(f"Model {self.model} is not available. Available models are {avaliable_models}, may need to check the endpoint setting...")

    def flush_messages(self):
        self.messages = []

    def load_templates(self):
        pass

    def __call__(self, *args, **kwds):
        pass
        
    