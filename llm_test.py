import openai

import certifi
import os
os.environ["SSL_CERT_FILE"] = certifi.where()

#  base_url and api_key
openai.base_url = ""
openai.api_key = ""

# prompt
prompt = "hello, I am a time series forecasting model. Can you tell me about the latest advancements in time series forecasting?"

# ChatCompletion api
response = openai.chat.completions.create(
    model="",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# 
print("Answer:", response.choices[0].message.content)
