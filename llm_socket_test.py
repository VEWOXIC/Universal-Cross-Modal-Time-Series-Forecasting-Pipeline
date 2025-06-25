from openai import OpenAI
import os

client = OpenAI(
    api_key = os.getenv("MY_API_KEY"),
    base_url = os.getenv("MY_BASE_URL"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "hello, I am a time series forecasting model. Can you tell me about the latest advancements in time series forecasting?",
        }
    ],
    model="gpt-4.1-nano",
)

print(chat_completion.choices[0].message.content)