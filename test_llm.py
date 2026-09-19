"""验证 DeepSeek API Key 连通性。"""
import os

from dotenv import load_dotenv

load_dotenv(r"D:\trae\111\.env")

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)
resp = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[{"role": "user", "content": "回复OK即可"}],
    max_tokens=8,
)
print("API 连通 ✓ 模型回复:", resp.choices[0].message.content)
