# test_api.py
import os
import dashscope  # type: ignore
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量读取（安全）
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

from dashscope import Generation

response = Generation.call(
    model='qwen-turbo',
    prompt='写一首关于春日樱花的五言诗'
)

if response.status_code == 200:  # type: ignore
    print(response.output.text)  # type: ignore
else:
    print(f"请求失败，状态码：{response.status_code}")  # type: ignore
    print(f"错误信息：{response.message}")  # type: ignore