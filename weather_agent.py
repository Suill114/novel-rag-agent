import os
from dataclasses import dataclass
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from pathlib import Path
import dashscope

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).parent / ".env")

# 设置全局 API key
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。
双关的意思是你回答天气的时候必须说一些与地名有关的诙谐的语气词，来让回答更有趣。
你可以使用两个工具：
- get_weather_for_location:用于获取特定地点的天气
- get_user_location:用于获取用户的位置
如果用户询问天气，请确保知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，
请使用 get_user_location 工具来查找他们的位置。"""


@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


@tool
def get_user_location(runtime) -> str:
    """根据用户ID获取用户信息。"""
    user_id = runtime.context.user_id
    return "上海" if user_id == "1" else "南京"


# 创建通义千问模型实例
llm = ChatTongyi(
    model="qwen-turbo"
)  # type: ignore


@dataclass
class ResponseFormat:
    """带双关语的回应（始终必须）"""
    punny_response: str
    weather_conditions: str | None = None


checkpoint_saver = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_user_location, get_weather_for_location],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpoint_saver,
)

# 运行代理
response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(user_id="2")
)

# 打印结果
print("=" * 50)
print("回答:", response["messages"][-1].content)
print("=" * 50)

