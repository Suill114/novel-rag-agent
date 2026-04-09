## 文档字符串
格式：“”“内容”“”<br>
用法：告诉智能体这个工具是用来干什么的

## 启用智能体
agent.invoke({"role": "user", "content": "南京的天气怎么样"})<br>
{ }内部保存对话对象和内容。<br>
invoke会返回智能体的回答，并且保存多轮对话

## @tool
可以把普通函数标记为AI可以使用的工具

## @dataclass
快速创建数据容器，把本来初始化变量过程简单化

## (city: str) -> str:
函数注解（类型提示），说经函数的参数和返回值分别应该是什么类型

## 创建agent
model=model   指定agent运用哪个模型<br>
system_prompt=SYSTEM_PROMPT   给AI设定角色、行为准则、任务目标<br>
tools=[ ]   告诉AI可以使用哪些函数<br>
context_shcema=Context  定义运行时上下文的数据结构。这些数据在 Agent 启动时传入，工具函数可以从中读取。<br>
response_format=ResponseFormat  强制AI按照指定格式返回结果，而不是随意输出文本。<br>
checkpointer=checkpointer   让 Agent 拥有短期记忆，能记住同一会话中的历史对话。<br>


