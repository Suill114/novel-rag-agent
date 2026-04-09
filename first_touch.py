# first_love_rag_v2.py
# 基于《第一次亲密接触》小说的RAG问答智能体
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

import os
import dashscope
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).parent / ".env")

# 设置全局 API key
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# ============================================
# 第一步：加载《第一次亲密接触》小说
# ============================================

TXT_FILE_PATH = "/Users/tianhao/Downloads/第一次亲密接触.txt"

# 用 Path 处理路径
file_path = Path(TXT_FILE_PATH)

if not file_path.exists():
    print(f"错误：找不到文件 {file_path}")
    print("请确认文件路径是否正确")
    exit(1)

print(f"正在加载小说：{file_path}")

try:
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()
except:
    loader = TextLoader(str(file_path), encoding="gbk")
    documents = loader.load()

print(f"小说加载成功，总字符数：{len(documents[0].page_content)}")

# ============================================
# 第二步：切分文档
# ============================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"文档已切分为 {len(chunks)} 个文本块")

# ============================================
# 第三步：向量化并构建检索库
# ============================================

print("正在构建向量检索库...")
print("提示：如果下载速度慢，请在 .env 文件中设置 HF_TOKEN 来提高下载速度")
print("正在加载嵌入模型 BAAI/bge-small-zh（离线模式）...")

try:
    # 向量语义搜索（可以理解文字含义），本地检索，离线加载已下载模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={
            'device': 'cpu',
            'local_files_only': True,
        },
        encode_kwargs={'normalize_embeddings': True}
    )
    print("嵌入模型加载成功！")
except Exception as e:
    print(f"加载嵌入模型失败: {e}")
    print("请检查本地模型是否已下载到 Hugging Face 缓存目录")
    exit(1)

try:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("向量库构建完成！")
except Exception as e:
    print(f"构建向量库失败: {e}")
    exit(1)


# ============================================
# 第四步：用 @tool 装饰器定义工具（新版方式）
# ============================================

@tool
def search_novel(query: str) -> str:
    """
    从《第一次亲密接触》小说中检索相关内容。
    当用户问关于小说情节、人物对话、故事发展、经典台词等问题时，使用这个工具。
    例如：痞子蔡和轻舞飞扬第一次见面在哪里？小说里有什么经典台词？
    """
    results = vectorstore.similarity_search(query, k=3)
    
    context_parts = []
    for i, doc in enumerate(results):
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        context_parts.append(f"【片段{i+1}】\n{preview}")
    
    return "\n\n---\n\n".join(context_parts)


@tool
def calculator(expression: str) -> str:
    """用于数学计算，输入一个数学表达式，返回计算结果。"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"


# ============================================
# 第五步：创建 Agent（新版方式）
# ============================================

# 初始化大模型
llm = Tongyi(
    model="qwen-turbo"
)

# 定义 Prompt 模板（新版需要显式提供）
agent_prompt = PromptTemplate.from_template("""
你是一个有帮助的助手，可以回答关于《第一次亲密接触》小说的问题。

你可以使用以下工具：
{tools}

工具名称：{tool_names}

回答时请遵循以下格式：

Question: 用户的问题
Thought: 思考应该做什么
Action: 要使用的工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户的最终回答

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

# 创建 Agent
agent = create_react_agent(
    llm=llm,
    tools=[search_novel, calculator],
    prompt=agent_prompt
)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_novel, calculator],
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

print("\n" + "=" * 60)
print("📖 《第一次亲密接触》问答智能体已启动")
print("=" * 60)


# ============================================
# 第六步：测试和交互
# ============================================

def interactive_mode():
    """交互模式"""
    while True:
        question = input("\n📝 你的问题: ").strip()
        if question.lower() in ['exit', 'quit', '退出']:
            print("再见！")
            break
        if not question:
            continue
        
        print("\n🤔 正在思考...")
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n💬 回答: {result['output']}")
        except Exception as e:
            print(f"出错了: {e}")


if __name__ == "__main__":
    # 运行测试
    print("\n【运行测试用例】\n")
    
    test_questions = [
        "《第一次亲密接触》的主角是谁？",
        "小说里有什么经典台词？",
    ]
    
    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        print("-" * 30)
        try:
            result = agent_executor.invoke({"input": q})
            print(f"回答: {result['output']}")
        except Exception as e:
            print(f"错误: {e}")
    
    # 进入交互模式
    interactive_mode()