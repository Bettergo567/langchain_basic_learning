import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# path = Path(__file__).resolve().parent
path = Path(__file__).parent.resolve()
# print(path)
path = path / ".env"
# print(path)

# 加载环境
load_dotenv(path)

# 拿取环境中的内容
model_id = os.getenv("HF_MODEL_ID")
model_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_provider = os.getenv("HF_PROVIDER")
base_url = "https://router.huggingface.co/v1"

if not model_api:
    raise RuntimeError(
        "You don't have MODEL_API!!!"
    )

model = ChatOpenAI(
    model=model_id,
    api_key=model_api,
    base_url=base_url,
    extra_body={"provider":model_provider}
)

# 方式一（使用LCEL）
# messages = PromptTemplate.from_template("你是一名大模型专家，请解答问题：\n\n{question}")

# chain = messages | model | StrOutputParser()

# result = chain.invoke({"question":"请回答什么是RAG？"})


# 方式二（原汁原味：写message，再喂进model，再解析输出）
messages = [
    {"role":"system", "content":"你是一名大模型专家，擅长用简短的话回答"},
    {"role":"user", "content":"RAG是什么？"}
]
parser = StrOutputParser()
responce = model.invoke(messages)
result = parser.invoke(responce)
print(result)
