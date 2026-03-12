from email import message
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import api_key
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 获取环境配置路径
path = Path(__file__).parent.resolve()
path = path / ".env"

# 加载环境
load_dotenv(path)

# 获取环境配置文件的内容
model_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_id = os.getenv("HF_MODEL_ID")
base_url = os.getenv("HF_BASE_URL")

if model_api is None:
    raise RuntimeError(
        "API IS NONE！！！"
    )

# 要求模型按照特定的输出格式进行输出
## 先编写输出指令的类, 必须继承BaseModel否则你写的Field不会生效
class OutputAssign(BaseModel):
    # 为什么要使用Field，因为这个函数可以用来控制写的类中的变量得到检查
    # 比如：这个变量使用了 Field(...， ge=0），则检查这个变量是否赋值 和 赋值必须大于0，否则报错
    # Field中的...表示占位符，要求一定要对这个变量进行赋值，否则报错
    short_answer: str = Field(..., description="简短的回答")
    confidence: float = Field(..., ge=0, le=1, description="0-1之间的置信度")
    key_points: list[str] = Field(...,description="5个要点, 每个要点使用序号标注")

## 编写对模型的输出指令
parser = PydanticOutputParser[OutputAssign](pydantic_object=OutputAssign)
format_instruct_ = parser.get_format_instructions()

# 消息，使用ChatPromp模板(也可以使用PromptTemplate)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个严谨的助手。严格按要求输出，不要输出多余文本。"),
        ("human", """问题：{question}, {format_instruct}""")
    ]
)
# prompt = PromptTemplate.from_template("你是一名大模型专家，严格按要求输出，不要输出多余文本。请解答问题：{question}, {format_instruct}")

# 定义模型
## 因为这是一个文本生成任务，要使用到ChatHuggingFace，所以先创建HuggingFaceEndpoint
model = HuggingFaceEndpoint(
    huggingfacehub_api_token=model_api,
    repo_id=model_id,
    # task="text-generation",
    do_sample=False,
    provider='auto',
    repetition_penalty=1.03,
)
## 再得到ChatHuggingFace
llm = ChatHuggingFace(llm=model)

# LCEL
## 注意prompt使用partial先把format_instruct填进去是常见的用法
chain = prompt.partial(format_instruct=format_instruct_) | llm | StrOutputParser()

"""
还可以
chain.invoke({
    "question": "请介绍一下RAG",
    "format_instruct": format_instruct_ # 这样是可以的
})
"""

# 执行
result = chain.invoke({"question":"请介绍一下RAG"})
print(result,flush=True)




