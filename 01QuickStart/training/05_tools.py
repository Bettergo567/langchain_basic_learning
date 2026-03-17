import os
import re
import ast
import operator as op
from pathlib import Path
from typing import Iterable, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 获取路径
path = Path(__file__).parent.resolve()
path_env = path / ".env"
path_doc = path / "data"

# 加载环境变量
load_dotenv(path_env)

# 获取环境变量中的值
model_id = os.getenv("HF_MODEL_ID")
embeddings_model_id = os.getenv("embeddings_model_id")
# print(model_id)

# 建立向量数据库和检索器
## 获取文档
loader = DirectoryLoader(str(path_doc), glob="**/*.txt", loader_cls=TextLoader)
doc = loader.load()

## 建立向量数据库
spilitor = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
doc_spilit = spilitor.split_documents(doc)
embedding_model = HuggingFaceEmbeddings(model_name=embeddings_model_id)
vector_store = FAISS.from_documents(doc_spilit, embedding_model)
## 建立检索器
retriver = vector_store.as_retriever(search_kwargs={"k": 3})

## 拼接检索的内容，并变为 str
def retrive_format(rag_doc : Iterable[Document]):
    return "\n\n".join(doc.page_content for doc in rag_doc)

# Prompt
Prompt = PromptTemplate.from_template(
    "你是一名智能回答助手，请根据输入生成答案。\n"
    "问题：{question}\n"
    "工具：{tool}\n"
    "调用工具的输出结果：{tool_output}\n"
    "上下文：{history}\n"
    "规则：\n"
    "1) 若工具输出有明确结果，优先使用工具输出。\n"
    "2) 否则参考检索上下文回答。\n"
    "3) 不确定就直接说不知道，不要编造。\n"
    "4) 回答简洁。"
)

# 拿取工具
def get_tool(question: str) -> str:
    cal_words = ["计算", "加", "算术"]
    retrive_words = ["退款", "客服", "工作时间", "付款", "套餐", "知识库", "支持哪些"]

    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+",question) or any(q in question for q in cal_words):
        return "calcutor"
    elif any(q in question for q in retrive_words):
        return "search_knowledge"
    return "rag"

# 提取表达式
def extract_exp(question:str):
    exp = re.search(r"[0-9\.\s\+\-\*\/\(\)]{3,}", question)
    return exp.group(0).strip()

# 计算
def safe_cal(expression:str):
    """Calculate a basic arithmetic expression like (12 + 3) * 2."""
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node: Any) -> float:
        # _eval 运行逻辑是：
        # 1. 常量节点：只接受 int/float，直接返回
        # 2. 二元运算节点：仅在白名单运算符中才递归计算
        # 3. 一元运算节点：仅在白名单运算符中才递归计算
        # 4. 其他 AST 节点一律拒绝，抛 Unsupported expression
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed.body)

@tool
def calculator(expression:str) -> str:
    """Calculate a basic arithmetic expression like (12 + 3) * 2."""
    partten = set("0123456789.+-*/() ")
    if any(p not in partten for p in expression):
        return "表达式有问题！超出计算规则的范围"
    else:
        try:
            result = safe_cal(expression)
            return f"{expression} == {result}"
        except Exception as exc:
            return f"Calculation error: {exc}"

@tool
def search_knowledge(query:str) -> str:
    """Search local knowledge base snippets related to the query."""
    result = retriver.invoke(query)
    if not result:
        return "No retrive knlowdge found"
    return retrive_format(result)

# 执行工具
def excutive_tool(message: dict):
    question = message["question"]
    tool = message["tool"]

    if tool == "calcutor":
        expression = extract_exp(question)
        if expression:
            result = calculator.invoke({"expression" : expression})
            return result
        else:
            return "No Expression!"
    elif tool == "search_knowledge":
        result = search_knowledge.invoke({"query" : question})
        return result
    else:
        return "No Tool Use"
    

# 模型
model = HuggingFaceEndpoint(
    repo_id=model_id,
    provider="auto",
    max_new_tokens=256,
    temperature=0.1
)
llm = ChatHuggingFace(llm=model)

chain = (
    RunnablePassthrough.assign(
        tool=RunnableLambda(lambda x: get_tool(x["question"]))
    )
    | RunnablePassthrough.assign(
        tool_output=RunnableLambda(excutive_tool),
        history=RunnableLambda(lambda x: retrive_format(retriver.invoke(x["question"])))
    )
    | Prompt
    | llm
    | StrOutputParser()
)

def ask(question: str):
    tool = get_tool(question)
    tool_output = excutive_tool({"question": question, "tool": tool})
    tool_used = tool if tool in ["calcutor", "search_knowledge"] and tool_output != "No Tool Use" else "None"

    print(f"Q: {question}")
    print("A: ", end="", flush=True)

    for chunk in chain.stream({"question":question}):
        print(chunk, end="", flush=True)
    
    print(f"\nTool: {tool_used}\n")
    print("-"*30)

if __name__ == "__main__":
    # q = "请计算 3 + 2"
    # s = extract_exp(q)
    # print(s)
    ask("计算 (12 + 8) * 3")
    ask("我们退款多久到账？")
    ask("请简述 RAG 是什么")