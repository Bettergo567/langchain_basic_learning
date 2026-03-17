from ast import Store
from pathlib import Path
from typing import Iterable
from dashscope import History
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 获取路径
path = Path(__file__).parent.resolve()
path_env = path / ".env"
path_doc = path / "data"

# 加载环境
load_dotenv(path_env)

# 获取环境变量
model_id = os.getenv("HF_MODEL_ID")
embeddings_model_id = os.getenv("embeddings_model_id")

# 建立向量数据库和检索器
## 加载文档
print(str(path_doc))
loader = DirectoryLoader(str(path_doc), glob="**/*.txt", loader_cls=TextLoader)
doc = loader.load()

## 切分文档
spiliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
spilit_doc = spiliter.split_documents(doc)

## 建立向量数据库
embedding_model = HuggingFaceEmbeddings(model_name=embeddings_model_id)
vector_store = FAISS.from_documents(spilit_doc,embedding_model)

## 建立检索器
retriver = vector_store.as_retriever(search_kwargs={"k": 3})

# 把检索到的内容拼接起来，并变为str形式
def format_retrivers(rag_doc:Iterable[Document]):
    return "\n\n".join(doc.page_content for doc in rag_doc)

# 提示词
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是公司的客服助理。请基于资料回答和历史记录，不确定就明确说不知道。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "资料：\n{context}\n\n问题：{question}")
    ]
)

# 模型
model = HuggingFaceEndpoint(
    repo_id=model_id,
    provider="auto",
    temperature=0.1,
    max_new_tokens=256,
)
llm = ChatHuggingFace(llm=model)

# LCEL
## 创建base_chain，其中RunnablePassthrough.assign添加新的context内容
## {"question": xxxxx, "history": xxxxxxx} --> {"question" : xxxx, "history" : xxxxxx, "context" : xxxxxx}
base_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: retriver.invoke(x["question"])) | RunnableLambda(format_retrivers)
    )
    | prompt
    | llm
    | StrOutputParser()
)

## 把history 和 question 填入
### 创建一个存储history的store
store: dict[str, BaseChatMessageHistory] = {}

### 创建一个可以拿到用户自己的 history
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

### RunnableWithMessageHistory的作用是： 
"""
    1. 拿到 question 和 通过 get_session_history 函数拿到 history
    2. 创建一个 dict{"question": xxxxx, "history": xxxxxxx}消息
    3. 把这个 Message 喂给 base_chain
    4. 每次运行一次都会把当前的累计 history 自动添加进 history
"""
chain_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

### 创建一个函数包装了模型的输入和输出，减少冗余
def ask(session_id:str, question:str):
    result = chain_history.invoke(
        {"question" : question},
        config={
            "configurable" : {"session_id" : session_id}
        }
    )
    print(f"session_id:{session_id}, question: {question}")
    print(f"session_id:{session_id}, answer: {result}\n")

if __name__ == "__main__":
    # 同一会话：第二问可承接第一问语境
    ask("demo-user-1", "我是梦泽，我想退款，多久能到账？")
    ask("demo-user-1", "那客服工作时间呢？")
    ask("demo-user-1", "我是谁？")

    # 新会话：历史隔离，不共享上下文
    ask("demo-user-2", "那客服工作时间呢？")


