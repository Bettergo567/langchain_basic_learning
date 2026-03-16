
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Iterable
from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

# 获取环境变量路径
path = Path(__file__).parent.resolve()
path_env = path / ".env"

# 加载环境
load_dotenv(path_env)

# 获取环境变量中的配置
model_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_id = os.getenv("HF_MODEL_ID")
embeding_model_id = os.getenv("HF_EMBEDDING_MODEL")
# print(embeding_model_id)
embeddings = HuggingFaceEmbeddings(model_name=embeding_model_id)
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# print(query_result[:3])

# 建立向量数据库和检索器
## 获取文档 
path_doc = path / "data"
loader = DirectoryLoader(str(path_doc), glob="**/*.txt", loader_cls=TextLoader)
doc = loader.load()

## 文档切分
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
splits_doc = splitter.split_documents(doc)
# print(splits_doc)

## 建立向量数据库
vector_store = FAISS.from_documents(splits_doc, embeddings)

## 建立检索器(拿到最相关的前三个)
retriver = vector_store.as_retriever(search_kwargs={"k": 3})

## 把检索到内容（Document类型）进行组合在一起，并且返回的是str类型
def format_retriver(rag_doc:Iterable[Document]):
    return "\n\n".join(doc.page_content for doc in rag_doc)

# 提示词
prompt = PromptTemplate.from_template(
    "你是公司的客服助理，请基于资料回答问题。\n\n"
    "资料：\n{context}\n"
    "问题：\n{question}\n"
    "要求：回答简洁、准确，不要编造。"
)

# 模型
model = HuggingFaceEndpoint(
    # huggingfacehub_api_token=model_api,
    repo_id=model_id,
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0,
    provider="auto",
)
llm = ChatHuggingFace(llm=model)

# LCEL
chain = (
    {"context": retriver | RunnableLambda[Iterable[Document], str](format_retriver), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

reslut = chain.invoke("你们的退款周期是多久？")
print(reslut)
