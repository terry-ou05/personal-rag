import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "personal_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
NO_ANSWER = "我在当前知识库里没有找到明确答案。"


@st.cache_resource
def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def get_llm() -> ChatDeepSeek | None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=api_key,
    )


def format_context(docs) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}, page {page}" if page else source
        blocks.append(f"[资料 {index}: {location}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(llm: ChatDeepSeek, docs, question: str) -> str:
    prompt = f"""你是一个个人知识库问答助手。请严格根据下面检索到的资料回答问题。

规则：
1. 只能使用资料中的信息回答。
2. 不要编造资料中没有出现的事实。
3. 如果资料里没有明确答案，只回答：{NO_ANSWER}

检索到的资料：
{format_context(docs)}

问题：
{question}

回答："""
    response = llm.invoke(prompt)
    return response.content.strip()


st.set_page_config(page_title="Personal RAG")
st.title("Personal RAG")

if not CHROMA_DIR.exists():
    st.warning("chroma_db 不存在。请先运行 ingest.py 构建知识库。")
    st.code(r"& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ingest.py", language="powershell")
    st.stop()

llm = get_llm()
if llm is None:
    st.warning("缺少 DEEPSEEK_API_KEY。请先在 .env 中填写后再提问。")
    st.stop()

question = st.text_input("请输入问题")
if question:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    with st.spinner("正在生成回答..."):
        answer = answer_question(llm, docs, question)

    st.subheader("回答")
    st.write(answer)

    st.subheader("参考片段")
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "-")
        with st.expander(f"片段 {index} | source: {source} | page: {page}"):
            st.write(doc.page_content)
