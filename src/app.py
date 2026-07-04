import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data" / "raw"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "deepseek-chat"
COLLECTION_NAME = "personal_knowledge"
RETRIEVER_TOP_K = 4


def format_source(metadata: dict, index: int) -> str:
    """把检索结果的 metadata 格式化成更适合页面展示的来源标题。"""
    source = metadata.get("source", "unknown")
    page = metadata.get("page", None)

    file_name = Path(str(source)).name

    if page in [None, "", "-", "N/A"]:
        page_text = "无页码"
    else:
        page_text = str(page)

    return f"Source {index}: {file_name} | Page: {page_text}"

def get_raw_documents() -> list[Path]:
    """返回 data/raw 目录下支持的知识库原始文件。"""
    supported_suffixes = {".txt", ".md", ".pdf"}

    if not DATA_DIR.exists():
        return []

    return sorted(
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
    )

@st.cache_resource
def build_rag_chain():
    """构建 RAG 问答链。"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_TOP_K}
    )

    llm = ChatDeepSeek(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一个个人知识库问答助手。

请严格根据下面的资料内容回答问题。

规则：
1. 只能依据资料内容回答。
2. 如果资料里没有答案，请说“我在当前知识库里没有找到明确答案”。
3. 不要编造资料里不存在的内容。
4. 回答要清晰、简洁、有条理。
5. 如果可以，请说明参考来源。

资料内容：
{context}
""",
            ),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    return rag_chain


st.set_page_config(
    page_title="Personal Knowledge Base",
    page_icon="📚",
    layout="centered",
)

st.title("Personal Knowledge Base")
st.caption("Ask questions based on your local documents.")

with st.sidebar:
    st.header("System Info")
    st.markdown(f"**LLM:** {LLM_MODEL}")
    st.markdown(f"**Embedding:** {EMBEDDING_MODEL}")
    st.markdown("**Vector DB:** Chroma")
    st.markdown(f"**Retriever top-k:** {RETRIEVER_TOP_K}")
    st.markdown(f"**Knowledge base path:** `{DB_DIR.name}`")

    st.divider()

    st.markdown("### Documents")
    raw_documents = get_raw_documents()

    if not raw_documents:
            st.caption("No documents found in data/raw")
    else:
        for index, file_path in enumerate(raw_documents, start=1):
            st.markdown(f"{index}. `{file_path.name}`")

    st.divider()

    st.markdown("### How to update knowledge base")
    st.caption("Run this command in the project root:")
    st.markdown(r"`.\.venv\Scripts\python.exe src\ingest.py`")
    
if not os.getenv("DEEPSEEK_API_KEY"):
    st.error("Please set DEEPSEEK_API_KEY in your .env file.")
    st.stop()

if not DB_DIR.exists():
    st.warning("Please run src/ingest.py first to build the knowledge base.")
    st.stop()

question = st.text_input(
    "请输入问题",
    placeholder="Ask something about your documents...",
)

if question:
    rag_chain = build_rag_chain()

    with st.spinner("正在检索知识库并生成回答..."):
        result = rag_chain.invoke({"input": question})

    st.subheader("回答")
    st.markdown(result["answer"])

    context_docs = result.get("context", [])

    st.subheader("参考片段")

    if not context_docs:
        st.info("没有检索到参考片段。")
    else:
        for index, doc in enumerate(context_docs, start=1):
            title = format_source(doc.metadata, index)

            with st.expander(title):
                st.markdown(doc.page_content)