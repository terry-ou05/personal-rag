import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from ingest import build_knowledge_base


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data" / "raw"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "deepseek-chat"
COLLECTION_NAME = "personal_knowledge"
RETRIEVER_TOP_K = 4
SUPPORTED_UPLOAD_SUFFIXES = {".txt", ".md", ".pdf"}


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


def build_source_items(docs) -> list[dict]:
    source_items = []
    for index, doc in enumerate(docs, start=1):
        source_items.append(
            {
                "title": format_source(doc.metadata, index),
                "content": doc.page_content,
            }
        )
    return source_items


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No reference snippets found.")
        return

    for source in sources:
        with st.expander(source["title"]):
            st.markdown(source["content"])


def get_raw_documents() -> list[Path]:
    """返回 data/raw 目录下支持的知识库原始文件。"""
    if not DATA_DIR.exists():
        return []

    return sorted(
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES
    )


def save_uploaded_documents(uploaded_files) -> list[str]:
    """Save supported uploaded files into data/raw using safe file names."""
    if not uploaded_files:
        return []

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []
    processed_uploads = st.session_state.setdefault("processed_uploads", set())

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        suffix = Path(file_name).suffix.lower()

        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            st.warning(f"Unsupported file type skipped: {file_name}")
            continue

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        upload_key = f"{file_name}:{uploaded_file.size}:{file_hash}"
        if upload_key in processed_uploads:
            continue

        target_path = DATA_DIR / file_name
        if target_path.exists():
            st.info(f"Existing file overwritten: {file_name}")

        target_path.write_bytes(file_bytes)
        processed_uploads.add(upload_key)
        saved_files.append(file_name)

    return saved_files


@st.cache_resource
def build_rag_chain(top_k: int):
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
        search_kwargs={"k": top_k}
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
    page_title="AI Learning Knowledge Assistant",
    page_icon="📚",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("AI Learning Knowledge Assistant")
st.caption("Ask questions based on your local AI learning documents.")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)
saved_file_names = save_uploaded_documents(uploaded_files)

if saved_file_names:
    st.success(
        f"Uploaded {len(saved_file_names)} file(s). "
        "Click Rebuild Knowledge Base to make them searchable."
    )

raw_documents = get_raw_documents()
db_ready = DB_DIR.exists()

with st.sidebar:
    st.header("System Info")
    st.markdown(f"**LLM:** {LLM_MODEL}")
    st.markdown(f"**Embedding:** {EMBEDDING_MODEL}")
    st.markdown("**Vector DB:** Chroma")
    retriever_top_k = st.slider(
        "Retriever top-k",
        min_value=1,
        max_value=8,
        value=RETRIEVER_TOP_K,
    )
    st.markdown(f"**Retriever top-k:** {retriever_top_k}")
    st.markdown(f"**Knowledge base path:** `{DB_DIR.name}`")

    st.divider()

    st.markdown("### Documents")
    if not raw_documents:
        st.caption("No documents found in data/raw")
    else:
        for index, file_path in enumerate(raw_documents, start=1):
            st.markdown(f"{index}. `{file_path.name}`")

    st.divider()

    st.markdown("### Knowledge Base")
    st.markdown(f"**Vector DB status:** {'Ready' if db_ready else 'Not built'}")
    st.markdown(f"**Raw files:** {len(raw_documents)}")
    st.markdown("**Supported formats:** `.txt` / `.md` / `.pdf`")

    st.divider()

    st.markdown("### Actions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    rebuild_clicked = st.button("Rebuild Knowledge Base")

    st.divider()

    st.markdown("### How to update knowledge base")
    st.caption("Or run this command in the project root:")
    st.markdown(r"`.\.venv\Scripts\python.exe src\ingest.py`")

if rebuild_clicked:
    with st.spinner("Rebuilding knowledge base..."):
        st.cache_resource.clear()
        rebuild_result = build_knowledge_base(reset=True)
        st.cache_resource.clear()

    if rebuild_result["documents"] == 0 or rebuild_result["chunks"] == 0:
        st.warning("No valid documents found in data/raw. Please upload supported files first.")
    else:
        st.success(
            "Knowledge base rebuilt. "
            f"Documents: {rebuild_result['documents']}, "
            f"Chunks: {rebuild_result['chunks']}"
        )

if not os.getenv("DEEPSEEK_API_KEY"):
    st.error("Please set DEEPSEEK_API_KEY in your .env file.")
    st.stop()

if not DB_DIR.exists():
    st.warning("Please run src/ingest.py first to build the knowledge base.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    rag_chain = build_rag_chain(retriever_top_k)

    with st.spinner("正在检索知识库并生成回答..."):
        result = rag_chain.invoke({"input": question})

    context_docs = result.get("context", [])
    sources = build_source_items(context_docs)
    answer = result["answer"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
        render_sources(sources)
