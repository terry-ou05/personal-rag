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

def get_raw_documents() -> list[Path]:
    """返回 data/raw 目录下支持的知识库原始文件。"""
    if not DATA_DIR.exists():
        return []

    return sorted(
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES
    )


def save_uploaded_documents(uploaded_files) -> int:
    """Save supported uploaded files into data/raw using safe file names."""
    if not uploaded_files:
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    processed_uploads = st.session_state.setdefault("processed_uploads", set())

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        suffix = Path(file_name).suffix.lower()

        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            st.warning(f"Unsupported file type skipped: {file_name}")
            continue

        upload_key = f"{file_name}:{uploaded_file.size}"
        if upload_key in processed_uploads:
            continue

        target_path = DATA_DIR / file_name
        if target_path.exists():
            st.info(f"Overwriting existing file: {file_name}")

        target_path.write_bytes(uploaded_file.getvalue())
        processed_uploads.add(upload_key)
        saved_count += 1

    return saved_count


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

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)
saved_files = save_uploaded_documents(uploaded_files)

if saved_files:
    st.success(f"Uploaded {saved_files} file(s). Please rebuild the knowledge base.")

if st.button("Rebuild Knowledge Base"):
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
    st.caption("Upload documents, then click Rebuild Knowledge Base.")
    st.caption("Or run this command in the project root:")
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
