import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "personal_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
NO_ANSWER = "我在当前知识库里没有找到明确答案。"


def load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def format_context(docs) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}, page {page}" if page else source
        blocks.append(f"[资料 {index}: {location}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_sources(docs) -> str:
    seen = set()
    lines = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "-")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- source: {source}, page: {page}")
    return "\n".join(lines)


def answer_question(llm: ChatDeepSeek, retriever, question: str) -> tuple[str, list]:
    docs = retriever.invoke(question)
    context = format_context(docs)
    prompt = f"""你是一个个人知识库问答助手。请严格根据下面检索到的资料回答问题。

规则：
1. 只能使用资料中的信息回答。
2. 不要编造资料中没有出现的事实。
3. 如果资料里没有明确答案，只回答：{NO_ANSWER}

检索到的资料：
{context}

问题：
{question}

回答："""
    response = llm.invoke(prompt)
    return response.content.strip(), docs


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY。请先在 .env 中填写 DEEPSEEK_API_KEY 后再运行 ask.py。")
        return

    if not CHROMA_DIR.exists():
        print("chroma_db 不存在。请先运行: .\\.venv\\Scripts\\python.exe src\\ingest.py")
        return

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=api_key,
    )

    print("个人知识库问答已启动。输入 exit / quit / q 退出。")
    while True:
        question = input("\n请输入问题: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not question:
            continue

        answer, docs = answer_question(llm, retriever, question)
        print(f"\n回答:\n{answer}")
        print("\n参考来源:")
        print(format_sources(docs) or "- 无")


if __name__ == "__main__":
    main()
