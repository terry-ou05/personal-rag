from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "personal_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        page_content=text,
        metadata={"source": str(path.relative_to(PROJECT_ROOT))},
    )


def load_pdf_file(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "page": page_index,
                },
            )
        )
    return documents


def load_documents() -> list[Document]:
    if not RAW_DATA_DIR.exists():
        return []

    documents = []
    for path in sorted(RAW_DATA_DIR.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            document = load_text_file(path)
            if document.page_content.strip():
                documents.append(document)
        elif suffix == ".pdf":
            documents.extend(load_pdf_file(path))

    return documents


def reset_vector_store() -> None:
    if not CHROMA_DIR.exists():
        return

    project_root = PROJECT_ROOT.resolve()
    db_dir = CHROMA_DIR.resolve()

    if not db_dir.is_relative_to(project_root):
        raise RuntimeError(f"Refusing to remove vector store outside project: {db_dir}")

    shutil.rmtree(db_dir)


def build_vector_store(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return chunks


def build_knowledge_base(reset: bool = True) -> dict:
    if reset:
        reset_vector_store()

    documents = load_documents()
    if not documents:
        return {
            "documents": 0,
            "chunks": 0,
            "db_path": str(CHROMA_DIR),
        }

    chunks = build_vector_store(documents)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "db_path": str(CHROMA_DIR),
    }


def main() -> None:
    result = build_knowledge_base(reset=True)

    print("知识库构建完成。")

    if result["documents"] == 0:
        print(f"未在 {RAW_DATA_DIR} 找到可用的 .txt、.md 或 .pdf 文档。")

    print(f"原始文档数量: {result['documents']}")
    print(f"切分后片段数量: {result['chunks']}")
    print(f"向量数据库位置: {result['db_path']}")


if __name__ == "__main__":
    main()
