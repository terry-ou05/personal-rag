# AI Learning Knowledge Assistant

AI Learning Knowledge Assistant is a local personal knowledge-base RAG application built with LangChain, DeepSeek, Chroma, BGE Embedding, and Streamlit. It is designed as a small but complete GitHub and resume showcase project for AI learning notes, internship preparation, and course materials.

The app can load local TXT, Markdown, and PDF documents, convert them into embeddings, store them in a local Chroma vector database, retrieve relevant chunks, and generate answers with DeepSeek while showing retrieved sources and reference snippets.

## Features

- TXT / Markdown / PDF document loading
- Local vector database with Chroma
- BGE Embedding semantic retrieval
- DeepSeek-based answer generation
- Streamlit Web UI
- Web-based document upload
- One-click knowledge base rebuild
- Chat history in UI
- Clear chat history
- Adjustable retriever top-k
- Retrieved sources and reference snippets

## Tech Stack

- Python
- LangChain
- langchain-classic
- DeepSeek
- Chroma
- HuggingFace Embeddings
- BAAI/bge-small-zh-v1.5
- Streamlit
- pypdf
- python-dotenv

## Project Structure

```text
personal-rag/
├── src/
│   ├── ingest.py        # Load raw documents, split text, build Chroma vector DB
│   ├── app.py           # Streamlit web UI for upload, rebuild, and RAG chat
│   └── ask.py           # Command-line RAG question-answering entry point
├── data/raw/            # Public sample documents and local knowledge files
├── chroma_db/           # Local generated vector database, not committed
├── .env                 # Local API key config, not committed
├── requirements.txt     # Python dependencies
└── AGENTS.md            # Project-specific Codex working rules
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

Do not commit `.env` to Git.

## How to Use

1. Put learning materials into `data/raw/`, or upload `.txt`, `.md`, or `.pdf` files in the Streamlit page.
2. Click `Rebuild Knowledge Base` to rebuild the local Chroma vector database.
3. Ask questions in the web chat or command-line interface.
4. Review retrieved sources and reference snippets under each answer.

## Commands

Build or rebuild the knowledge base:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ingest.py
```

Start command-line QA:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ask.py
```

Start the Streamlit web app:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' -m streamlit run src\app.py
```

## Example Questions

- RAG 的核心流程是什么？
- LangChain 在 RAG 里负责什么？
- 我现在适合找哪些 AI 实习岗位？
- 这个 Personal RAG 项目可以怎么写进简历？

## Security Notes

- `.env` is not committed.
- `chroma_db/` is not committed.
- `.venv/` is not committed.
- Private files uploaded to `data/raw/` should not be committed to Git by default.
- Public sample files may be committed, but real personal documents should stay local.

## Resume Description

基于 LangChain、DeepSeek、Chroma、BGE Embedding 和 Streamlit 开发了一个本地个人知识库 RAG 问答系统，支持 TXT / Markdown / PDF 文档加载、网页端文件上传、一键重建知识库、可调 top-k 语义检索、聊天历史展示和回答来源追踪。项目通过 Chroma 保存本地向量库，并使用 DeepSeek 基于检索片段生成回答，可用于 AI 学习资料管理、课程笔记问答和实习准备。
