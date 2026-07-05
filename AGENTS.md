# AGENTS.md

## 1. Project Overview

This project is a personal knowledge-base RAG question-answering system based on LangChain, DeepSeek, Chroma, BGE Embedding, and Streamlit.

Current project goals:

- Support local TXT / Markdown / PDF documents as the knowledge base.
- Use BGE Embedding for semantic vectorization.
- Use Chroma as the local vector database.
- Use DeepSeek as the chat model.
- Use Streamlit to provide a web question-answering interface.
- Show answer sources and reference snippets.
- Gradually expand into an AI Learning Knowledge Assistant.

## 2. Tech Stack

- Python
- LangChain
- langchain-classic
- langchain-deepseek
- Chroma
- HuggingFace Embeddings
- BAAI/bge-small-zh-v1.5
- DeepSeek Chat
- Streamlit
- pypdf
- python-dotenv

## 3. Project Structure

- `src/ingest.py`: reads documents from `data/raw`, splits text, generates embeddings, and writes vectors into `chroma_db`.
- `src/ask.py`: command-line question-answering entry point.
- `src/app.py`: Streamlit web question-answering entry point.
- `data/raw/`: public sample knowledge-base document directory.
- `chroma_db/`: locally generated vector database; do not commit to Git.
- `.env`: local API key configuration; do not commit to Git.
- `requirements.txt`: dependency list.
- `README.md`: project presentation and usage notes.

## 4. Development Rules

- Do not use a bare `python` command.
- When running Python in this project, always use:

```powershell
C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe
```

- Recommended commands:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ingest.py

& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ask.py

& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' -m streamlit run src\app.py
```

- Do not read, print, modify, or commit `.env`.
- Do not commit `.venv/`, `chroma_db/`, `__pycache__/`, or `*.pyc`.
- Do not upload real personal documents, real API keys, or private PDFs to Git.
- Only explicitly approved public sample files under `data/raw/` may be committed.
- Run `git status` before making changes.
- After making changes, explain which files changed.
- Before committing, inspect staged files and ensure no sensitive files are included.

## 5. Git Workflow

1. Pull the latest code from `master`:

```powershell
git checkout master
git pull origin master
```

2. Create a new branch for each version, for example:

```powershell
git checkout -b v4-upload-rebuild
```

3. After changes are complete, inspect the working tree:

```powershell
git status
git diff
```

4. Add only necessary files. Avoid `git add .` to prevent accidental commits.

5. Before committing, inspect staged files:

```powershell
git diff --cached --name-only
```

6. Confirm these must not appear:

```text
.env
.venv/
chroma_db/
__pycache__/
```

7. Keep commit messages short and clear, for example:

```text
add document upload and rebuild workflow
```

8. Push to the corresponding branch. Do not overwrite `master` directly.

## 6. Verification Commands

Syntax check:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' -m py_compile src\app.py src\ask.py src\ingest.py
```

Build knowledge base:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ingest.py
```

Start command-line QA:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ask.py
```

Start web UI:

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' -m streamlit run src\app.py
```

Git checks:

```powershell
git status
git diff
git diff --cached --name-only
```

## 7. RAG Architecture Notes

1. The user places materials into `data/raw/`.
2. `ingest.py` reads TXT / Markdown / PDF files.
3. Documents are split into chunks.
4. `BAAI/bge-small-zh-v1.5` converts chunks into vectors.
5. Chroma stores vectors in `chroma_db`.
6. The user asks questions through `app.py` or `ask.py`.
7. The retriever searches relevant chunks from Chroma.
8. DeepSeek generates answers based on retrieved content.
9. The page shows the answer and reference sources.

## 8. Future Version Plan

- V4: web file upload and one-click knowledge-base rebuild.
- V5: conversation history.
- V6: adjustable retrieval parameters, such as top-k.
- V7: README and resume project packaging.
- V8: deployment, with careful handling of API keys and the local vector database.

## 9. Output Style for Codex

After each task, Codex should report:

- Which commands were executed.
- Which files were modified.
- Why the changes were made.
- How to verify the result.
- Whether verification passed.
- Any risks or unfinished items.
- If there is a commit, provide the commit hash.
- If there is a push, state the branch name.

## 10. Safety Checklist

Before and after each change, check:

- Whether `.env` was accidentally read or modified.
- Whether `.venv/` was accidentally committed.
- Whether `chroma_db/` was accidentally committed.
- Whether real personal documents were accidentally committed.
- Whether the correct Python path was used.
- Whether `py_compile` passed when Python files were changed.
- Whether the core RAG logic remained stable.
