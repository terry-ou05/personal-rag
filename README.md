# Personal RAG

Personal RAG 是一个最小可运行的个人知识库问答系统。它会把 `data/raw` 下的文档切分成文本块，使用本地 BGE Embedding 生成向量，存入 Chroma，然后通过 DeepSeek 大模型基于检索结果回答问题。

## 技术栈

- Python 3.13
- LangChain
- DeepSeek Chat
- Chroma
- HuggingFace Embeddings
- BAAI/bge-small-zh-v1.5
- pypdf
- Streamlit

## Features

- TXT / Markdown / PDF document ingestion
- Local Chroma vector database
- DeepSeek-based answers with retrieved context
- Streamlit web interface
- Reference source display
- Web-based document upload
- One-click knowledge base rebuild

## 配置 .env

在项目根目录创建或编辑 `.env`：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
```

不要把 `.env` 提交到公开仓库。

## 放入资料

把个人资料放到：

```text
data/raw
```

当前支持的文件类型：

- `.txt`
- `.md`
- `.pdf`

PDF 会按页读取，并保留 `source` 和 `page` metadata。TXT / MD 会直接读取文本，并保留 `source` metadata。

## 构建知识库

在项目根目录运行：

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ingest.py
```

成功后会生成：

```text
chroma_db
```

## 启动命令行问答

确认 `.env` 已填写 `DEEPSEEK_API_KEY` 后运行：

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' src\ask.py
```

输入 `exit`、`quit` 或 `q` 可以退出。

## 启动网页

确认 `.env` 已填写 `DEEPSEEK_API_KEY`，并且已经运行过 `src\ingest.py` 后执行：

```powershell
& 'C:\Users\14985\Desktop\personal-rag\.venv\Scripts\python.exe' -m streamlit run src\app.py
```

## 网页端使用 V4 功能

1. 打开 Streamlit 页面。
2. 上传 `.txt`、`.md` 或 `.pdf` 文件。
3. 点击 `Rebuild Knowledge Base`。
4. 等待页面显示 documents/chunks 数量。
5. 基于新上传的资料提问，并查看参考来源。

## Notes

- 上传到 `data/raw/` 的个人文件默认不应提交到 Git。
- `chroma_db` 是本地生成的向量数据库，不提交到 Git。
- `.env` 只保存本地 API Key，不提交到 Git。
