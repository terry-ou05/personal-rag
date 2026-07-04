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
