# RAG Workflow

This note explains how Retrieval-Augmented Generation works in this project and is intended as a quick interview review document.

## What RAG Means

RAG stands for Retrieval-Augmented Generation. The main idea is to retrieve relevant information from a knowledge base first, then let a language model generate an answer based on that retrieved context.

In this project, RAG helps the assistant answer questions from local learning documents instead of only relying on the model's general knowledge.

## Workflow in This Project

```text
data/raw
-> src/ingest.py
-> text chunks
-> BGE Embedding
-> Chroma
-> retriever
-> DeepSeek
-> Streamlit answer with sources
```

## Step-by-Step Explanation

1. `data/raw/`

   This folder contains local knowledge files. The project supports `.txt`, `.md`, and `.pdf` files. Public sample files can be committed, but private uploaded files should stay local.

2. `src/ingest.py`

   This script reads supported files from `data/raw/`. TXT and Markdown files are read as text. PDF files are read page by page with `pypdf`, and page metadata is preserved.

3. Text chunks

   Long documents are split into smaller chunks using `RecursiveCharacterTextSplitter`. In the current project, the chunk settings are:

   - `chunk_size=800`
   - `chunk_overlap=150`

   Chunking makes retrieval more precise because the system can search smaller pieces of text.

4. BGE Embedding

   The project uses `BAAI/bge-small-zh-v1.5` through HuggingFace Embeddings. Embedding converts text chunks into vectors so that semantic similarity can be calculated.

5. Chroma vector database

   Chroma stores the generated vectors locally in `chroma_db/`. This folder is generated locally and should not be committed to Git.

6. Retriever

   The retriever searches Chroma for chunks that are semantically close to the user's question. These retrieved chunks become the context for the model.

7. DeepSeek Chat Model

   DeepSeek receives the user question and retrieved context, then generates an answer. The prompt asks the model to answer based on the retrieved materials and avoid inventing unsupported information.

8. Streamlit UI

   `src/app.py` provides the web interface. Users can upload documents, rebuild the knowledge base, ask questions, view chat history, adjust top-k, and inspect retrieved sources.

## What top-k Means

`top-k` controls how many relevant chunks the retriever returns from Chroma.

For example:

- `top-k=1` returns only the most similar chunk
- `top-k=4` returns four relevant chunks
- `top-k=8` returns more context, but may include less focused information

In this project, top-k is adjustable in the Streamlit sidebar. This is useful for testing how retrieval breadth affects answer quality.

## Why Show Sources and Reference Snippets

Sources and reference snippets make the answer more trustworthy and easier to verify. They help users check:

- Which file the answer came from
- Whether the retrieved content actually supports the answer
- Whether the model missed or misunderstood the original material

For interviews, this is important because it shows that the project is not just calling an LLM directly. It includes a retrieval layer and exposes evidence behind the generated answer.

## How to Explain This in an Interview

A short explanation:

> I built a local RAG assistant that reads documents from `data/raw`, splits them into chunks, converts chunks into BGE embeddings, stores them in Chroma, retrieves relevant chunks for each question, and uses DeepSeek to generate answers in a Streamlit UI. I also display sources and snippets so the answer can be checked against the original documents.
