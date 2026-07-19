# RAG Workflow

This note explains how Retrieval-Augmented Generation works in this IT operations showcase project and is intended as a quick interview review document.

## What RAG Means

RAG stands for Retrieval-Augmented Generation. The main idea is to retrieve relevant information from a knowledge base first, then let a language model generate an answer based on that retrieved context.

In this project, RAG helps the assistant answer IT operations troubleshooting questions from local runbooks instead of only relying on the model's general knowledge.

## Workflow in This Project

```text
data/raw + data/metadata.json
-> src/ingest.py
-> text chunks
-> BGE Embedding
-> Chroma with metadata
-> retriever with optional metadata filters
-> DeepSeek
-> Streamlit answer with sources
```

## Step-by-Step Explanation

1. `data/raw/`

   This folder contains local knowledge files. The V6 showcase includes public IT operations runbooks. The project supports `.txt`, `.md`, and `.pdf` files. Public sample files can be committed, but private uploaded files should stay local.

2. `data/metadata.json`

   This file stores sample metadata for public IT operations documents, including category, system, document type, severity, and tags. This metadata is attached to documents before they are written into Chroma.

3. `src/ingest.py`

   This script reads supported files from `data/raw/`. TXT and Markdown files are read as text. PDF files are read page by page with `pypdf`, and page metadata is preserved.

4. Text chunks

   Long documents are split into smaller chunks using `RecursiveCharacterTextSplitter`. In the current project, the chunk settings are:

   - `chunk_size=800`
   - `chunk_overlap=150`

   Chunking makes retrieval more precise because the system can search smaller pieces of text.

5. BGE Embedding

   The project uses `BAAI/bge-small-zh-v1.5` through HuggingFace Embeddings. Embedding converts text chunks into vectors so that semantic similarity can be calculated.

6. Chroma vector database

   Chroma stores the generated vectors and metadata locally in `chroma_db/`. This folder is generated locally and should not be committed to Git.

7. Retriever

   The retriever searches Chroma for chunks that are semantically close to the user's question. In V6, optional metadata filters can limit retrieval by category, system, severity, or document type.

8. DeepSeek Chat Model

   DeepSeek receives the user question and retrieved context, then generates an answer. The prompt asks the model to answer based on the retrieved materials and avoid inventing unsupported information.

9. Streamlit UI

   `src/app.py` provides the web interface. Users can upload documents, rebuild the knowledge base, ask questions, view chat history, adjust top-k, apply metadata filters, and inspect retrieved sources.

## What top-k Means

`top-k` controls how many relevant chunks the retriever returns from Chroma.

For example:

- `top-k=1` returns only the most similar chunk
- `top-k=4` returns four relevant chunks
- `top-k=8` returns more context, but may include less focused information

In this project, top-k is adjustable in the Streamlit sidebar. This is useful for testing how retrieval breadth affects answer quality.

## What Metadata Filtering Means

Metadata filtering narrows the search space before retrieval. For example, if the user selects:

- `system = nginx`
- `severity = high`

the retriever searches only chunks whose metadata matches those fields. This makes the project closer to an enterprise knowledge-base scenario where documents are usually organized by system, category, severity, and document type.

## Why Show Sources and Reference Snippets

Sources and reference snippets make the answer more trustworthy and easier to verify. They help users check:

- Which file the answer came from
- Whether the retrieved content actually supports the answer
- Whether the model missed or misunderstood the original material

For interviews, this is important because it shows that the project is not just calling an LLM directly. It includes a retrieval layer and exposes evidence behind the generated answer.

## Retrieval Evaluation Baseline

V7 adds `src/evaluate_retrieval.py`, which evaluates only the Chroma dense retriever. It does not call DeepSeek and does not generate answers.

The evaluation questions live in `eval/questions.json`. Each question includes:

- `question`
- `expected_source`
- `expected_system`
- `expected_category`
- `difficulty`
- `query_type`

The script retrieves Top-K chunks, compares each returned metadata `source` with `expected_source`, and calculates:

- Recall@1, Recall@3, Recall@5
- MRR
- Zero-hit Rate
- Average Retrieval Latency
- P95 Retrieval Latency

Recall@K means the expected source appears within the top K retrieved chunks. MRR means Mean Reciprocal Rank, which rewards the expected source appearing earlier in the result list.

Current V7 result:

- Recall@1: 86.67%
- Recall@3: 100.00%
- Recall@5: 100.00%
- MRR: 0.9278
- Zero-hit Rate: 0.00%
- Average Retrieval Latency: 9.81 ms
- P95 Retrieval Latency: 10.35 ms

This baseline matters because V8 can compare Hybrid Retrieval against the current dense retriever instead of relying on subjective impressions.

## How to Explain This in an Interview

A short explanation:

> I built a local IT operations RAG assistant that reads troubleshooting runbooks from `data/raw`, attaches metadata from `data/metadata.json`, splits documents into chunks, converts chunks into BGE embeddings, stores them in Chroma, retrieves relevant chunks with optional metadata filters, and uses DeepSeek to generate answers in a Streamlit UI. I also display sources and snippets so the answer can be checked against the original documents.
