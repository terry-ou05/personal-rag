# Project Overview

## Project Background

IT Ops Knowledge Base RAG Assistant is a local RAG question-answering project for simulated IT operations knowledge-base scenarios. It uses public sample runbooks such as server CPU overload, disk capacity, Nginx 502, MySQL slow query, Redis memory usage, and login failure troubleshooting.

This project is not positioned as a commercial product or production service desk. It is a learning and showcase project that demonstrates RAG application development, metadata-aware retrieval, AI coding with Codex, debugging, documentation, and Git branch-based iteration.

## Why I Built This Project

As an AI student preparing for internships, I want a project that is closer to an enterprise knowledge-base scenario than a simple personal note demo. IT operations is a practical domain because troubleshooting documents are structured, source-based, and easy to explain in interviews.

The project also gives me a concrete way to practice using Codex for AI coding tasks, including reading an existing codebase, making scoped changes, running verification commands, documenting decisions, and managing Git branches.

## Problem It Solves

The project helps answer IT troubleshooting questions from local runbooks instead of relying only on general model knowledge. This is useful for:

- Simulating enterprise service desk knowledge-base search
- Searching troubleshooting runbooks quickly
- Filtering documents by category, system, severity, or document type
- Turning TXT, Markdown, and PDF files into a searchable local knowledge base
- Showing retrieved sources so answers can be checked against the original material

## Core Features

- Load TXT, Markdown, and PDF files from `data/raw/`
- Read public sample metadata from `data/metadata.json`
- Split documents into text chunks
- Generate semantic embeddings with `BAAI/bge-small-zh-v1.5`
- Store vectors locally with Chroma
- Ask questions through a Streamlit web interface
- Upload documents from the web UI
- Rebuild the knowledge base with one click
- Keep chat history in the current Streamlit session
- Adjust retriever top-k from the sidebar
- Filter retrieval by metadata fields
- Display retrieved sources and reference snippets
- Run a retrieval-only evaluation baseline with Recall@K, MRR, zero-hit rate, and latency
- Compare Dense, BM25, and Hybrid RRF retrieval modes
- Compare Cross-Encoder reranking on top of Dense and Hybrid candidates

## Tech Stack

- Python
- LangChain
- langchain-classic
- langchain-deepseek
- DeepSeek Chat
- Chroma
- HuggingFace Embeddings
- BAAI/bge-small-zh-v1.5
- rank-bm25
- jieba
- sentence-transformers CrossEncoder
- BAAI/bge-reranker-base
- Streamlit
- pypdf
- python-dotenv
- Git

## Current Version

The current version is a local runnable IT ops showcase version. It includes the full basic RAG workflow, public IT troubleshooting sample documents, metadata ingestion, a Streamlit UI, web-based document upload, one-click knowledge-base rebuild, chat history, adjustable top-k retrieval, metadata filters, source/reference snippet display, retrieval evaluation baselines, optional Hybrid Retrieval, and optional Cross-Encoder reranking.

It is intended for GitHub and resume presentation. It does not include public deployment, real enterprise documents, multi-user accounts, cloud storage, OCR, complex agent workflows, reranking, Qdrant, or production monitoring.

## V7 Retrieval Baseline

V7 adds a repeatable retrieval-only evaluation. The goal is to measure whether the dense Chroma retriever can find the expected source document before answer generation. The evaluation does not call DeepSeek, so results are not affected by generation quality.

Current baseline:

- Questions: 30
- Documents: 10
- Chunks: 16
- Recall@1: 86.67%
- Recall@3: 100.00%
- Recall@5: 100.00%
- MRR: 0.9278
- Zero-hit Rate: 0.00%
- Average Retrieval Latency: 9.81 ms
- P95 Retrieval Latency: 10.35 ms

This gives the project a measurable starting point before adding Hybrid Retrieval, BM25, RRF, reranking, or query rewrite.

## V8 Hybrid Retrieval Result

V8 adds BM25 sparse retrieval and Reciprocal Rank Fusion while keeping the V7 dense retriever available. BM25 is useful for exact tokens such as commands, error codes, file names, configuration keys, and service names. Dense retrieval is still better for short symptom descriptions.

The V8 evaluation compares three modes on the same 30 questions:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |
|---|---:|---:|---:|---:|---:|
| Dense | 86.67% | 100.00% | 100.00% | 0.9278 | 0.00% |
| BM25 | 53.33% | 70.00% | 70.00% | 0.6167 | 30.00% |
| Hybrid RRF | 86.67% | 96.67% | 100.00% | 0.9122 | 0.00% |

Hybrid improved `disk-002` from rank 2 to rank 1 because the query contains `df -h` and `90%`. It also introduced one Top-1 regression on `login-003`, and its MRR is lower than Dense. Based on this small dataset, the app keeps Dense as the default and exposes Hybrid RRF as an optional mode.

## V9 Reranker Result

V9 adds a Cross-Encoder reranker on top of candidate retrieval. It uses `BAAI/bge-reranker-base` on CPU. The reranker does not retrieve new documents; it only reranks Top-5 candidates from Dense or Hybrid.

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |
|---|---:|---:|---:|---:|---:|
| Dense | 86.67% | 100.00% | 100.00% | 0.9278 | 0.00% |
| Hybrid RRF | 86.67% | 96.67% | 100.00% | 0.9122 | 0.00% |
| Dense + Reranker | 96.67% | 100.00% | 100.00% | 0.9833 | 0.00% |
| Hybrid + Reranker | 96.67% | 100.00% | 100.00% | 0.9833 | 0.00% |

Dense + Reranker improved `disk-002`, `disk-003`, and `login-005` to Top-1, with no Top-1 regressions. `nginx-005` stayed at rank 2. The tradeoff is latency: model loading took about 8.20 seconds in the final cached run, and warm reranking averaged about 749 ms on CPU. The Streamlit UI defaults to Dense + Reranker for quality, while Dense remains available as the faster mode.

## Future Extensions

Possible next steps include:

- Add screenshots after local UI testing
- Add a small test document set for reproducible demos
- Improve error messages for missing API keys or failed PDF parsing
- Add lightweight unit tests for document loading and source formatting
- Add hybrid retrieval with BM25 + vector search
- Add retrieval trace for interview-friendly explanation
- Add query rewrite only if future reports show unresolved vague-query failures
- Add retrieval sufficiency checks before asking DeepSeek to answer
- Add a safer upload overwrite confirmation flow
