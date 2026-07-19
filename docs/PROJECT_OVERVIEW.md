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

## Tech Stack

- Python
- LangChain
- langchain-classic
- langchain-deepseek
- DeepSeek Chat
- Chroma
- HuggingFace Embeddings
- BAAI/bge-small-zh-v1.5
- Streamlit
- pypdf
- python-dotenv
- Git

## Current Version

The current version is a local runnable IT ops showcase version. It includes the full basic RAG workflow, public IT troubleshooting sample documents, metadata ingestion, a Streamlit UI, web-based document upload, one-click knowledge-base rebuild, chat history, adjustable top-k retrieval, metadata filters, and source/reference snippet display.

It is intended for GitHub and resume presentation. It does not include public deployment, real enterprise documents, multi-user accounts, cloud storage, OCR, complex agent workflows, reranking, or production monitoring.

## Future Extensions

Possible next steps include:

- Add screenshots after local UI testing
- Add a small test document set for reproducible demos
- Improve error messages for missing API keys or failed PDF parsing
- Add lightweight unit tests for document loading and source formatting
- Add hybrid retrieval with BM25 + vector search
- Add retrieval trace for interview-friendly explanation
- Add query rewrite for vague IT operations questions
- Add a safer upload overwrite confirmation flow
