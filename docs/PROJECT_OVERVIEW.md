# Project Overview

## Project Background

AI Learning Knowledge Assistant is a local RAG question-answering project for organizing AI learning notes, internship preparation materials, and small project documentation. It is built around a practical student workflow: put learning materials into a local folder, build a searchable knowledge base, and ask questions through a simple web interface.

This project is not positioned as a commercial product. It is a learning and showcase project that demonstrates basic RAG application development, AI coding with Codex, debugging, documentation, and Git branch-based iteration.

## Why I Built This Project

As an AI student preparing for internships, I need a clear way to review scattered learning materials such as RAG notes, LangChain notes, project notes, and interview preparation content. A local knowledge-base assistant is a good small project because it combines LLM application development, embeddings, vector databases, Python engineering, and a usable web UI.

The project also gives me a concrete way to practice using Codex for AI coding tasks, including reading an existing codebase, making scoped changes, running verification commands, documenting decisions, and managing Git branches.

## Problem It Solves

The project helps answer questions from local documents instead of relying only on general model knowledge. This is useful for:

- Reviewing AI learning notes before interviews
- Searching project notes quickly
- Preparing internship-related answers
- Turning scattered TXT, Markdown, and PDF files into a searchable local knowledge base
- Showing retrieved sources so answers can be checked against the original material

## Core Features

- Load TXT, Markdown, and PDF files from `data/raw/`
- Split documents into text chunks
- Generate semantic embeddings with `BAAI/bge-small-zh-v1.5`
- Store vectors locally with Chroma
- Ask questions through a Streamlit web interface
- Upload documents from the web UI
- Rebuild the knowledge base with one click
- Keep chat history in the current Streamlit session
- Adjust retriever top-k from the sidebar
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

The current version is a local runnable showcase version. It includes the full basic RAG workflow, a Streamlit UI, web-based document upload, one-click knowledge-base rebuild, chat history, adjustable top-k retrieval, and source/reference snippet display.

It is intended for GitHub and resume presentation. It does not include public deployment, multi-user accounts, cloud storage, OCR, complex agent workflows, or production monitoring.

## Future Extensions

Possible next steps include:

- Add screenshots after local UI testing
- Add a small test document set for reproducible demos
- Improve error messages for missing API keys or failed PDF parsing
- Add lightweight unit tests for document loading and source formatting
- Add conversation export for interview review
- Add a safer upload overwrite confirmation flow
- Prepare a deployment plan later, with careful API key and vector database handling
