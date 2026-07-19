# Interview Notes

## 1-Minute Project Introduction

IT Ops Knowledge Base RAG Assistant is a local RAG question-answering system I built for simulated IT operations troubleshooting documents. It uses LangChain, DeepSeek, Chroma, BGE Embedding, and Streamlit. The app can load TXT, Markdown, and PDF files, attach metadata, build a local vector knowledge base, and answer questions based on retrieved document chunks.

The project also supports web-based document upload, one-click knowledge-base rebuild, metadata filtering, chat history, adjustable top-k retrieval, and source/reference snippet display. I used Codex to help with coding, debugging, documentation, verification commands, and Git branch-based iteration.

## 3-Minute Project Introduction

This project is a local IT operations knowledge-base RAG assistant. The goal is to simulate an enterprise service desk knowledge-base scenario with public sample troubleshooting documents. Instead of asking a model from memory, the system retrieves relevant chunks from local runbooks and then asks DeepSeek to answer based on that retrieved context.

The backend workflow is handled mainly by `src/ingest.py`. It reads files from `data/raw/`, supports TXT, Markdown, and PDF, merges metadata from `data/metadata.json`, splits text into chunks, generates embeddings using `BAAI/bge-small-zh-v1.5`, and stores vectors plus metadata in a local Chroma database.

The web interface is built with Streamlit in `src/app.py`. It lets users upload documents, rebuild the knowledge base, select metadata filters, ask questions, keep chat history, adjust retriever top-k, and inspect sources and reference snippets. There is also a command-line entry point in `src/ask.py`.

For development, I used Codex as an AI coding assistant. I asked it to inspect the codebase, make scoped changes, run verification commands, improve documentation, and manage Git branches. This helped me practice a more structured AI coding workflow rather than only writing small isolated scripts.

## Possible Interview Questions

### What problem does this project solve?

It helps simulate an IT service desk knowledge-base search workflow. Instead of manually searching troubleshooting runbooks, the user can ask questions and get answers grounded in local documents, with sources shown for verification.

### Why use RAG?

RAG makes the answer depend on retrieved documents, not only the model's general knowledge. This is useful when the information is local, project-specific, or needs to be checked against original notes.

### What does LangChain do in this project?

LangChain helps connect the RAG components: prompts, retriever, vector store, document chains, and the DeepSeek chat model. It provides a structured way to build the retrieval and answer-generation pipeline.

### What is Chroma?

Chroma is the local vector database used to store document embeddings. After documents are converted into vectors, Chroma lets the retriever search for chunks that are semantically similar to the user's question.

### What is Embedding?

Embedding converts text into numerical vectors. Text with similar meaning should have similar vector representations, which makes semantic search possible.

### What is top-k?

Top-k is the number of retrieved chunks returned by the retriever. A smaller top-k gives more focused context, while a larger top-k provides more information but may include less relevant chunks.

### What is metadata filtering?

Metadata filtering narrows retrieval by fields such as category, system, severity, and document type. In this project, it makes the search closer to an enterprise knowledge-base workflow where documents are organized by operational context.

### Why show sources and reference snippets?

Sources and snippets help verify the answer. They show which file and text chunk supported the response, which reduces the risk of blindly trusting generated text.

### How was Codex used in this project?

Codex was used to read the project structure, implement scoped features, refactor carefully, write documentation, run verification commands, inspect Git status, and help maintain a branch-based workflow.

### What did you personally take responsibility for?

I defined the project goal, selected the feature direction, reviewed each change, enforced safety rules, verified outputs, and guided the project from a basic demo toward a GitHub-ready showcase project.

### What are the project's limitations?

It is still a local showcase project. It does not include public deployment, user accounts, cloud storage, OCR, BM25 hybrid retrieval, reranking, automated tests, or production monitoring. Uploaded private files also need to be handled carefully and should not be committed to Git.

### If you continue improving it, what would you do next?

I would add screenshots, improve error handling, write lightweight tests for document loading and source formatting, add retrieval trace, and then consider hybrid retrieval with BM25 plus vector search.

### Why did you choose Streamlit?

Streamlit is simple for building a local AI demo UI quickly. It lets me focus on RAG workflow, document upload, retrieval controls, and source display without spending too much time on frontend engineering.

### Why not build a complex agent system?

The goal of this project is to demonstrate a clear local RAG workflow. A complex agent system would add unnecessary scope. For an internship showcase, a stable and understandable RAG project is more appropriate.
