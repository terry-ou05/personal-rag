# Interview Notes

## 1-Minute Project Introduction

IT Ops Knowledge Base RAG Assistant is a local RAG question-answering system I built for simulated IT operations troubleshooting documents. It uses LangChain, DeepSeek, Chroma, BGE Embedding, and Streamlit. The app can load TXT, Markdown, and PDF files, attach metadata, build a local vector knowledge base, and answer questions based on retrieved document chunks.

The project also supports web-based document upload, one-click knowledge-base rebuild, metadata filtering, chat history, adjustable top-k retrieval, source/reference snippet display, retrieval baseline evaluation, optional Hybrid Retrieval, and optional Cross-Encoder reranking. I used Codex to help with coding, debugging, documentation, verification commands, and Git branch-based iteration.

## 3-Minute Project Introduction

This project is a local IT operations knowledge-base RAG assistant. The goal is to simulate an enterprise service desk knowledge-base scenario with public sample troubleshooting documents. Instead of asking a model from memory, the system retrieves relevant chunks from local runbooks and then asks DeepSeek to answer based on that retrieved context.

The backend workflow is handled mainly by `src/ingest.py`. It reads files from `data/raw/`, supports TXT, Markdown, and PDF, merges metadata from `data/metadata.json`, splits text into chunks, generates embeddings using `BAAI/bge-small-zh-v1.5`, and stores vectors plus metadata in a local Chroma database.

The web interface is built with Streamlit in `src/app.py`. It lets users upload documents, rebuild the knowledge base, select metadata filters, ask questions, keep chat history, adjust retriever top-k, and inspect sources and reference snippets. There is also a command-line entry point in `src/ask.py`.

In V7, I added a retrieval evaluation baseline. It uses 30 manually checked questions from the public IT ops documents and measures Recall@1, Recall@3, Recall@5, MRR, zero-hit rate, and retrieval latency. The current dense Chroma retriever reaches Recall@1 86.67%, Recall@3 100%, Recall@5 100%, and MRR 0.9278.

In V8, I added BM25 sparse retrieval and Reciprocal Rank Fusion. The result was not blindly positive: Hybrid kept Recall@1 at 86.67% but reduced MRR to 0.9122 and introduced one Top-1 regression, so I kept Dense as the default and exposed Hybrid as an optional mode.

In V9, I added a Cross-Encoder reranker experiment using `BAAI/bge-reranker-base`. Dense + Reranker improved Recall@1 to 96.67% and MRR to 0.9833 with no Top-1 regression, but it added CPU latency and model load time.

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

### What are Recall@K and MRR?

Recall@K checks whether the expected source document appears within the top K retrieved chunks. MRR, or Mean Reciprocal Rank, rewards the expected source appearing earlier in the ranked results. These metrics let me evaluate retrieval quality before answer generation.

### Why build a baseline before Hybrid Retrieval?

A baseline gives a measurable comparison point. Before adding BM25, RRF, reranking, or query rewrite, I want to know how well the existing Chroma dense retriever performs. Then V8 improvements can be compared with real numbers.

### What is the difference between Dense retrieval and BM25?

Dense retrieval uses embeddings to match semantic meaning. BM25 is keyword-based retrieval, so it is useful for exact commands, error codes, service names, file names, and configuration keys. In this project, BM25 helped a `df -h` disk-space question, but it was weaker on short Chinese symptom queries.

### Why not add Dense score and BM25 score directly?

The scores come from different systems and do not share the same scale. V8 uses Reciprocal Rank Fusion instead, which combines rankings rather than raw scores.

### What is RRF?

RRF stands for Reciprocal Rank Fusion. It gives each retrieved chunk a score based on its rank in each retriever. A chunk that appears high in both Dense and BM25 results gets a stronger fused score.

### Did Hybrid Retrieval improve the project?

It improved one known Dense weakness, `disk-002`, by moving the expected disk document from rank 2 to rank 1. But it also caused one Top-1 regression and lowered MRR. So the honest conclusion is to keep Dense as the default and leave Hybrid as an optional experiment.

### What is a Cross-Encoder reranker?

A Cross-Encoder scores a query and a candidate chunk together. It is slower than Dense retrieval, but it can make a more direct relevance judgment. In this project, it is used only after Dense or Hybrid retrieves Top-5 candidates.

### Why not run the Cross-Encoder on every document?

Because Cross-Encoder scoring is much more expensive. The scalable pattern is candidate retrieval first, then reranking a small candidate set. This keeps the system understandable and avoids turning every query into a full-corpus model scoring task.

### Did V9 improve the results?

Yes, on this small evaluation set. Dense + Reranker improved Recall@1 from 86.67% to 96.67% and MRR from 0.9278 to 0.9833. It improved `disk-002`, `disk-003`, and `login-005`, while `nginx-005` remained not Top-1. The tradeoff is latency: model loading took about 8.20 seconds in the final cached run and warm reranking averaged about 749 ms on CPU. I made Dense + Reranker the quality-focused default, while keeping Dense available for faster use.

### Why not implement Query Rewrite now?

V9 is focused on verifying whether reranking improves candidate ordering. Query Rewrite changes the input query and can affect recall behavior, so it should be evaluated separately after the reranker baseline is clear.

### What is metadata filtering?

Metadata filtering narrows retrieval by fields such as category, system, severity, and document type. In this project, it makes the search closer to an enterprise knowledge-base workflow where documents are organized by operational context.

### Why show sources and reference snippets?

Sources and snippets help verify the answer. They show which file and text chunk supported the response, which reduces the risk of blindly trusting generated text.

### How was Codex used in this project?

Codex was used to read the project structure, implement scoped features, refactor carefully, write documentation, run verification commands, inspect Git status, and help maintain a branch-based workflow.

### What did you personally take responsibility for?

I defined the project goal, selected the feature direction, reviewed each change, enforced safety rules, verified outputs, and guided the project from a basic demo toward a GitHub-ready showcase project.

### What are the project's limitations?

It is still a local showcase project. It does not include public deployment, user accounts, cloud storage, OCR, Qdrant, automated CI, or production monitoring. Uploaded private files also need to be handled carefully and should not be committed to Git.

### If you continue improving it, what would you do next?

I would use the V9 report as the control group, then consider Conditional Query Rewrite for vague questions that are still not fixed by reranking. I would keep it as a separate experiment because query rewrite changes the retrieval input, not just the ranking order.

### Why did you choose Streamlit?

Streamlit is simple for building a local AI demo UI quickly. It lets me focus on RAG workflow, document upload, retrieval controls, and source display without spending too much time on frontend engineering.

### Why not build a complex agent system?

The goal of this project is to demonstrate a clear local RAG workflow. A complex agent system would add unnecessary scope. For an internship showcase, a stable and understandable RAG project is more appropriate.
