from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .bm25_retriever import BM25Retriever
from .common import RetrievalResult, result_from_document
from .rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        *,
        vectorstore: Chroma,
        bm25_documents: list[Document],
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever(bm25_documents)
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

    def retrieve_dense(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if not query or not query.strip():
            raise ValueError("Cannot run dense retrieval for an empty query.")

        search_kwargs = {"k": top_k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )
        return [
            result_from_document(
                document,
                retrieval_mode="dense",
                candidate_rank=rank,
                dense_score=float(score),
                dense_rank=rank,
            )
            for rank, (document, score) in enumerate(docs_and_scores, start=1)
        ]

    def retrieve_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        return self.bm25.retrieve(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "dense",
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        mode = mode.lower()
        if mode == "dense":
            return self.retrieve_dense(query, top_k=top_k, metadata_filter=metadata_filter)
        if mode == "bm25":
            return self.retrieve_bm25(query, top_k=top_k, metadata_filter=metadata_filter)
        if mode == "hybrid":
            dense_results = self.retrieve_dense(
                query,
                top_k=max(top_k, self.dense_top_k),
                metadata_filter=metadata_filter,
            )
            bm25_results = self.retrieve_bm25(
                query,
                top_k=max(top_k, self.bm25_top_k),
                metadata_filter=metadata_filter,
            )
            return reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                final_top_k=top_k,
                rrf_k=self.rrf_k,
            )
        raise ValueError(f"Unsupported retrieval mode: {mode}")
