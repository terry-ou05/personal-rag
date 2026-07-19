from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .common import RetrievalResult, metadata_matches_filter, result_from_document
from .tokenizer import tokenize


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build BM25 retriever from an empty corpus.")

        self.documents = documents
        self.tokenized_corpus = [tokenize(document.page_content) for document in documents]
        if not any(self.tokenized_corpus):
            raise ValueError("Cannot build BM25 retriever because all corpus tokens are empty.")
        self.token_sets = [set(tokens) for tokens in self.tokenized_corpus]
        self.index = BM25Okapi(self.tokenized_corpus)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("Cannot run BM25 retrieval for an empty query.")

        candidate_indexes = [
            index
            for index, document in enumerate(self.documents)
            if metadata_matches_filter(document.metadata, metadata_filter)
        ]
        if not candidate_indexes:
            return []

        scores = self.index.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        scored_indexes = [
            index
            for index in candidate_indexes
            if query_token_set.intersection(self.token_sets[index])
        ]
        if not scored_indexes:
            return []

        ranked_indexes = sorted(
            scored_indexes,
            key=lambda index: (
                -float(scores[index]),
                str(self.documents[index].metadata.get("chunk_id", "")),
            ),
        )

        results = []
        for rank, index in enumerate(ranked_indexes[:top_k], start=1):
            results.append(
                result_from_document(
                    self.documents[index],
                    bm25_score=float(scores[index]),
                    bm25_rank=rank,
                )
            )
        return results
