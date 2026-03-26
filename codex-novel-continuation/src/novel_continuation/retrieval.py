"""Simple retrieval helpers for continuation prompts."""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_retriever(candidates: list[str]) -> dict[str, object]:
    """Build a TF-IDF index over candidates already restricted to earlier passages."""
    if not candidates:
        raise ValueError("fit_tfidf_retriever requires at least one candidate")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(candidates)
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "candidates": candidates,
    }


def query_tfidf_index(
    index: dict[str, object],
    query: str,
    top_k: int = 1,
    candidate_limit: int | None = None,
) -> list[dict[str, object]]:
    """Query a pre-built TF-IDF index.

    The caller is responsible for ensuring ``index["candidates"]`` only contains
    earlier passages relative to the current continuation target.
    """
    if top_k <= 0:
        return []

    candidates = index.get("candidates", [])
    if not candidates:
        return []

    if candidate_limit is not None:
        candidate_limit = max(0, min(candidate_limit, len(candidates)))
        if candidate_limit == 0:
            return []
        active_candidates = candidates[:candidate_limit]
        active_matrix = index["matrix"][:candidate_limit]
    else:
        active_candidates = candidates
        active_matrix = index["matrix"]

    scores = (
        active_matrix @ index["vectorizer"].transform([query]).T
    ).toarray().ravel()
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [
        {
            "text": active_candidates[index],
            "candidate_index": int(index),
            "score": float(scores[index]),
        }
        for index in ranked_indices
    ]


def retrieve_top_k(query: str, candidates: list[str], top_k: int = 1) -> list[dict[str, object]]:
    """Return ranked retrieval hits.

    The caller is responsible for ensuring ``candidates`` only contains earlier
    passages relative to the current continuation target.
    """
    if not candidates:
        return []

    return query_tfidf_index(index=fit_tfidf_retriever(candidates), query=query, top_k=top_k)


@dataclass
class BM25Retriever:
    candidates: list[str]

    def __post_init__(self) -> None:
        tokenised = [candidate.lower().split() for candidate in self.candidates]
        self._retriever = BM25Okapi(tokenised)

    def retrieve(self, query: str, top_k: int = 1) -> list[dict[str, object]]:
        if not self.candidates or top_k <= 0:
            return []
        scores = self._retriever.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            {
                "text": self.candidates[index],
                "candidate_index": int(index),
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
