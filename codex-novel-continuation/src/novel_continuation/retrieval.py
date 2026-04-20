"""Simple retrieval helpers for continuation prompts."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised indirectly in tests
    BM25Okapi = None


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
            "text": active_candidates[idx],
            "candidate_index": int(idx),
            "score": float(scores[idx]),
        }
        for idx in ranked_indices
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
        self._tokenised = tokenised
        self._retriever = BM25Okapi(tokenised) if BM25Okapi is not None else None

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
        query_set = set(query_tokens)
        scores: list[float] = []
        for candidate_tokens in self._tokenised:
            overlap = sum(1 for token in candidate_tokens if token in query_set)
            scores.append(float(overlap))
        return scores

    def retrieve(self, query: str, top_k: int = 1) -> list[dict[str, object]]:
        if not self.candidates or top_k <= 0:
            return []
        query_tokens = query.lower().split()
        if self._retriever is None:
            scores = self._fallback_scores(query_tokens)
        else:
            scores = self._retriever.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            {
                "text": self.candidates[index],
                "candidate_index": int(index),
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
