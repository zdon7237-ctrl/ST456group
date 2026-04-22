from pathlib import Path
import sys
from importlib.util import module_from_spec, spec_from_file_location
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.retrieval import BM25Retriever, fit_tfidf_retriever, query_tfidf_index, retrieve_top_k


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_retrieval_index.py"
SPEC = spec_from_file_location("build_retrieval_index", SCRIPT_PATH)
BUILD_RETRIEVAL_INDEX = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILD_RETRIEVAL_INDEX)


def test_fit_tfidf_retriever_keeps_candidate_order():
    candidates = ["first clue", "second clue"]
    index = fit_tfidf_retriever(candidates)
    assert index["candidates"] == candidates


def test_fit_tfidf_retriever_rejects_empty_candidates():
    try:
        fit_tfidf_retriever([])
    except ValueError as exc:
        assert "at least one candidate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty candidates")


def test_retrieve_top_k_returns_ranked_results_with_metadata():
    corpus = ["holmes lit the pipe", "watson entered the room", "the fog was dense"]
    query = "watson in the room"
    result = retrieve_top_k(query=query, candidates=corpus[:2], top_k=1)
    assert result[0]["text"] == "watson entered the room"
    assert result[0]["candidate_index"] == 1


def test_retrieve_top_k_respects_top_k():
    corpus = [
        "holmes lit the pipe",
        "watson entered the room",
        "watson quietly entered baker street",
    ]
    result = retrieve_top_k(query="watson entered", candidates=corpus, top_k=2)
    assert len(result) == 2


def test_retrieve_top_k_returns_empty_list_for_empty_candidates():
    assert retrieve_top_k(query="anything", candidates=[], top_k=3) == []


def test_query_tfidf_index_reuses_prebuilt_index():
    candidates = ["holmes lit the pipe", "watson entered the room"]
    index = fit_tfidf_retriever(candidates)
    result = query_tfidf_index(index=index, query="watson room", top_k=1)
    assert result[0]["text"] == "watson entered the room"
    assert result[0]["candidate_index"] == 1


def test_query_tfidf_index_returns_empty_list_for_non_positive_top_k():
    index = fit_tfidf_retriever(["holmes lit the pipe"])
    assert query_tfidf_index(index=index, query="holmes", top_k=0) == []


def test_query_tfidf_index_truncates_to_available_prefix():
    index = fit_tfidf_retriever(
        [
            "train clue",
            "first eval target",
            "second eval target",
        ]
    )

    result = query_tfidf_index(
        index=index,
        query="eval target",
        top_k=5,
        candidate_limit=2,
    )

    assert len(result) == 2
    assert {item["text"] for item in result} == {"train clue", "first eval target"}


def test_bm25_retriever_returns_ranked_results_with_metadata():
    retriever = BM25Retriever(["holmes lit the pipe", "watson entered the room"])
    result = retriever.retrieve(query="watson room", top_k=1)
    assert result[0]["text"] == "watson entered the room"
    assert result[0]["candidate_index"] == 1


def test_build_retrieval_index_rejects_empty_dataset(tmp_path):
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "index.pkl"

    try:
        BUILD_RETRIEVAL_INDEX.build_retrieval_index(dataset_path, output_path)
    except ValueError as exc:
        assert "No retrieval candidates" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty retrieval dataset")


def test_build_retrieval_index_persists_candidate_records(tmp_path):
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        '{"target":"holmes lit the pipe","book_id":"1661","source_title":"Adventures","paragraph_index":3,"split":"train"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "index.pkl"

    BUILD_RETRIEVAL_INDEX.build_retrieval_index(dataset_path, output_path)

    with output_path.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved["candidate_records"][0]["source_title"] == "Adventures"
