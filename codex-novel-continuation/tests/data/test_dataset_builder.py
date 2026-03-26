from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.dataset_builder import assign_split, build_examples, serialise_examples
import build_dataset as build_dataset_script


def test_build_examples_uses_only_previous_paragraphs():
    paragraphs = ["p0", "p1", "p2", "p3"]
    examples = build_examples(
        paragraphs,
        context_size=2,
        book_id="1661",
        source_title="The Adventures of Sherlock Holmes",
    )
    assert examples[0]["context"] == ["p0", "p1"]
    assert examples[0]["target"] == "p2"
    assert examples[0]["paragraph_index"] == 2
    assert examples[0]["book_id"] == "1661"
    assert examples[0]["source_title"] == "The Adventures of Sherlock Holmes"


def test_assign_split_uses_contiguous_blocks():
    labels = [assign_split(index, 10) for index in range(10)]
    assert labels == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "val",
        "test",
    ]


def test_assign_split_handles_small_corpora_without_randomness():
    labels = [assign_split(index, 3) for index in range(3)]
    assert labels == ["train", "val", "test"]


def test_serialise_examples_writes_jsonl(tmp_path):
    output_path = tmp_path / "examples.jsonl"
    examples = [
        {
            "context": ["p0", "p1"],
            "target": "p2",
            "paragraph_index": 2,
        }
    ]

    serialise_examples(examples, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == examples[0]


def test_build_dataset_adds_split_and_source_metadata_across_books(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    (raw_dir / "1661.txt").write_text("p0 enough text\n\np1 enough text\n\np2 enough text", encoding="utf-8")
    (raw_dir / "834.txt").write_text("q0 enough text\n\nq1 enough text\n\nq2 enough text", encoding="utf-8")

    original_sources = build_dataset_script.SHERLOCK_SOURCES
    build_dataset_script.SHERLOCK_SOURCES = [
        {"book_id": "1661", "title": "Adventures", "url": "https://example.com/1661.txt"},
        {"book_id": "834", "title": "Memoirs", "url": "https://example.com/834.txt"},
    ]
    try:
        split_examples = build_dataset_script.build_dataset(raw_dir, output_dir, context_size=1, min_chars=1)
    finally:
        build_dataset_script.SHERLOCK_SOURCES = original_sources

    all_examples = split_examples["train"] + split_examples["val"] + split_examples["test"]
    assert all_examples
    assert all("source_title" in example for example in all_examples)
    assert all("split" in example for example in all_examples)
    assert {example["source_title"] for example in all_examples} == {"Adventures", "Memoirs"}
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def test_build_dataset_missing_raw_file_has_actionable_error(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    original_sources = build_dataset_script.SHERLOCK_SOURCES
    build_dataset_script.SHERLOCK_SOURCES = [
        {"book_id": "9999", "title": "Missing Book", "url": "https://example.com/9999.txt"},
    ]
    try:
        try:
            build_dataset_script.build_dataset(raw_dir, output_dir, context_size=1, min_chars=1)
        except FileNotFoundError as error:
            message = str(error)
        else:
            raise AssertionError("Expected FileNotFoundError")
    finally:
        build_dataset_script.SHERLOCK_SOURCES = original_sources

    assert "9999" in message
    assert "Missing Book" in message
    assert "download_gutenberg.py" in message
