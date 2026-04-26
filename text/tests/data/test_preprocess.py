from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.preprocess import (
    filter_short_paragraphs,
    normalise_whitespace,
    split_paragraphs,
    strip_gutenberg_boilerplate,
)


def test_split_paragraphs_removes_empty_blocks():
    raw_text = "CHAPTER I\n\nFirst paragraph.\n\n\nSecond paragraph."
    result = split_paragraphs(raw_text)
    assert result == ["First paragraph.", "Second paragraph."]


def test_normalise_whitespace_collapses_extra_spaces():
    raw_text = " Holmes\t   spoke.  \r\nWatson   listened. "
    result = normalise_whitespace(raw_text)
    assert result == "Holmes spoke. \nWatson listened."


def test_filter_short_paragraphs_removes_short_entries():
    paragraphs = ["Tiny", "This paragraph is long enough to keep."]
    result = filter_short_paragraphs(paragraphs, min_chars=10)
    assert result == ["This paragraph is long enough to keep."]


def test_strip_gutenberg_boilerplate_removes_common_markers():
    raw_text = (
        "Intro line\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Story text.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Footer line\n"
    )
    result = strip_gutenberg_boilerplate(raw_text)
    assert result == "Story text."


def test_strip_gutenberg_boilerplate_returns_text_when_no_markers_exist():
    raw_text = "Plain story text without Gutenberg markers."
    result = strip_gutenberg_boilerplate(raw_text)
    assert result == raw_text


def test_strip_gutenberg_boilerplate_handles_marker_case_variants():
    raw_text = (
        "cover page\n"
        "*** start of this project gutenberg ebook sample ***\n"
        "Story text.\n"
        "*** end of this project gutenberg ebook sample ***\n"
        "legal footer\n"
    )
    result = strip_gutenberg_boilerplate(raw_text)
    assert result == "Story text."


def test_split_paragraphs_keeps_short_uppercase_story_text():
    raw_text = "HOLMES WAITED.\n\nWatson entered the room."
    result = split_paragraphs(raw_text)
    assert result == ["HOLMES WAITED.", "Watson entered the room."]


def test_split_paragraphs_filters_standalone_uppercase_headings():
    raw_text = "ADVENTURE I\n\nFirst paragraph.\n\nTHE CASE-BOOK\n\nSecond paragraph."
    result = split_paragraphs(raw_text)
    assert result == ["First paragraph.", "Second paragraph."]
