from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.prompting import build_prompt, build_training_text


def test_build_prompt_includes_context_and_retrieved_sections():
    prompt = build_prompt(
        context=["p1", "p2"],
        retrieved=["old clue"],
        include_retrieval=True,
    )
    assert "[CONTEXT]" in prompt
    assert "[RETRIEVED]" in prompt
    assert "old clue" in prompt


def test_build_training_text_places_retrieval_before_target():
    text = build_training_text(["p1"], ["clue"], "target")
    assert text.index("[RETRIEVED]") < text.index("[TARGET]")


def test_build_training_text_puts_target_section_at_end():
    text = build_training_text(["p1"], ["clue"], "target")
    assert "[TARGET]" in text
    assert text.endswith("[TARGET]\ntarget")


def test_build_prompt_without_retrieval_stays_clear():
    prompt = build_prompt(
        context=["p1", "p2"],
        retrieved=["old clue"],
        include_retrieval=False,
    )
    assert "[CONTEXT]" in prompt
    assert "[RETRIEVED]" not in prompt
    assert prompt == "[CONTEXT]\np1\np2"


def test_build_prompt_suppresses_empty_retrieval_section():
    prompt = build_prompt(
        context=["p1", "p2"],
        retrieved=[],
        include_retrieval=True,
    )
    assert "[CONTEXT]" in prompt
    assert "[RETRIEVED]" not in prompt
    assert prompt == "[CONTEXT]\np1\np2"


def test_build_prompt_supports_structured_context_format():
    prompt = build_prompt(
        context=["Holmes lit his pipe.", "Watson watched in silence."],
        include_retrieval=False,
        context_format="structured",
    )

    assert "[CONTEXT]" in prompt
    assert "[PARAGRAPH 1]" in prompt
    assert "[PARAGRAPH 2]" in prompt
    assert "Holmes lit his pipe." in prompt
    assert "Watson watched in silence." in prompt
