"""Prompt builders for baseline and retrieval-augmented continuation."""

from __future__ import annotations


def build_prompt(context: list[str], retrieved: list[str] | None = None, include_retrieval: bool = False) -> str:
    sections = ["[CONTEXT]", *context]
    retrieval_items = retrieved or []
    if include_retrieval and retrieval_items:
        sections.extend(["", "[RETRIEVED]"])
        sections.extend(retrieval_items)
    return "\n".join(sections).strip()


def build_training_text(
    context: list[str],
    retrieved: list[str] | None,
    target: str,
    include_retrieval: bool = True,
) -> str:
    prompt = build_prompt(context, retrieved=retrieved, include_retrieval=include_retrieval)
    return f"{prompt}\n\n[TARGET]\n{target}".strip()
