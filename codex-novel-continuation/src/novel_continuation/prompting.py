"""Prompt builders for proposal-aligned continuation experiments."""

from __future__ import annotations


def build_prompt(
    context: list[str],
    retrieved: list[str] | None = None,
    include_retrieval: bool = False,
    context_format: str = "plain",
) -> str:
    if context_format not in {"plain", "structured"}:
        raise ValueError(f"Unsupported context_format: {context_format}")

    sections = ["[CONTEXT]"]
    if context_format == "plain":
        sections.extend(context)
    else:
        for index, paragraph in enumerate(context, start=1):
            sections.extend([f"[PARAGRAPH {index}]", paragraph])

    retrieval_items = retrieved or []
    if include_retrieval and retrieval_items:
        sections.extend(["", "[RETRIEVED]"])
        if context_format == "plain":
            sections.extend(retrieval_items)
        else:
            for index, paragraph in enumerate(retrieval_items, start=1):
                sections.extend([f"[EVIDENCE {index}]", paragraph])
    return "\n".join(sections).strip()


def build_target_section(target: str) -> str:
    return f"\n\n[TARGET]\n{target}".strip("\n")


def build_training_text(
    context: list[str],
    retrieved: list[str] | None,
    target: str,
    include_retrieval: bool = True,
    context_format: str = "plain",
) -> str:
    prompt = build_prompt(
        context,
        retrieved=retrieved,
        include_retrieval=include_retrieval,
        context_format=context_format,
    )
    return f"{prompt}\n\n{build_target_section(target)}".strip()
