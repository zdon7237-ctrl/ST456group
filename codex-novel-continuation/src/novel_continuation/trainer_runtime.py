"""Runtime utilities for proposal-aligned continuation training."""

from __future__ import annotations

from novel_continuation.prompting import build_prompt, build_target_section


def truncate_text_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)


def build_prompt_with_budget(
    tokenizer,
    *,
    context: list[str],
    retrieved: list[str],
    include_retrieval: bool,
    context_format: str,
    max_prompt_tokens: int,
) -> tuple[str, list[int]]:
    effective_context = list(context)
    effective_retrieved = list(retrieved)

    prompt_text = build_prompt(
        context=effective_context,
        retrieved=effective_retrieved,
        include_retrieval=include_retrieval,
        context_format=context_format,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    while len(prompt_ids) > max_prompt_tokens and len(effective_context) > 1:
        effective_context = effective_context[1:]
        prompt_text = build_prompt(
            context=effective_context,
            retrieved=effective_retrieved,
            include_retrieval=include_retrieval,
            context_format=context_format,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    while len(prompt_ids) > max_prompt_tokens and effective_retrieved:
        effective_retrieved = effective_retrieved[:-1]
        prompt_text = build_prompt(
            context=effective_context,
            retrieved=effective_retrieved,
            include_retrieval=include_retrieval,
            context_format=context_format,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
        prompt_text = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False)

    return prompt_text, prompt_ids


def _pad_sequence_batch(sequences: list[list[int]], pad_value: int) -> list[list[int]]:
    max_length = max((len(sequence) for sequence in sequences), default=0)
    return [sequence + ([pad_value] * (max_length - len(sequence))) for sequence in sequences]


class ContinuationDataCollator:
    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        max_target_tokens: int,
        aux_objective: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_tokens = max_target_tokens
        self.aux_objective = aux_objective
        self.pad_token_id = tokenizer.pad_token_id
        self.delimiter_ids = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

    def _encode_target_section(self, target: str) -> list[int]:
        truncated_target = truncate_text_to_tokens(
            self.tokenizer,
            target,
            self.max_target_tokens,
        )
        target_section = build_target_section(truncated_target)
        return self.tokenizer(target_section, add_special_tokens=False)["input_ids"]

    def _encode_feature(self, feature: dict) -> tuple[list[int], list[int], str, list[int]]:
        target_section_ids = self._encode_target_section(feature["target"])
        max_prompt_tokens = max(0, self.max_length - len(self.delimiter_ids) - len(target_section_ids))
        prompt_text, prompt_ids = build_prompt_with_budget(
            self.tokenizer,
            context=feature["context"],
            retrieved=feature.get("retrieved", []),
            include_retrieval=feature.get("include_retrieval", False),
            context_format=feature.get("context_format", "plain"),
            max_prompt_tokens=max_prompt_tokens,
        )
        full_input_ids = prompt_ids + self.delimiter_ids + target_section_ids
        labels = list(full_input_ids)
        return full_input_ids, labels, prompt_text, prompt_ids

    def _encode_candidate(self, prompt_ids: list[int], candidate_target: str) -> tuple[list[int], list[int]]:
        target_section_ids = self._encode_target_section(candidate_target)
        input_ids = prompt_ids + self.delimiter_ids + target_section_ids
        labels = ([-100] * (len(prompt_ids) + len(self.delimiter_ids))) + target_section_ids
        return input_ids, labels

    def __call__(self, features: list[dict]):
        import torch

        input_ids_batch: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels_batch: list[list[int]] = []
        candidate_input_ids_batch: list[list[list[int]]] = []
        candidate_attention_masks_batch: list[list[list[int]]] = []
        candidate_labels_batch: list[list[list[int]]] = []
        candidate_class_labels_batch: list[list[int]] = []

        for feature in features:
            input_ids, labels, _prompt_text, prompt_ids = self._encode_feature(feature)
            input_ids_batch.append(input_ids)
            attention_masks.append([1] * len(input_ids))
            labels_batch.append(labels)

            candidate_targets = feature.get("candidate_targets")
            if self.aux_objective != "none" and candidate_targets:
                candidate_input_ids: list[list[int]] = []
                candidate_attention_masks: list[list[int]] = []
                candidate_labels: list[list[int]] = []
                for candidate_target in candidate_targets:
                    candidate_input, candidate_label = self._encode_candidate(prompt_ids, candidate_target)
                    candidate_input_ids.append(candidate_input)
                    candidate_attention_masks.append([1] * len(candidate_input))
                    candidate_labels.append(candidate_label)
                candidate_input_ids_batch.append(candidate_input_ids)
                candidate_attention_masks_batch.append(candidate_attention_masks)
                candidate_labels_batch.append(candidate_labels)
                candidate_class_labels_batch.append(feature["candidate_labels"])

        batch = {
            "input_ids": torch.tensor(_pad_sequence_batch(input_ids_batch, self.pad_token_id), dtype=torch.long),
            "attention_mask": torch.tensor(_pad_sequence_batch(attention_masks, 0), dtype=torch.long),
            "labels": torch.tensor(_pad_sequence_batch(labels_batch, -100), dtype=torch.long),
        }

        if candidate_input_ids_batch:
            max_candidates = max(len(candidate_group) for candidate_group in candidate_input_ids_batch)
            flat_input_ids: list[list[int]] = []
            flat_attention_masks: list[list[int]] = []
            flat_labels: list[list[int]] = []
            flat_class_labels: list[int] = []
            flat_candidate_masks: list[int] = []

            for candidate_inputs, candidate_masks, candidate_labels, class_labels in zip(
                candidate_input_ids_batch,
                candidate_attention_masks_batch,
                candidate_labels_batch,
                candidate_class_labels_batch,
            ):
                padded_count = max_candidates - len(candidate_inputs)
                candidate_inputs = list(candidate_inputs)
                candidate_masks = list(candidate_masks)
                candidate_labels = list(candidate_labels)
                class_labels = list(class_labels)
                for _ in range(padded_count):
                    candidate_inputs.append([])
                    candidate_masks.append([])
                    candidate_labels.append([])
                    class_labels.append(0)

                for candidate_input, candidate_mask, candidate_label, class_label in zip(
                    candidate_inputs,
                    candidate_masks,
                    candidate_labels,
                    class_labels,
                ):
                    flat_input_ids.append(candidate_input)
                    flat_attention_masks.append(candidate_mask)
                    flat_labels.append(candidate_label)
                    flat_class_labels.append(class_label)
                    flat_candidate_masks.append(1 if candidate_input else 0)

            padded_input_ids = _pad_sequence_batch(flat_input_ids, self.pad_token_id)
            padded_attention_masks = _pad_sequence_batch(flat_attention_masks, 0)
            padded_labels = _pad_sequence_batch(flat_labels, -100)
            batch_size = len(features)
            candidate_sequence_length = len(padded_input_ids[0]) if padded_input_ids else 0
            batch["candidate_input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long).view(
                batch_size,
                max_candidates,
                candidate_sequence_length,
            )
            batch["candidate_attention_mask"] = torch.tensor(padded_attention_masks, dtype=torch.long).view(
                batch_size,
                max_candidates,
                candidate_sequence_length,
            )
            batch["candidate_labels"] = torch.tensor(padded_labels, dtype=torch.long).view(
                batch_size,
                max_candidates,
                candidate_sequence_length,
            )
            batch["candidate_class_labels"] = torch.tensor(flat_class_labels, dtype=torch.long).view(
                batch_size,
                max_candidates,
            )
            batch["candidate_mask"] = torch.tensor(flat_candidate_masks, dtype=torch.bool).view(
                batch_size,
                max_candidates,
            )

        return batch


def compute_candidate_scores(logits, labels):
    import torch.nn.functional as F

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shifted_labels.size())

    token_mask = shifted_labels.ne(-100)
    token_counts = token_mask.sum(dim=-1).clamp(min=1)
    normalised_loss = (losses * token_mask).sum(dim=-1) / token_counts
    return -normalised_loss
