from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from function_calling_ft.loss_mask import (
    IGNORE_INDEX,
    build_expected_loss_mask_for_record,
)


class ToolCallChatDataset:
    """Local AutoModel-compatible JSONL chat dataset with explicit label policy."""

    def __init__(
        self,
        path_or_dataset_id: str | Sequence[str],
        tokenizer: Any,
        *,
        split: str | None = None,
        name: str | None = None,
        seq_length: int | None = None,
        padding: str | bool = "do_not_pad",
        truncation: str | bool = "do_not_truncate",
        start_of_turn_token: str | None = None,
        chat_template: str | None = None,
        loss_mask_policy: str = "assistant_only",
        enable_thinking: bool = False,
    ) -> None:
        del split, name, start_of_turn_token, chat_template
        if loss_mask_policy not in {"assistant_only", "full_sequence"}:
            raise ValueError(
                "loss_mask_policy must be 'assistant_only' or 'full_sequence'",
            )
        if padding not in {"do_not_pad", False}:
            raise ValueError("ToolCallChatDataset only supports do_not_pad padding")
        if truncation not in {"do_not_truncate", False}:
            raise ValueError(
                "ToolCallChatDataset refuses truncation to protect gold tool calls",
            )
        self.paths = self._normalize_paths(path_or_dataset_id)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.loss_mask_policy = loss_mask_policy
        self.enable_thinking = enable_thinking
        self.records = self._load_records(self.paths)

    @staticmethod
    def _normalize_paths(path_or_dataset_id: str | Sequence[str]) -> tuple[Path, ...]:
        if isinstance(path_or_dataset_id, str):
            return (Path(path_or_dataset_id),)
        return tuple(Path(path) for path in path_or_dataset_id)

    @staticmethod
    def _load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    loaded = json.loads(stripped)
                    if not isinstance(loaded, dict):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    records.append(loaded)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        result = build_expected_loss_mask_for_record(
            self.tokenizer,
            record,
            enable_thinking=self.enable_thinking,
        )
        input_ids = list(result.input_ids)
        if self.seq_length is not None and len(input_ids) > self.seq_length:
            record_id = record.get("id", record.get("example_id", index))
            raise ValueError(
                f"record {record_id!r} renders to {len(input_ids)} tokens, "
                f"exceeding seq_length={self.seq_length}",
            )
        if self.loss_mask_policy == "assistant_only":
            labels = list(result.labels)
        else:
            labels = [
                token_id if token_id != result.pad_token_id else IGNORE_INDEX
                for token_id in input_ids
            ]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }
