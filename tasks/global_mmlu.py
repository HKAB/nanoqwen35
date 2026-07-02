"""
MMLU-style multiple-choice task backed by a custom JSONL dataset.

Expected JSONL format (one sample per line):
{
  "question": "....",
  "choices": ["...", "...", "...", "..."],
  "answer": 0,              # or "A"/"B"/"C"/"D"
  "subject": "optional_tag"
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_LABELS = ("A", "B", "C", "D")


def _render_mc(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    options = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{options}\n\n"
        "Hãy chọn đáp án đúng. Chỉ trả lời bằng một chữ cái: A, B, C hoặc D."
    )


def _normalize_answer(answer: Any) -> int:
    if isinstance(answer, int):
        if 0 <= answer < len(_LABELS):
            return answer
        raise ValueError(f"answer index must be in [0, 3], got: {answer}")

    if isinstance(answer, str):
        letter = answer.strip().upper()
        if letter in _LABELS:
            return _LABELS.index(letter)
        raise ValueError(f"answer letter must be one of {_LABELS}, got: {answer!r}")

    raise ValueError(f"unsupported answer type: {type(answer).__name__}")


def _validate_row(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    if "query" not in row or "choices" not in row or "gold" not in row:
        raise ValueError(
            f"JSONL line {line_no}: required keys are query, choices, gold"
        )

    query = row["query"]
    choices = row["choices"]
    gold = row["gold"]
    subject = row.get("subject")

    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"JSONL line {line_no}: query must be a non-empty string")

    if not isinstance(choices, list) or len(choices) != 4 or not all(isinstance(c, str) for c in choices):
        raise ValueError(f"JSONL line {line_no}: choices must be a list of 4 strings")

    answer_idx = _normalize_answer(gold)

    return {
        "query": query,
        "choices": choices,
        "gold": answer_idx,
        "subject": subject,
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {i}: each line must be a JSON object")
            rows.append(_validate_row(row, i))

    if not rows:
        raise ValueError(f"No valid samples found in {path}")

    return rows


class GlobalMMLU:
    """
    Custom JSONL MMLU-style task compatible with categorical evaluation loops.
    """
    eval_type = "categorical"
    letters = _LABELS

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.ds = _load_jsonl(jsonl_path)
        if shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(self.ds)
        if limit is not None:
            self.ds = self.ds[:limit]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get_example(idx)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        answer_letter = self.letters[row["gold"]]
        user_message = _render_mc(row["query"], self.letters, row["choices"])
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_letter},
        ]
        return {
            "messages": messages,
            "subject": row.get("subject"),
            "letters": self.letters,
            "answer": answer_letter,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        assert assistant_response in self.letters, (
            f"assistant_response must be one of {self.letters}, got {assistant_response!r}"
        )
        return assistant_response == conversation["messages"][-1]["content"]
