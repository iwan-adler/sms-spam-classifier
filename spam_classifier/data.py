"""Load the UCI SMS Spam Collection and split it without leaking near-duplicates."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ("ham", "spam")


@dataclass(frozen=True)
class Message:
    text: str
    label: int  # 1 = spam, 0 = ham


@dataclass(frozen=True)
class Split:
    train: list[Message]
    validation: list[Message]
    test: list[Message]


def parse_lines(lines: list[str]) -> list[Message]:
    """Parse 'label<TAB>text' lines. Malformed lines raise instead of being silently skipped."""
    messages = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        label, sep, text = line.partition("\t")
        if not sep or label not in LABELS or not text.strip():
            raise ValueError(f"Line {number} is not 'ham|spam<TAB>text': {line[:60]!r}")
        messages.append(Message(text=text, label=int(label == "spam")))
    return messages


def load_zip(path: Path) -> list[Message]:
    with zipfile.ZipFile(path) as archive:
        text = archive.read("SMSSpamCollection").decode("utf-8")
    return parse_lines(text.splitlines())


def deduplicate(messages: list[Message]) -> list[Message]:
    """Drop exact repeats (403 in the raw data), keeping the first occurrence.

    Without this, identical messages land in both training and test data and
    inflate the reported scores.
    """
    seen: set[tuple[str, int]] = set()
    unique = []
    for message in messages:
        key = (message.text, message.label)
        if key not in seen:
            seen.add(key)
            unique.append(message)
    return unique


def template_key(text: str) -> str:
    """Normalise case, whitespace and digits so template spam that only differs
    in a phone number or prize amount maps to the same key."""
    return re.sub(r"\d+", "0", re.sub(r"\s+", " ", text.lower())).strip()


def split(messages: list[Message], seed: int = 42, n_folds: int = 20) -> Split:
    """About 70% train / 15% validation / 15% test, stratified by label and
    grouped by template, so near-duplicates never straddle two splits."""
    if len(messages) < n_folds:
        raise ValueError(f"Need at least {n_folds} messages to split, got {len(messages)}")
    labels = np.array([m.label for m in messages])
    groups = np.array([template_key(m.text) for m in messages])
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(messages), dtype=int)
    for fold, (_, indices) in enumerate(folds.split(np.zeros(len(messages)), labels, groups)):
        fold_of[indices] = fold

    test_folds = round(n_folds * 0.15)
    validation_folds = round(n_folds * 0.15)

    def pick(condition) -> list[Message]:
        return [m for m, keep in zip(messages, condition) if keep]

    return Split(
        train=pick(fold_of >= test_folds + validation_folds),
        validation=pick((fold_of >= test_folds) & (fold_of < test_folds + validation_folds)),
        test=pick(fold_of < test_folds),
    )


def texts(messages: list[Message]) -> list[str]:
    return [m.text for m in messages]


def labels(messages: list[Message]) -> np.ndarray:
    return np.array([m.label for m in messages])
