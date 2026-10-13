import random

import pytest

from spam_classifier.data import Message

SPAM_WORDS = ["win", "prize", "claim", "cash", "free", "urgent", "txt", "reward", "bonus", "offer", "winner", "call"]
HAM_WORDS = ["lunch", "home", "tonight", "dinner", "meeting", "class", "see", "later", "thanks", "mom", "bus", "call"]


def synthetic_messages(n: int = 400, spam_rate: float = 0.15, seed: int = 0) -> list[Message]:
    """Clearly separable fake SMS. Each message gets unique filler words so no two share a template."""
    rng = random.Random(seed)
    messages = []
    for i in range(n):
        spam = rng.random() < spam_rate
        words = rng.sample(SPAM_WORDS if spam else HAM_WORDS, 4) + [f"filler{i}a", f"filler{i}b"]
        rng.shuffle(words)
        messages.append(Message(" ".join(words), int(spam)))
    return messages


@pytest.fixture
def messages() -> list[Message]:
    return synthetic_messages()
