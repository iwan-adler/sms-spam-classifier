import pytest

from spam_classifier.data import Message, deduplicate, labels, parse_lines, split, template_key


def test_parse_lines():
    parsed = parse_lines(["ham\tSee you at 5", "", "spam\tWIN a prize!\tnow"])
    assert parsed == [Message("See you at 5", 0), Message("WIN a prize!\tnow", 1)]


@pytest.mark.parametrize("line", ["ham See you", "junk\ttext", "spam\t   ", "\tno label"])
def test_malformed_lines_raise(line):
    with pytest.raises(ValueError, match="Line 1"):
        parse_lines([line])


def test_deduplicate_keeps_first_occurrence():
    a, b = Message("hi", 0), Message("win", 1)
    assert deduplicate([a, b, a, b, a]) == [a, b]
    # Same text with a different label is not an exact duplicate.
    assert len(deduplicate([Message("hi", 0), Message("hi", 1)])) == 2


def test_template_key_ignores_numbers_case_and_spacing():
    assert template_key("Call 0906 1701461  NOW") == template_key("call 0800 999   now")
    assert template_key("call now") != template_key("call later")


def test_split_sizes_stratification_and_determinism(messages):
    parts = split(messages, seed=1)
    total = len(messages)
    assert len(parts.train) + len(parts.validation) + len(parts.test) == total
    assert 0.6 < len(parts.train) / total < 0.8
    assert 0.1 < len(parts.test) / total < 0.2
    overall = labels(messages).mean()
    for part in (parts.train, parts.validation, parts.test):
        assert abs(labels(part).mean() - overall) < 0.05
    assert split(messages, seed=1) == parts
    assert split(messages, seed=2) != parts


def test_split_never_puts_a_template_in_two_parts(messages):
    # Add near-duplicates that differ only in digits; they must stay together.
    variants = [Message(f"URGENT you won {n} cash call 0906{n}", 1) for n in range(30)]
    parts = split(messages + variants, seed=3)
    keys = [{template_key(m.text) for m in part} for part in (parts.train, parts.validation, parts.test)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    holders = [i for i, part in enumerate((parts.train, parts.validation, parts.test))
               if any(m in part for m in variants)]
    assert len(holders) == 1


def test_split_needs_enough_data():
    with pytest.raises(ValueError, match="at least"):
        split([Message("a", 0)] * 5)
