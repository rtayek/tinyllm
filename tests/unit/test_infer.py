import pytest

from llm.infer import parse_args


def test_parse_args_accepts_prompt_and_token_count() -> None:
    args = parse_args(["--prompt", "Mr. Sherlock Holmes", "--tokens", "25"])

    assert args.prompt == "Mr. Sherlock Holmes"
    assert args.tokens == 25


def test_parse_args_preserves_existing_defaults() -> None:
    args = parse_args([])

    assert args.prompt == ""
    assert args.tokens == 400


def test_parse_args_rejects_negative_token_count() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--tokens", "-1"])
