"""Versioned prompt text. Prompts are executable inputs: change them with their parser/schema."""

from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent


def read_prompt(name: str) -> str:
    return (DIRECTORY / name).read_text(encoding="utf-8")
