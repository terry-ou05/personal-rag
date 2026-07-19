from __future__ import annotations

import logging
import re

import jieba


jieba.setLogLevel(logging.WARNING)

TECH_TOKEN_PATTERN = re.compile(
    r"""
    (?:/[a-zA-Z0-9._/-]+)
    |(?:\b\d{1,3}(?:\.\d{1,3}){3}\b)
    |(?:\b\d+(?:\.\d+)?%?)
    |(?:-[a-zA-Z0-9]+)
    |(?:\b[a-zA-Z0-9_]+(?:[._-][a-zA-Z0-9_]+)+\b)
    |(?:\b[a-zA-Z]+\b)
    """,
    re.VERBOSE,
)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    lowered = text.lower()
    tokens: list[str] = []

    tokens.extend(match.group(0) for match in TECH_TOKEN_PATTERN.finditer(lowered))

    for token in jieba.cut(lowered, cut_all=False):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if CJK_PATTERN.search(normalized):
            tokens.append(normalized)

    return tokens
