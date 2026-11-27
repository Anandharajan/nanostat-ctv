"""Deterministic parsing of the claim direction and pairing cues from question text."""

from __future__ import annotations

import re

from nanostat.schemas import Alternative

_GREATER = (
    r"\bincreas\w*",
    r"\bhigher\b",
    r"\bgreater\b",
    r"\bimprov\w*",
    r"\braise[sd]?\b",
    r"\bboost\w*",
    r"\bmore than\b",
    r"\bexceed\w*",
    r"\bpositive(ly)?\b",
    r"\blarger\b",
)
_LESS = (
    r"\bdecreas\w*",
    r"\blower\w*",
    r"\breduc\w*",
    r"\bless than\b",
    r"\bfewer\b",
    r"\bnegative(ly)?\b",
    r"\bsmaller\b",
    r"\bworsen\w*",
    r"\bdrop\w*",
    r"\bshorter\b",
)
_PAIRING = (
    r"same (participants|subjects|patients|students|people|units|individuals)",
    r"\bbefore and after\b",
    r"\bmeasured twice\b",
    r"\brepeated(ly)? measure\w*",
    r"\bpre[- ]?(test|intervention)? and post",
    r"\bmatched pairs?\b",
    r"\beach (participant|subject|patient)\b.*\bboth\b",
)


_IDENT = re.compile(r"[a-z0-9]*_[a-z0-9_]*")


def infer_alternative(question: str) -> Alternative:
    # snake_case identifiers (variable names such as `dropped_out`) are not claim words
    q = _IDENT.sub(" ", question.lower())
    g = any(re.search(p, q) for p in _GREATER)
    l_ = any(re.search(p, q) for p in _LESS)
    if g and not l_:
        return Alternative.greater
    if l_ and not g:
        return Alternative.less
    return Alternative.two_sided


def mentions_pairing(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _PAIRING)
