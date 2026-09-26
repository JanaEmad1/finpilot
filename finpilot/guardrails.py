"""Input guardrails: redact personal data before it reaches the LLM or the logs,
and flag obvious prompt-injection attempts.

These are simple, readable rules — a first layer, not a complete defence. The strongest
protection is architectural: the LLM cannot run SQL or take actions on its own
(see tools.py and agent.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS = [
    # card numbers: 13-19 digits, optionally split by spaces/dashes; checked with Luhn below
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # phones: international (+...), local starting with 0, or digits in groups. A bare run of
    # digits is NOT treated as a phone — the first version redacted order numbers too.
    ("PHONE", re.compile(r"(?<!\w)(?:\+\d[\d ()-]{7,}\d|0\d{9,10}|\d{3}[ -]\d{3,4}[ -]\d{3,4})\b")),
    ("CVV", re.compile(r"\b(?:cvv|cvc|security code)\D{0,5}\d{3,4}\b", re.I)),
]

_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)"
    r"|disregard (the |your )?(system|previous) (prompt|instructions)"
    r"|you are now|act as (an? )?(admin|developer|system)"
    r"|reveal (your|the) (system )?prompt|show (me )?(your|the) system prompt"
    r"|other (customer|user)'?s? (account|data|balance)",
    re.I,
)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


@dataclass
class GuardResult:
    text: str
    redacted: list[str] = field(default_factory=list)
    injection: bool = False


def check(message: str) -> GuardResult:
    text, found = message, []
    for kind, pattern in _PATTERNS:
        def _replace(m: re.Match) -> str:
            if kind == "CARD" and not _luhn_ok(re.sub(r"\D", "", m.group())):
                return m.group()  # a long number that is not a card (e.g. an order id)
            found.append(kind)
            return f"[{kind}]"
        text = pattern.sub(_replace, text)
    return GuardResult(text, found, bool(_INJECTION.search(message)))
