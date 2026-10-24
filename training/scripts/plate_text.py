"""Turns the plate reader's per-slot probabilities into text, using Ghana's plate formats.

The reader (fast-plate-ocr, 10 slots x 37 symbols: 0-9, A-Z, pad "_") predicts each slot on its
own, so it can say "AS4633I2" although a Ghanaian plate never has a letter in its year. Ghanaian
plates are two letters, 1-4 digits, then a two-digit year or one letter:

    GR 3966-21   GN 8654-23   CD 590 18   GR 672 H   WR 376-M   DV 334J   GT 9556 V

`decode` finds the most probable string of that shape. If the unconstrained reading is far more
probable than any Ghanaian one (a foreign plate), it keeps that instead. web/src/lib/plateText.ts
is the same algorithm for the app; keep them in sync.
"""

from __future__ import annotations

import math
import re

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD = ALPHABET.index("_")
DIGITS = list(range(10))
LETTERS = list(range(10, 36))

# How much less likely (in nats) the Ghanaian reading may be than the free one and still win.
# 6 nats ~ a factor of 400: enough to fix a 0/O or 1/I slip, not enough to bend a foreign plate.
MARGIN = 6.0


def _patterns(slots: int):
    """Character classes for every Ghanaian shape that fits in `slots`."""
    for n in range(1, 5):
        for suffix in (["D", "D"], ["L"]):
            p = ["L", "L"] + ["D"] * n + suffix
            if len(p) <= slots:
                yield p


def _best(probs, cls: str):
    ids = DIGITS if cls == "D" else LETTERS
    i = max(ids, key=lambda c: probs[c])
    return i, probs[i]


def decode(probs: list[list[float]], ghana: bool = True) -> dict:
    """probs: slots x 37 probabilities. Returns {text, plate, chars, confidence, format}."""
    log = lambda p: math.log(max(p, 1e-12))
    free = [max(range(len(ALPHABET)), key=lambda c: row[c]) for row in probs]
    free_score = sum(log(row[c]) for row, c in zip(probs, free))
    best = None
    if ghana:
        for pattern in _patterns(len(probs)):
            picks = [_best(probs[i], cls) for i, cls in enumerate(pattern)]
            score = sum(log(p) for _, p in picks) + sum(log(row[PAD]) for row in probs[len(pattern):])
            if best is None or score > best[0]:
                best = (score, pattern, [c for c, _ in picks])
    if best and best[0] >= free_score - MARGIN:
        _, pattern, ids = best
        chars = [(ALPHABET[c], probs[i][c]) for i, c in enumerate(ids)]
        fmt = "ghana"
    else:
        chars = [(ALPHABET[c], probs[i][c]) for i, c in enumerate(free) if c != PAD]
        pattern, fmt = None, "free"
    text = "".join(c for c, _ in chars)
    return {
        "text": text,
        "plate": display(text, pattern),
        "chars": chars,
        "confidence": math.prod(p for _, p in chars) if chars else 0.0,
        "format": fmt,
    }


def display(text: str, pattern: list[str] | None) -> str:
    """GR396621 -> "GR 3966-21"; GR672H -> "GR 672 H"; other plates as read."""
    if pattern is None:
        return text
    if pattern[-1] == "D":
        return f"{text[:2]} {text[2:-2]}-{text[-2:]}"
    return f"{text[:2]} {text[2:-1]} {text[-1]}"


def normalise(plate: str) -> str:
    """For comparing readings: letters and digits only."""
    return re.sub(r"[^A-Z0-9]", "", plate.upper())
