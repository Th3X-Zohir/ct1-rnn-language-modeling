"""Data pipeline: download Jane Austen novels from Project Gutenberg, clean them,
build a character vocabulary, split into train/val/test and serve batches.

Cleaning steps (documented in the report):
  1. Strip the Project Gutenberg licence header/footer (``*** START OF`` / ``*** END OF``).
  2. Normalise line endings and Unicode punctuation (curly quotes, dashes) to ASCII.
  3. Remove illustration / transcriber markup lines such as ``[Illustration: ...]``.
  4. Un-wrap hard line breaks inside paragraphs (Gutenberg wraps at ~70 chars),
     keeping blank lines as paragraph separators.
  5. Collapse repeated spaces and drop any character that still occurs < 20 times.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch

BOOKS = {  # Gutenberg id -> title
    1342: "Pride and Prejudice",
    158: "Emma",
    161: "Sense and Sensibility",
    141: "Mansfield Park",
    121: "Northanger Abbey",
    105: "Persuasion",
}
MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"

_UNICODE_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "—": "--", "–": "-", "…": "...", " ": " ",
    "﻿": "", "é": "e", "è": "e", "ê": "e", "à": "a",
    "â": "a", "ï": "i", "ô": "o", "ü": "u", "ç": "c",
}


def download(raw_dir: Path) -> dict[int, str]:
    """Download each book once (cached on disk) and return {id: raw_text}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for bid in BOOKS:
        p = raw_dir / f"pg{bid}.txt"
        if not p.exists():
            req = urllib.request.Request(MIRROR.format(id=bid), headers={"User-Agent": "Mozilla/5.0"})
            p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        out[bid] = p.read_text(encoding="utf-8", errors="replace")
    return out


def strip_gutenberg(text: str) -> str:
    """Keep only the body between the Gutenberg START and END markers."""
    s = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    e = re.search(r"\*\*\* ?END OF", text)
    return text[s.end() if s else 0: e.start() if e else len(text)]


def clean(text: str) -> str:
    """Apply cleaning steps 2-4 (see module docstring)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for k, v in _UNICODE_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.S)
    text = re.sub(r"_([^_]+)_", r"\1", text)            # _italic_ markup
    paras = re.split(r"\n\s*\n", text)
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paras]
    text = "\n\n".join(p for p in paras if p)
    return re.sub(r"[ \t]+", " ", text)


def drop_rare(text: str, min_count: int = 20) -> tuple[str, dict]:
    """Remove characters occurring fewer than ``min_count`` times (cleaning step 5)."""
    cnt = Counter(text)
    rare = {c for c, n in cnt.items() if n < min_count and c != "\x00"}  # \x00 = book separator
    if rare:
        text = "".join(c for c in text if c not in rare)
    return text, {c: cnt[c] for c in rare}


def build_corpus(root: Path, splits=(0.90, 0.05, 0.05)):
    """Full pipeline. Each book is split contiguously 90/5/5 so that every split
    contains every novel but no passage leaks between splits."""
    raw = download(root / "raw")
    cleaned = {b: clean(strip_gutenberg(t)) for b, t in raw.items()}
    joined, removed = drop_rare("\x00".join(cleaned.values()))
    cleaned = dict(zip(cleaned.keys(), joined.split("\x00")))
    parts = {"train": [], "val": [], "test": []}
    for t in cleaned.values():
        n = len(t); a = int(n * splits[0]); b = int(n * (splits[0] + splits[1]))
        parts["train"].append(t[:a]); parts["val"].append(t[a:b]); parts["test"].append(t[b:])
    texts = {k: "\n\n".join(v) for k, v in parts.items()}
    chars = sorted(set("".join(texts.values())))
    stoi = {c: i for i, c in enumerate(chars)}
    meta = {
        "books": {str(b): BOOKS[b] for b in BOOKS},
        "raw_chars": {str(b): len(t) for b, t in raw.items()},
        "clean_chars": {str(b): len(t) for b, t in cleaned.items()},
        "split_chars": {k: len(v) for k, v in texts.items()},
        "vocab": chars, "vocab_size": len(chars), "removed_rare": removed,
    }
    (root / "processed").mkdir(parents=True, exist_ok=True)
    for k, v in texts.items():
        (root / "processed" / f"{k}.txt").write_text(v, encoding="utf-8")
    (root / "processed" / "meta.json").write_text(json.dumps(meta, indent=2))
    return texts, stoi, meta, cleaned


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor(np.frombuffer(bytes(stoi[c] for c in text), dtype=np.uint8).astype(np.int64))


def random_batch(data: torch.Tensor, batch: int, seq_len: int, gen: torch.Generator):
    """Sample ``batch`` random windows of length ``seq_len`` (+1 for targets)."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y
