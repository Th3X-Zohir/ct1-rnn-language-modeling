"""Text generation with different sampling strategies and simple quality metrics.

Strategies: greedy, temperature sampling, top-k sampling, nucleus (top-p) sampling.
Metrics on generated text (no human labels needed):
  * valid_word_rate   – share of generated words that exist in the training vocabulary
  * distinct_1/2      – unique word unigrams / bigrams over total (diversity)
  * repeat_4gram_rate – share of word 4-grams that are repeats (degeneration)
  * self_nll          – NLL of the sample under the model (how "confident" it is)
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

WORD = re.compile(r"[A-Za-z']+")


@torch.no_grad()
def sample(model, stoi, itos, prompt: str, n: int = 600, strategy: str = "temperature",
           temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([[stoi[c] for c in prompt]])
    logits, state = model(x)
    out, nll = [], 0.0
    last = logits[0, -1]
    for _ in range(n):
        logp_full = F.log_softmax(last, -1)
        if strategy == "greedy":
            idx = int(last.argmax())
        else:
            lg = last / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(lg, top_k).values[-1]
                lg = torch.where(lg < kth, torch.full_like(lg, -1e9), lg)
            if top_p < 1.0:
                sp, si = torch.sort(F.softmax(lg, -1), descending=True)
                cut = torch.cumsum(sp, 0) - sp > top_p
                sp[cut] = 0
                probs = torch.zeros_like(sp).scatter(0, si, sp)
            else:
                probs = F.softmax(lg, -1)
            idx = int(torch.multinomial(probs / probs.sum(), 1, generator=g))
        nll -= float(logp_full[idx])
        out.append(itos[idx])
        logits, state = model(torch.tensor([[idx]]), state)
        last = logits[0, -1]
    return prompt + "".join(out), nll / n


def text_metrics(text: str, vocab_words: set[str]) -> dict:
    w = [t.lower() for t in WORD.findall(text)]
    if not w:
        return {"valid_word_rate": 0, "distinct_1": 0, "distinct_2": 0, "repeat_4gram_rate": 0, "n_words": 0}
    bi = list(zip(w, w[1:])); q = list(zip(w, w[1:], w[2:], w[3:]))
    return {
        "valid_word_rate": sum(x in vocab_words for x in w) / len(w),
        "distinct_1": len(set(w)) / len(w),
        "distinct_2": len(set(bi)) / max(1, len(bi)),
        "repeat_4gram_rate": 1 - len(set(q)) / max(1, len(q)),
        "n_words": len(w),
    }
