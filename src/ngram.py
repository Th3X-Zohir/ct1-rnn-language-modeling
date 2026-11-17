"""Non-neural baseline: interpolated Witten-Bell character n-gram language model.

P_k(c | h) = (C(h c) + T(h) * P_{k-1}(c | h')) / (C(h) + T(h))
where T(h) = number of distinct characters seen after context h, and h' drops
the oldest character of h. The order-0 model is add-one smoothed unigram.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class WittenBellLM:
    def __init__(self, order: int):
        self.order = order

    def fit(self, text: str):
        self.V = len(set(text))
        self.uni = Counter(text); self.N = len(text)
        self.joint = [None] + [Counter() for _ in range(self.order)]   # joint[k][h+c], |h| = k
        self.ctx = [None] + [Counter() for _ in range(self.order)]     # ctx[k][h]
        types = [None] + [defaultdict(set) for _ in range(self.order)]
        for k in range(1, self.order + 1):
            J = self.joint[k]
            for i in range(k, len(text)):
                J[text[i - k:i + 1]] += 1
            for g, n in J.items():
                self.ctx[k][g[:-1]] += n
                types[k][g[:-1]].add(g[-1])
        self.types = [None] + [{h: len(s) for h, s in types[k].items()} for k in range(1, self.order + 1)]
        return self

    def prob(self, h: str, c: str) -> float:
        p = (self.uni[c] + 1) / (self.N + self.V)
        for k in range(1, min(self.order, len(h)) + 1):
            hk = h[-k:]
            tot = self.ctx[k].get(hk, 0)
            if tot == 0:
                break
            t = self.types[k][hk]
            p = (self.joint[k].get(hk + c, 0) + t * p) / (tot + t)
        return p

    def nll(self, text: str) -> float:
        """Mean negative log-likelihood in nats/char."""
        tot = 0.0
        for i in range(1, len(text)):
            tot -= math.log(self.prob(text[max(0, i - self.order):i], text[i]))
        return tot / (len(text) - 1)
