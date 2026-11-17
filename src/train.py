"""Training and evaluation loops.

Metrics
-------
* cross-entropy loss in nats/char
* bits-per-character  BPC = loss / ln 2
* character perplexity PPL = exp(loss)

Evaluation is *stateful*: the held-out text is cut into ``n_streams`` contiguous
streams that are read left-to-right in chunks of 256 characters while the hidden
state is carried over. The score therefore does not depend on the training
sequence length, which makes the sequence-length experiment a fair comparison.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from .data import random_batch
from .models import CharRNN, detach_state, n_params

LN2 = math.log(2)


@dataclass
class Config:
    name: str
    cell: str = "lstm"
    emb_dim: int = 64
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    seq_len: int = 128
    chars_per_batch: int = 16384   # batch = chars_per_batch // seq_len (equal compute per step)
    steps: int = 3000
    lr: float = 2e-3
    weight_decay: float = 0.0
    clip: float = 1.0
    eval_every: int = 250
    seed: int = 42
    extra: dict = field(default_factory=dict)

    @property
    def batch(self) -> int:
        return max(1, self.chars_per_batch // self.seq_len)


@torch.no_grad()
def evaluate(model, data: torch.Tensor, n_streams: int = 32, chunk: int = 256,
             max_chars: int | None = None) -> float:
    """Return mean NLL (nats/char) of ``data`` under ``model`` (stateful)."""
    model.eval()
    if max_chars:
        data = data[:max_chars]
    L = (len(data) - 1) // n_streams
    x_all = data[: L * n_streams].view(n_streams, L)
    y_all = data[1: L * n_streams + 1].view(n_streams, L)
    state, tot, cnt = None, 0.0, 0
    for i in range(0, L, chunk):
        x, y = x_all[:, i:i + chunk], y_all[:, i:i + chunk]
        logits, state = model(x, state)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        cnt += y.numel()
    model.train()
    return tot / cnt


def train(cfg: Config, train_data, val_data, vocab_size: int, log=print, val_chars=150_000):
    """Train one configuration; returns (model, record-dict)."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = CharRNN(vocab_size, cfg.cell, cfg.emb_dim, cfg.hidden, cfg.layers, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    hist = {"step": [], "train_loss": [], "grad_norm": [], "eval_step": [], "val_loss": [], "time": []}
    best, best_state, t0, run_loss, run_gn = float("inf"), None, time.time(), 0.0, 0.0
    for step in range(1, cfg.steps + 1):
        x, y = random_batch(train_data, cfg.batch, cfg.seq_len, gen)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip).item()
        opt.step(); sched.step()
        run_loss += loss.item(); run_gn += gn
        if step % 50 == 0:
            hist["step"].append(step); hist["train_loss"].append(run_loss / 50)
            hist["grad_norm"].append(run_gn / 50); run_loss = run_gn = 0.0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            vl = evaluate(model, val_data, max_chars=val_chars)
            hist["eval_step"].append(step); hist["val_loss"].append(vl); hist["time"].append(time.time() - t0)
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            tl = hist["train_loss"][-1] if hist["train_loss"] else loss.item()
            log(f"[{cfg.name}] step {step:5d}  train {tl:.4f}  "
                f"val {vl:.4f} ({vl / LN2:.3f} bpc)  {time.time() - t0:6.0f}s")
    model.load_state_dict(best_state)
    full_val = evaluate(model, val_data)
    rec = {
        "config": asdict(cfg) | {"batch": cfg.batch}, "params": n_params(model),
        "train_time_s": time.time() - t0, "history": hist,
        "best_val_loss": full_val, "best_val_bpc": full_val / LN2, "best_val_ppl": math.exp(full_val),
        "final_train_loss": hist["train_loss"][-1] if hist["train_loss"] else None,
    }
    return model, rec
