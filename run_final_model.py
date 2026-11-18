"""E6 — final model: the best configuration of the controlled study (2-layer GRU, 256 units,
sequence length 128) trained 3.2x longer (8,000 steps instead of 2,500).

E1–E5 compare architectures under an *equal* budget; E6 answers a different question:
how good does the winning architecture get when it is simply trained longer?
Results are written to results/final/ so that they do not mix with the controlled comparison.

Usage:  nice -n 19 python run_final_model.py --threads 2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.data import build_corpus, encode
from src.train import LN2, Config, evaluate, train

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "final"
STEPS = 8_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)   # small GRUs train fastest on 2 threads (see benchmark in report)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    texts, stoi, meta, _ = build_corpus(ROOT / "data")
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "val", "test"))
    cfg = Config("E6_gru_long", cell="gru", steps=a.steps, eval_every=500, chars_per_batch=8192)
    logf = open(OUT / "E6_gru_long.log", "w")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    model, rec = train(cfg, tr, va, meta["vocab_size"], log=log, val_chars=100_000)
    t = time.time(); tl = evaluate(model, te)
    rec |= {"test_loss": tl, "test_bpc": tl / LN2, "test_ppl": math.exp(tl), "test_eval_s": time.time() - t}
    torch.save({"state_dict": model.state_dict(), "config": rec["config"], "stoi": stoi},
               ROOT / "checkpoints" / "E6_gru_long.pt")
    (OUT / "E6_gru_long.json").write_text(json.dumps(rec, indent=1))
    log(f"[E6_gru_long] DONE  val {rec['best_val_bpc']:.3f} bpc  test {rec['test_bpc']:.3f} bpc  "
        f"{rec['train_time_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
