"""Run every experiment of the CT1 Task-3 study and write results to ./results.

E1  Cell type        : vanilla RNN vs GRU vs LSTM (hidden 256, 2 layers, seq 128)
E2  Sequence length  : LSTM trained with BPTT length 32 / 64 / 128 / 256
E3  Capacity / depth : LSTM hidden 128 / 256 / 384 and layers 1 / 2 / 3
E4  Baseline         : Witten-Bell character n-gram (order 3 / 5 / 7)
E5  Sampling         : greedy, temperature 0.5 / 0.8 / 1.0 / 1.3, top-k 5, nucleus p=0.9

The script is resumable: finished runs (results/runs/<name>.json) are skipped.
Usage:  nice -n 19 python run_experiments.py --workers 3 --threads 4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import get_context
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"; RUNS = RES / "runs"; CKPT = ROOT / "checkpoints"
STEPS = 2500
BASE = dict(chars_per_batch=8192, steps=STEPS)


def job_list():
    from src.train import Config
    J = [
        Config("E1_rnn", cell="rnn", **BASE), Config("E1_gru", cell="gru", **BASE),
        Config("E1_lstm", cell="lstm", **BASE),
        Config("E2_lstm_seq32", seq_len=32, **BASE), Config("E2_lstm_seq64", seq_len=64, **BASE),
        Config("E2_lstm_seq256", seq_len=256, **BASE),
        Config("E3_lstm_h128", hidden=128, **BASE), Config("E3_lstm_h384", hidden=384, **BASE),
        Config("E3_lstm_L1", layers=1, **BASE), Config("E3_lstm_L3", layers=3, **BASE),
    ]
    return J


def _worker(args):
    cfg, threads = args
    torch.set_num_threads(threads)
    from src.data import build_corpus, encode
    texts, stoi, meta, _ = build_corpus(ROOT / "data")
    tr, va, te = (encode(texts[k], stoi) for k in ("train", "val", "test"))
    from src.train import LN2, evaluate, train
    logf = open(RES / "logs" / f"{cfg.name}.log", "w")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()
    model, rec = train(cfg, tr, va, meta["vocab_size"], log=log, val_chars=100_000)
    t = time.time(); tl = evaluate(model, te)
    rec |= {"test_loss": tl, "test_bpc": tl / LN2, "test_ppl": math.exp(tl),
            "test_eval_s": time.time() - t}
    torch.save({"state_dict": model.state_dict(), "config": rec["config"], "stoi": stoi}, CKPT / f"{cfg.name}.pt")
    (RUNS / f"{cfg.name}.json").write_text(json.dumps(rec, indent=1))
    log(f"[{cfg.name}] DONE  val {rec['best_val_bpc']:.3f} bpc  test {rec['test_bpc']:.3f} bpc  "
        f"{rec['train_time_s'] / 60:.1f} min")
    return cfg.name


def run_ngrams():
    out = RES / "ngram_baseline.json"
    if out.exists():
        return
    from src.data import build_corpus
    from src.ngram import WittenBellLM
    texts, *_ = build_corpus(ROOT / "data")
    rows = []
    for order in (3, 5, 7):
        t = time.time(); lm = WittenBellLM(order).fit(texts["train"]); fit = time.time() - t
        v, te = lm.nll(texts["val"]), lm.nll(texts["test"])
        rows.append({"model": f"Witten-Bell {order}-gram", "order": order, "val_bpc": v / math.log(2),
                     "test_bpc": te / math.log(2), "test_ppl": math.exp(te), "fit_s": fit,
                     "n_contexts": sum(len(c) for c in lm.ctx[1:])})
        print(rows[-1], flush=True)
    out.write_text(json.dumps(rows, indent=1))


def run_sampling(best: str, threads: int):
    torch.set_num_threads(threads)
    from src.data import build_corpus
    from src.generate import WORD, sample, text_metrics
    from src.models import CharRNN
    texts, stoi, meta, _ = build_corpus(ROOT / "data")
    itos = {i: c for c, i in stoi.items()}
    vocab_words = {w.lower() for w in WORD.findall(texts["train"])}
    ck = torch.load(CKPT / f"{best}.pt"); c = ck["config"]
    model = CharRNN(meta["vocab_size"], c["cell"], c["emb_dim"], c["hidden"], c["layers"], c["dropout"])
    model.load_state_dict(ck["state_dict"])
    strategies = [("greedy", dict(strategy="greedy")),
                  ("temp_0.5", dict(temperature=0.5)), ("temp_0.8", dict(temperature=0.8)),
                  ("temp_1.0", dict(temperature=1.0)), ("temp_1.3", dict(temperature=1.3)),
                  ("top_k_5", dict(top_k=5)), ("nucleus_0.9", dict(top_p=0.9))]
    prompts = ["It is a truth universally acknowledged", "Mr. Darcy", "\"My dear Emma,\" said she",
               "The weather was"]
    # reference: real held-out text metrics
    ref = text_metrics(texts["test"][:20000], vocab_words)
    rows, examples = [], {}
    for name, kw in strategies:
        agg = []
        for pi, p in enumerate(prompts):
            for seed in range(3):
                txt, nll = sample(model, stoi, itos, p, n=800, seed=seed, **kw)
                m = text_metrics(txt[len(p):], vocab_words) | {"self_nll_bpc": nll / math.log(2)}
                agg.append(m)
                if seed == 0 and pi == 0:
                    examples[name] = txt
        row = {"strategy": name} | {k: sum(a[k] for a in agg) / len(agg) for k in agg[0]}
        rows.append(row); print(row, flush=True)
    (RES / "sampling.json").write_text(json.dumps({"model": best, "reference_test_text": ref,
                                                   "rows": rows, "examples": examples}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    for d in (RES, RUNS, CKPT, RES / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    from src.data import build_corpus
    build_corpus(ROOT / "data")          # download once before forking
    todo = [c for c in job_list() if not (RUNS / f"{c.name}.json").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        ng = pool.apply_async(run_ngrams)
        for name in pool.imap_unordered(_worker, [(c, a.threads) for c in todo]):
            print("finished", name, flush=True)
        ng.get()
    recs = {p.stem: json.loads(p.read_text()) for p in RUNS.glob("*.json")}
    best = min(recs, key=lambda k: recs[k]["best_val_loss"])
    print("best model by validation:", best, flush=True)
    if not (RES / "sampling.json").exists():
        run_sampling(best, a.threads)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
