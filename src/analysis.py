"""Exploratory data analysis and result plotting.

Every function saves a PNG to ``results/figures`` and/or a CSV to ``results/tables``
and returns the matplotlib figure or DataFrame so that the notebook can display it.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import sys

import matplotlib
if "ipykernel" not in sys.modules:      # headless scripts; notebooks keep their inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; FIG = RES / "figures"; TAB = RES / "tables"
for _d in (FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

PALETTE = {"rnn": "#d1495b", "gru": "#edae49", "lstm": "#00798c", "ngram": "#7a7a7a", "accent": "#30638e"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "axes.titleweight": "bold"})
LN2 = math.log(2)
WORD = re.compile(r"[A-Za-z']+")
SENT = re.compile(r"[^.!?]+[.!?]")


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", bbox_inches="tight"); return fig


def _tab(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TAB / f"{name}.csv", index=False); return df


# ------------------------------------------------------------------ data analysis
def corpus_table(meta: dict, cleaned: dict) -> pd.DataFrame:
    rows = []
    for bid, title in meta["books"].items():
        t = cleaned[int(bid)]
        words = WORD.findall(t)
        rows.append({"Book": title, "Gutenberg ID": int(bid), "Raw chars": meta["raw_chars"][bid],
                     "Clean chars": len(t), "Removed (%)": round(100 * (1 - len(t) / meta["raw_chars"][bid]), 1),
                     "Words": len(words), "Unique words": len({w.lower() for w in words}),
                     "Paragraphs": t.count("\n\n") + 1})
    df = pd.DataFrame(rows)
    tot = {"Book": "TOTAL", "Gutenberg ID": "", **{c: df[c].sum() for c in df.columns[2:] if c != "Removed (%)"}}
    tot["Removed (%)"] = round(100 * (1 - tot["Clean chars"] / tot["Raw chars"]), 1)
    tot["Unique words"] = len({w.lower() for t in cleaned.values() for w in WORD.findall(t)})
    return _tab(pd.concat([df, pd.DataFrame([tot])], ignore_index=True), "corpus_statistics")


def split_table(meta: dict) -> pd.DataFrame:
    s = meta["split_chars"]; n = sum(s.values())
    return _tab(pd.DataFrame([{"Split": k, "Characters": v, "Share (%)": round(100 * v / n, 1)} for k, v in s.items()]),
                "split_statistics")


def fig_book_sizes(df: pd.DataFrame):
    d = df[df.Book != "TOTAL"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(d.Book, d["Clean chars"] / 1e3, color=PALETTE["accent"])
    for y, v in enumerate(d["Clean chars"] / 1e3):
        ax.text(v + 8, y, f"{v:,.0f}k", va="center", fontsize=9)
    ax.set_xlabel("Characters after cleaning (thousands)"); ax.set_title("Corpus size per novel")
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    return _save(fig, "eda_book_sizes")


def fig_char_freq(train: str):
    cnt = Counter(train); n = len(train)
    items = cnt.most_common()
    labels = [repr(c)[1:-1] if c not in " \n" else {" ": "space", "\n": "\\n"}[c] for c, _ in items]
    p = np.array([v / n for _, v in items])
    H = -(p * np.log2(p)).sum()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 1.2]})
    ax[0].bar(range(40), p[:40] * 100, color=PALETTE["accent"])
    ax[0].set_xticks(range(40)); ax[0].set_xticklabels(labels[:40], fontsize=7, rotation=90)
    ax[0].set_ylabel("Frequency (%)"); ax[0].set_title("40 most frequent characters (train split)")
    ax[1].loglog(range(1, len(p) + 1), p, "o-", ms=3, color=PALETTE["lstm"])
    ax[1].set_xlabel("Rank"); ax[1].set_ylabel("Probability"); ax[1].set_title("Rank-frequency (log-log)")
    ax[1].text(0.05, 0.08, f"Unigram entropy\n= {H:.3f} bits/char", transform=ax[1].transAxes, fontsize=9)
    _save(fig, "eda_char_frequency")
    cls = Counter("letter" if c.isalpha() else "space/newline" if c in " \n" else "digit" if c.isdigit() else "punctuation"
                  for c in train)
    return fig, H, {k: round(100 * v / n, 2) for k, v in cls.items()}


def fig_lengths(train: str):
    words = WORD.findall(train)
    wl = np.array([len(w) for w in words])
    sl = np.array([len(WORD.findall(s)) for s in SENT.findall(train)])
    pl = np.array([len(p) for p in train.split("\n\n")])
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax[0].hist(wl, bins=range(1, 18), color=PALETTE["accent"], edgecolor="white")
    ax[0].set_title("Word length"); ax[0].set_xlabel("characters")
    ax[1].hist(sl[sl < 100], bins=50, color=PALETTE["lstm"], edgecolor="white")
    ax[1].set_title("Sentence length"); ax[1].set_xlabel("words")
    ax[2].hist(np.clip(pl, 0, 3000), bins=60, color=PALETTE["gru"], edgecolor="white")
    ax[2].set_title("Paragraph length"); ax[2].set_xlabel("characters (clipped at 3000)")
    for a, v, u in zip(ax, (wl, sl, pl), ("chars", "words", "chars")):
        a.axvline(np.median(v), color="k", ls="--", lw=1)
        a.text(0.97, 0.92, f"median {np.median(v):.0f} {u}\nmean {v.mean():.1f}", transform=a.transAxes,
               ha="right", va="top", fontsize=8.5)
    _save(fig, "eda_length_distributions")
    return fig, {"word_len_mean": wl.mean(), "sent_len_median": float(np.median(sl)),
                 "sent_len_mean": sl.mean(), "para_len_median": float(np.median(pl)),
                 "para_len_p95": float(np.percentile(pl, 95)), "n_sentences": len(sl)}


def fig_top_words(train: str, k: int = 25):
    cnt = Counter(w.lower() for w in WORD.findall(train))
    names = ["elizabeth", "darcy", "emma", "elinor", "marianne", "fanny", "anne", "catherine", "knightley",
             "edmund", "wentworth", "bennet", "jane", "harriet", "elton", "weston", "crawford", "bingley"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    top = cnt.most_common(k)
    ax[0].barh([w for w, _ in top][::-1], [v for _, v in top][::-1], color=PALETTE["accent"])
    ax[0].set_title(f"Top {k} words"); ax[0].set_xlabel("count")
    nm = sorted(((n, cnt[n]) for n in names), key=lambda x: -x[1])[:15]
    ax[1].barh([n.title() for n, _ in nm][::-1], [v for _, v in nm][::-1], color=PALETTE["lstm"])
    ax[1].set_title("Most frequent character names"); ax[1].set_xlabel("count")
    # Zipf / Heaps statistics
    _save(fig, "eda_top_words")
    hapax = sum(1 for v in cnt.values() if v == 1)
    return fig, {"vocab_words": len(cnt), "hapax": hapax, "hapax_pct": 100 * hapax / len(cnt),
                 "top_word": top[0]}


def fig_ngram_coverage(train: str, val: str):
    """How many validation character n-grams were never seen in training -> sparsity argument for RNNs."""
    rows = []
    for n in range(1, 11):
        tr = {train[i:i + n] for i in range(0, len(train) - n)}
        v = [val[i:i + n] for i in range(0, len(val) - n, 7)]
        rows.append({"n": n, "unseen_val_ngrams_pct": 100 * sum(g not in tr for g in v) / len(v),
                     "distinct_train_ngrams": len(tr)})
    df = _tab(pd.DataFrame(rows), "ngram_sparsity")
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.plot(df.n, df.unseen_val_ngrams_pct, "o-", color=PALETTE["rnn"])
    ax.set_xlabel("n (characters)"); ax.set_ylabel("% of validation n-grams unseen in train")
    ax.set_title("Data sparsity of character n-grams"); ax.set_xticks(df.n)
    return _save(fig, "eda_ngram_sparsity"), df


# ------------------------------------------------------------------ diagrams
def _box(ax, x, y, w, h, text, fc, fs=8.5, tc="black"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#333333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def _arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))


def fig_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    c1, c2, c3, c4 = "#dbe9f6", "#d7efe6", "#fbe8c8", "#f6d5d9"
    top = [(0.1, "Project Gutenberg\n6 Austen novels\n(4.18 M chars)", c1),
           (2.3, "Cleaning\nstrip licence, ASCII-fold,\nun-wrap, drop rare chars", c1),
           (4.5, "Contiguous split\nper novel\n90 / 5 / 5 %", c1),
           (6.7, "Char vocabulary\n75 symbols\ninteger encoding", c1),
           (8.9, "EDA\nstatistics, entropy,\nn-gram sparsity", c1)]
    for x, t, c in top:
        _box(ax, x, 3.2, 2.0, 1.2, t, c)
    for i in range(4):
        _arrow(ax, (top[i][0] + 2.0, 3.8), (top[i + 1][0], 3.8))
    mid = [(0.1, "Random windows\nseq_len + 1 chars\n8 192 chars / batch", c2),
           (2.3, "CharRNN\nEmbedding → RNN / GRU /\nLSTM × L → Linear", c2),
           (4.5, "Training\nAdamW, warm-up + cosine,\nclip 1.0, dropout 0.2", c2),
           (6.7, "Model selection\nbest validation BPC\n(stateful eval)", c2),
           (8.9, "Final test\nBPC, perplexity,\ntime, parameters", c2)]
    for x, t, c in mid:
        _box(ax, x, 1.65, 2.0, 1.2, t, c)
    for i in range(4):
        _arrow(ax, (mid[i][0] + 2.0, 2.25), (mid[i + 1][0], 2.25))
    _arrow(ax, (7.7, 3.2), (1.1, 2.85))
    bot = [(0.1, "E1 cell type\nRNN · GRU · LSTM", c3), (2.3, "E2 sequence length\n32 · 64 · 128 · 256", c3),
           (4.5, "E3 capacity\nhidden 128–384, layers 1–3", c3), (6.7, "E4 n-gram baseline\nWitten-Bell 3/5/7", c4),
           (8.9, "E5 sampling\ngreedy · T · top-k · top-p", c4)]
    for x, t, c in bot:
        _box(ax, x, 0.15, 2.0, 1.0, t, c)
    for x, _, _ in bot:
        _arrow(ax, (x + 1.0, 1.65), (x + 1.0, 1.15))
    return _save(fig, "workflow_diagram")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    items = [(0.1, "input characters\n$x_1 \\ldots x_T$\n(B × T integers)", "#eeeeee"), (2.05, "Embedding\n75 → 64", "#dbe9f6"),
             (4.0, "Recurrent stack\nRNN | GRU | LSTM\nhidden H, L layers\ndropout between layers", "#d7efe6"),
             (5.95, "Dropout 0.2\n+ Linear\nH → 75", "#fbe8c8"),
             (7.9, "softmax\n$P(x_{t+1} \\mid x_{\\leq t})$\ncross-entropy", "#f6d5d9")]
    for x, t, c in items:
        _box(ax, x, 0.5, 1.75, 1.6, t, c, fs=8.5)
    for i in range(4):
        _arrow(ax, (items[i][0] + 1.75, 1.3), (items[i + 1][0], 1.3))
    ax.annotate("", xy=(4.35, 2.12), xytext=(5.4, 2.12),
                arrowprops=dict(arrowstyle="-|>", color="#00798c", lw=1.3, connectionstyle="arc3,rad=0.9"))
    ax.text(4.88, 2.78, "recurrent state $h_{t-1}$ (LSTM: also $c_{t-1}$)", ha="center", fontsize=8, color="#00798c")
    return _save(fig, "model_architecture")


# ------------------------------------------------------------------ results
def load_runs() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((RES / "runs").glob("*.json"))}


def results_table(runs: dict) -> pd.DataFrame:
    rows = []
    for k, r in runs.items():
        c = r["config"]
        rows.append({"Run": k, "Experiment": k.split("_")[0], "Cell": c["cell"].upper(), "Hidden": c["hidden"],
                     "Layers": c["layers"], "Seq len": c["seq_len"], "Batch": c["batch"],
                     "Params (M)": round(r["params"] / 1e6, 3),
                     "Train loss": round(r["final_train_loss"], 4),
                     "Val BPC": round(r["best_val_bpc"], 4), "Test BPC": round(r["test_bpc"], 4),
                     "Test PPL": round(r["test_ppl"], 3),
                     "Gen. gap (BPC)": round(r["best_val_bpc"] - r["final_train_loss"] / LN2, 4),
                     "Train time (min)": round(r["train_time_s"] / 60, 1)})
    return _tab(pd.DataFrame(rows), "all_runs")


def fig_curves(runs: dict, names: list[str], labels: list[str], title: str, fname: str, colors=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
    cmap = plt.get_cmap("viridis")
    for i, (n, lab) in enumerate(zip(names, labels)):
        if n not in runs:
            continue
        h = runs[n]["history"]
        col = colors[i] if colors else cmap(i / max(1, len(names) - 1) * 0.9)
        ax[0].plot(h["step"], np.array(h["train_loss"]) / LN2, color=col, lw=1.4, label=lab)
        ax[1].plot(h["eval_step"], np.array(h["val_loss"]) / LN2, "o-", ms=3, color=col, lw=1.4, label=lab)
    for a, t in zip(ax, ("Training loss", "Validation loss")):
        a.set_xlabel("training step"); a.set_ylabel("bits per character"); a.set_title(t); a.legend(fontsize=8.5)
        a.set_ylim(top=min(a.get_ylim()[1], 3.2))
    fig.suptitle(title, fontweight="bold")
    return _save(fig, fname)


def fig_grad_norms(runs: dict):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for n, c in (("E1_rnn", "rnn"), ("E1_gru", "gru"), ("E1_lstm", "lstm")):
        if n in runs:
            h = runs[n]["history"]
            ax.plot(h["step"], h["grad_norm"], color=PALETTE[c], label=c.upper(), lw=1.3)
    ax.axhline(1.0, color="k", ls=":", lw=1); ax.text(30, 1.05, "clip threshold", fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("global grad norm (pre-clip)")
    ax.set_title("Gradient norm during training"); ax.legend()
    return _save(fig, "e1_grad_norms")


def fig_bar(df: pd.DataFrame, xcol: str, title: str, fname: str, ngram_bpc: float | None = None, colors=None):
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["Val BPC"], w, label="validation", color=colors or PALETTE["accent"], alpha=0.65)
    ax.bar(x + w / 2, df["Test BPC"], w, label="test", color=colors or PALETTE["accent"])
    for i, v in enumerate(df["Test BPC"]):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
    if ngram_bpc:
        ax.axhline(ngram_bpc, color=PALETTE["ngram"], ls="--", lw=1.2)
        ax.text(len(df) - 0.5, ngram_bpc + 0.01, f"best n-gram {ngram_bpc:.3f}", ha="right", fontsize=8,
                color=PALETTE["ngram"])
    ax.set_xticks(x); ax.set_xticklabels(df[xcol].astype(str))
    lo = min(df["Val BPC"].min(), df["Test BPC"].min(), ngram_bpc or 9)
    ax.set_ylim(lo - 0.15, max(df["Val BPC"].max(), ngram_bpc or 0) + 0.12)
    ax.set_ylabel("bits per character (lower = better)"); ax.set_title(title); ax.legend(fontsize=8.5)
    return _save(fig, fname)


def fig_tradeoff(tab: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    col = [PALETTE[c.lower()] for c in tab.Cell]
    for a, x, xl in ((ax[0], "Params (M)", "parameters (millions)"), (ax[1], "Train time (min)", "training time (minutes)")):
        a.scatter(tab[x], tab["Test BPC"], c=col, s=55, edgecolor="k", lw=0.5, zorder=3)
        for _, r in tab.iterrows():
            lab = r.Run.replace("E1_", "").replace("E2_lstm_", "").replace("E3_lstm_", "")
            if x == "Params (M)" and r.Run.startswith("E2_"):
                continue                      # same size as E1_lstm -> one shared label
            if x == "Params (M)" and r.Run == "E1_lstm":
                lab = "lstm 256×2 (seq 32–256)"
            a.annotate(lab, (r[x], r["Test BPC"]), xytext=(4, 4), textcoords="offset points", fontsize=7.5,
                       ha="right" if x == "Params (M)" and r[x] > 0.8 else "left")
        a.set_xlabel(xl); a.set_ylabel("test BPC")
    ax[0].set_title("Accuracy vs model size"); ax[1].set_title("Accuracy vs compute")
    return _save(fig, "tradeoff")


def fig_sampling(samp: dict):
    df = pd.DataFrame(samp["rows"])
    ref = samp["reference_test_text"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    x = np.arange(len(df))
    for a, k, t in ((ax[0], "valid_word_rate", "Valid-word rate (higher = fewer misspellings)"),
                    (ax[1], "distinct_2", "Distinct-2 (higher = more diverse)"),
                    (ax[2], "repeat_4gram_rate", "Repeated 4-grams (lower = less looping)")):
        a.bar(x, df[k], color=PALETTE["lstm"])
        a.axhline(ref[k], color=PALETTE["rnn"], ls="--", lw=1.2, label="real Austen text")
        a.set_xticks(x); a.set_xticklabels(df.strategy, rotation=40, ha="right", fontsize=8)
        a.set_title(t, fontsize=9); a.legend(fontsize=7.5)
    return _save(fig, "e5_sampling_metrics"), _tab(df.round(4), "sampling_metrics")
