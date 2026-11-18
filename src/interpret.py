"""Interpretability of the trained character model (what do the hidden units track?).

1. Run the model statefully over held-out text and record the top-layer hidden state h_t.
2. Define interpretable per-character properties (inside a quotation, start of a word,
   position inside the word, capital letter, end of sentence, position in the paragraph).
3. For each property find the single hidden unit with the strongest Pearson correlation.
4. Linear probes: logistic / linear regression from h_t to each property, compared with a
   probe that only sees the *current character* (one-hot). If the hidden state beats the
   current-character probe, the network has stored information about the past.
5. Visualise the activation of the best units over a passage of real text (Karpathy et al., 2015).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

from .models import CharRNN

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"


@torch.no_grad()
def hidden_states(model: CharRNN, ids: torch.Tensor, chunk: int = 512) -> np.ndarray:
    """Top-layer hidden state for every character (stateful single stream)."""
    model.eval(); state, out = None, []
    for i in range(0, len(ids), chunk):
        h, state = model.rnn(model.embed(ids[i:i + chunk][None]), state)
        out.append(h[0].numpy())
    return np.concatenate(out)


def char_features(text: str) -> pd.DataFrame:
    """Properties of the text *after reading character t* (what the state could know)."""
    in_q, pos_w, pos_p, q = [], [], [], False
    w = p = 0
    for c in text:
        if c == "\n":
            q = False          # quotations never run across a paragraph break without re-opening
        elif c == '"':
            q = not q
        in_q.append(int(q))
        w = w + 1 if c.isalpha() else 0
        p = 0 if c == "\n" else p + 1
        pos_w.append(w); pos_p.append(p)
    nxt = text[1:] + " "
    return pd.DataFrame({
        "inside quotation": in_q,
        "next char starts a word": [int(not a.isalpha() and b.isalpha()) for a, b in zip(text, nxt)],
        "position inside word": pos_w,
        "capital letter": [int(c.isupper()) for c in text],
        "end of sentence": [int(c in ".!?") for c in text],
        "position in paragraph (log)": np.log1p(pos_p),
    })


def analyse(ckpt: str = "E1_gru", n_chars: int = 60_000):
    from .data import build_corpus, encode
    texts, stoi, meta, _ = build_corpus(ROOT / "data")
    ck = torch.load(ROOT / "checkpoints" / f"{ckpt}.pt"); c = ck["config"]
    model = CharRNN(meta["vocab_size"], c["cell"], c["emb_dim"], c["hidden"], c["layers"], c["dropout"])
    model.load_state_dict(ck["state_dict"])
    text = texts["test"][:n_chars]
    H = hidden_states(model, encode(text, stoi))
    F = char_features(text)
    onehot = np.eye(meta["vocab_size"])[[stoi[ch] for ch in text]]
    split = int(0.7 * len(text))

    rows, probes = [], []
    Hz = (H - H.mean(0)) / (H.std(0) + 1e-8)
    for name in F.columns:
        y = F[name].to_numpy().astype(float)
        yz = (y - y.mean()) / (y.std() + 1e-8)
        r = Hz.T @ yz / len(y)
        u = int(np.abs(r).argmax())
        rows.append({"property": name, "best unit": u, "pearson r": round(float(r[u]), 3),
                     "units with |r|>0.3": int((np.abs(r) > 0.3).sum())})
        binary = set(np.unique(y)) <= {0.0, 1.0}
        res = {"property": name, "type": "accuracy" if binary else "R²"}
        for lab, X in (("hidden state", H), ("current char only", onehot)):
            if binary:
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[:split], y[:split])
                res[lab] = round(float(clf.score(X[split:], y[split:])), 4)
            else:
                reg = LinearRegression().fit(X[:split], y[:split])
                res[lab] = round(float(r2_score(y[split:], reg.predict(X[split:]))), 4)
        if binary:
            res["majority baseline"] = round(float(max(y[split:].mean(), 1 - y[split:].mean())), 4)
        else:
            res["majority baseline"] = 0.0
        probes.append(res)
    units = pd.DataFrame(rows); probe = pd.DataFrame(probes)
    units.to_csv(TAB / "interp_units.csv", index=False); probe.to_csv(TAB / "interp_probes.csv", index=False)

    # ---- figure 1: unit activations over a dialogue passage
    start = text.find('"', 3000)
    start = text.rfind("\n", 0, start) + 1
    seg = text[start:start + 560]
    pick = [("inside quotation", "a unit that tracks whether we are inside a quotation"),
            ("next char starts a word", "a unit that fires at word boundaries (a new word comes next)")]
    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 3.6 * len(pick)))
    for ax, (prop, desc) in zip(axes, pick):
        u = int(units.set_index("property").loc[prop, "best unit"])
        r = float(units.set_index("property").loc[prop, "pearson r"])
        act = H[start:start + len(seg), u] * np.sign(r)
        lo, hi = np.percentile(H[:, u] * np.sign(r), [2, 98])
        norm = np.clip((act - lo) / (hi - lo + 1e-8), 0, 1)
        cols, x, y = 80, 0, 0
        ax.set_xlim(0, cols); ax.set_ylim(-(len(seg) // cols + 2), 1); ax.axis("off")
        cmap = plt.get_cmap("RdBu_r")
        for ch, v in zip(seg, norm):
            if ch == "\n":
                x, y = 0, y - 1; continue
            ax.add_patch(plt.Rectangle((x, y - 0.5), 1, 1, color=cmap(v), lw=0))
            ax.text(x + 0.5, y, ch, ha="center", va="center", family="monospace", fontsize=8.5)
            x += 1
            if x >= cols:
                x, y = 0, y - 1
        ax.set_title(f"Unit {u} — {desc} (r = {r:+.2f}); red = high, blue = low", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "interp_unit_activations.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: probe accuracy vs baselines
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    b = probe[probe.type == "accuracy"]
    xs = np.arange(len(b)); w = 0.27
    axs[0].bar(xs - w, b["majority baseline"], w, label="majority class", color="#bbbbbb")
    axs[0].bar(xs, b["current char only"], w, label="current char only", color="#edae49")
    axs[0].bar(xs + w, b["hidden state"], w, label="GRU hidden state", color="#00798c")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(b.property, fontsize=8, rotation=15)
    axs[0].set_ylim(0.5, 1.01); axs[0].set_ylabel("probe accuracy (held-out)"); axs[0].legend(fontsize=8)
    axs[0].set_title("Linear probes: what does the hidden state know?", fontsize=10)
    axs[1].barh(units.property, units["pearson r"].abs(), color="#30638e")
    axs[1].set_xlabel("|Pearson r| of the single best unit"); axs[1].invert_yaxis()
    axs[1].set_title("Strongest single-unit correlation", fontsize=10)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "interp_probes.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    summary = {"checkpoint": ckpt, "n_chars": len(text), "hidden_units": int(H.shape[1]),
               "units": rows, "probes": probes}
    (ROOT / "results" / "interpretability.json").write_text(json.dumps(summary, indent=1))
    return units, probe


if __name__ == "__main__":
    u, p = analyse()
    print(u.to_string()); print(p.to_string())
