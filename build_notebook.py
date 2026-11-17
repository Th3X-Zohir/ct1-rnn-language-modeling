"""Generate CT1_Task3_RNN_Language_Modeling.ipynb (executed afterwards with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CT1 – Task 3: RNN Sequence Modeling
## Character-Level Language Modeling of Jane Austen's Novels with Vanilla RNN, GRU and LSTM

**Course:** CSE638 Deep Learning, Daffodil International University
**Group:** Recurrent Minds

| Member | Student ID |
|---|---|
| Shafin Ahmed | 262-25-035 |
| Zohir Rayhan | 262-25-036 |
| Sheikh Ajwad | 262-25-337 |

**Task (from the assignment sheet):** *Train an RNN (vanilla, GRU, simple LSTM) for character-level or word-level language
modeling / text generation … compare different RNN variants, sequence length effects, sampling strategies, and perplexity.*

**Research questions**
1. **RQ1 – Cell type:** how much do gated cells (GRU, LSTM) improve on a vanilla RNN at equal width and compute?
2. **RQ2 – Sequence length:** how does the BPTT window (32–256 characters) affect quality and cost?
3. **RQ3 – Capacity:** how do hidden size and depth trade off against accuracy and training time?
4. **RQ4 – Baseline:** do neural models beat a strong smoothed character n-gram model?
5. **RQ5 – Sampling:** how do greedy, temperature, top-k and nucleus decoding change the generated text?

> The heavy training runs are executed by `run_experiments.py` (all ten runs, ~3 CPU-hours) and saved to `results/`.
> This notebook re-uses the same modules in `src/`, performs the full data analysis live, loads the saved runs,
> and produces every table and figure used in the report. Set `RETRAIN = True` to retrain from scratch.
""")

code("""
import sys, json, math, time, textwrap
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from src.data import build_corpus, encode, BOOKS
from src.models import CharRNN, n_params
from src.train import Config, evaluate, LN2
from src import analysis as A
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
torch.set_num_threads(4); torch.manual_seed(42)
RETRAIN = False
print("torch", torch.__version__, "| threads", torch.get_num_threads())
""")

md("""
## 1. Data acquisition and cleaning
Six complete novels by Jane Austen are downloaded from Project Gutenberg (public domain). A single-author corpus keeps
style consistent, so differences between models reflect modeling ability rather than domain shift.

**Cleaning pipeline (`src/data.py`)**
1. strip Gutenberg licence header/footer; 2. normalise line endings and Unicode punctuation to ASCII;
3. remove `[Illustration]` markup and `_italic_` underscores; 4. un-wrap hard line breaks inside paragraphs;
5. drop characters that occur fewer than 20 times (e.g. `£`, `æ`).

**Split:** every novel is split *contiguously* 90 % / 5 % / 5 % into train / validation / test, so each split contains all
six novels but no passage appears in two splits (no leakage from overlapping windows).
""")

code("""
texts, stoi, meta, cleaned = build_corpus(ROOT / "data")
itos = {i: c for c, i in stoi.items()}
print("Vocabulary size:", meta["vocab_size"])
print("Vocabulary:", repr("".join(meta["vocab"])))
print("Rare characters removed:", meta["removed_rare"])
corpus = A.corpus_table(meta, cleaned); display(corpus)
display(A.split_table(meta))
print(cleaned[1342][:600])
""")

md("## 2. Exploratory data analysis")
code("""
A.fig_book_sizes(corpus); plt.show()
fig, H, classes = A.fig_char_freq(texts["train"]); plt.show()
print(f"Unigram entropy: {H:.3f} bits/char  (upper bound a model must beat; uniform over 75 chars = {math.log2(75):.3f})")
print("Character classes (%):", classes)
""")
code("""
fig, L = A.fig_lengths(texts["train"]); plt.show()
fig, W = A.fig_top_words(texts["train"]); plt.show()
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in {**L, **W}.items()})
""")
code("""
fig, sparsity = A.fig_ngram_coverage(texts["train"], texts["val"]); plt.show()
display(sparsity)
""")
md("""
**EDA findings**
* The cleaned corpus has 4.05 M characters (0.73 M words, 14.5 k distinct word types); cleaning removed only 3.1 %.
  75 distinct characters remain: letters 78.2 %, space/newline 18.2 %, punctuation 3.6 %, digits 0.01 %.
* The unigram entropy (4.41 bits/char) is the score of a model that ignores context; every useful model must go far below it.
* Mean word length is 4.3 characters, so a 128-character window spans ~25 words. Sentences have a median of 13 and a
  mean of 18.5 words (long tail); paragraphs have a median of 239 characters, but the 95th percentile is 1 192 characters,
  so long-range structure (quotations, dialogue turns) exists — the motivation for testing longer BPTT windows.
* One third of the training word types (33 %) are hapax legomena (seen once): a word-level model would struggle
  with rare words, while a character model can still spell them.
* Sparsity: 16.3 % of validation 8-grams and 36.7 % of 10-grams never occur in training. Count-based n-gram models
  therefore run out of usable context after ~6–7 characters, which is the core argument for recurrent models that
  generalise over long histories.
""")

md("""
## 3. Workflow
```
Gutenberg (6 novels) ─► clean & normalise ─► 90/5/5 contiguous split ─► char vocabulary (75)
        │                                                                     │
        ▼                                                                     ▼
   EDA (stats, sparsity)                    random windows (seq_len+1) ─► CharRNN {RNN|GRU|LSTM}
                                                                              │  AdamW, warm-up + cosine LR,
                                                                              │  grad-clip 1.0, dropout 0.2
                                                                              ▼
                          stateful evaluation on val (model selection) / test (final) ─► BPC, perplexity
                                                                              │
                                              n-gram baseline ◄───────────────┤
                                                                              ▼
                                         sampling study (greedy / temperature / top-k / nucleus)
```
""")

md("""
## 4. Models
All neural models share one class (`src/models.py`): **Embedding(75→64) → stacked recurrent layers → Dropout → Linear(→75)**.
Only the recurrent cell or its size changes between runs.

* **Vanilla RNN:** $h_t=\\tanh(W_x x_t + W_h h_{t-1} + b)$ – no gating, gradients vanish/explode over long spans.
* **GRU:** update gate $z_t$ and reset gate $r_t$; $h_t=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde h_t$ (3 weight blocks).
* **LSTM:** input/forget/output gates and a separate cell state $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t$ (4 weight blocks).

**Training setup (identical for every run):** AdamW (lr 2e-3), 100-step warm-up + cosine decay, 2 500 steps,
8 192 characters per batch (batch = 8192 / seq_len, so each step costs the same compute regardless of sequence length),
gradient clipping at 1.0, dropout 0.2, seed 42. The checkpoint with the best validation loss is kept.

**Metrics:** cross-entropy in bits per character (**BPC** = nats / ln 2) and character **perplexity** = e^loss.
Evaluation is stateful (hidden state carried across 256-char chunks), so scores are comparable across training sequence lengths.
""")
code("""
rows = []
for cell in ("rnn", "gru", "lstm"):
    m = CharRNN(meta["vocab_size"], cell)
    rows.append({"Cell": cell.upper(), "Embedding": 75 * 64, "Recurrent": sum(p.numel() for p in m.rnn.parameters()),
                 "Output": sum(p.numel() for p in m.head.parameters()), "Total": n_params(m)})
display(A._tab(pd.DataFrame(rows), "model_params"))
print(CharRNN(meta["vocab_size"], "lstm"))
""")

md("""
## 5. Experiments
| ID | Question | Runs |
|---|---|---|
| E1 | cell type | RNN, GRU, LSTM — hidden 256, 2 layers, seq 128 |
| E2 | sequence length | LSTM with seq 32, 64, 128 (=E1_lstm), 256 |
| E3 | capacity / depth | LSTM hidden 128, 256, 384; layers 1, 2, 3 |
| E4 | non-neural baseline | Witten-Bell character n-gram, order 3 / 5 / 7 |
| E5 | sampling | 7 decoding strategies on the best model |
""")
code("""
if RETRAIN:
    import subprocess; subprocess.run([sys.executable, "run_experiments.py", "--workers", "3", "--threads", "4"], check=True)
runs = A.load_runs()
tab = A.results_table(runs)
ngram = pd.DataFrame(json.loads((ROOT / "results/ngram_baseline.json").read_text()))
display(tab)
display(ngram.round(4))
""")

md("### 5.1 E1 – Vanilla RNN vs GRU vs LSTM")
code("""
best_ng = ngram.test_bpc.min()
e1 = tab[tab.Experiment == "E1"].set_index("Cell").loc[["RNN", "GRU", "LSTM"]].reset_index()
display(e1[["Cell", "Params (M)", "Val BPC", "Test BPC", "Test PPL", "Train time (min)"]])
A.fig_curves(runs, ["E1_rnn", "E1_gru", "E1_lstm"], ["RNN", "GRU", "LSTM"], "E1: learning curves by cell type",
             "e1_learning_curves", colors=[A.PALETTE[c] for c in ("rnn", "gru", "lstm")]); plt.show()
A.fig_bar(e1, "Cell", "E1: cell type (hidden 256, 2 layers)", "e1_bpc_bar", best_ng,
          colors=[A.PALETTE[c] for c in ("rnn", "gru", "lstm")]); plt.show()
A.fig_grad_norms(runs); plt.show()
""")

md("### 5.2 E2 – Effect of BPTT sequence length (LSTM)")
code("""
e2names = ["E2_lstm_seq32", "E2_lstm_seq64", "E1_lstm", "E2_lstm_seq256"]
e2 = tab.set_index("Run").loc[e2names].reset_index()
display(e2[["Run", "Seq len", "Batch", "Val BPC", "Test BPC", "Train time (min)"]])
A.fig_curves(runs, e2names, ["seq 32", "seq 64", "seq 128", "seq 256"], "E2: learning curves by sequence length",
             "e2_learning_curves"); plt.show()
A.fig_bar(e2, "Seq len", "E2: BPTT length (equal chars per batch)", "e2_bpc_bar", best_ng); plt.show()
""")

md("### 5.3 E3 – Capacity and depth (LSTM)")
code("""
e3h = tab.set_index("Run").loc[["E3_lstm_h128", "E1_lstm", "E3_lstm_h384"]].reset_index()
e3l = tab.set_index("Run").loc[["E3_lstm_L1", "E1_lstm", "E3_lstm_L3"]].reset_index()
display(e3h[["Run", "Hidden", "Params (M)", "Train loss", "Val BPC", "Test BPC", "Train time (min)"]])
display(e3l[["Run", "Layers", "Params (M)", "Train loss", "Val BPC", "Test BPC", "Train time (min)"]])
A.fig_curves(runs, ["E3_lstm_h128", "E1_lstm", "E3_lstm_h384", "E3_lstm_L1", "E3_lstm_L3"],
             ["hidden 128", "hidden 256 (2L)", "hidden 384", "1 layer", "3 layers"],
             "E3: learning curves by capacity", "e3_learning_curves"); plt.show()
A.fig_bar(e3h, "Hidden", "E3a: hidden size (2 layers)", "e3_hidden_bar", best_ng); plt.show()
A.fig_bar(e3l, "Layers", "E3b: depth (hidden 256)", "e3_layers_bar", best_ng); plt.show()
A.fig_tradeoff(tab); plt.show()
""")

md("### 5.4 E4 – Comparison with the n-gram baseline (test set)")
code("""
best_run = tab.loc[tab["Val BPC"].idxmin()]
final = pd.concat([
    ngram.assign(Model=ngram.model)[["Model", "test_bpc", "test_ppl"]].rename(columns={"test_bpc": "Test BPC", "test_ppl": "Test PPL"}),
    e1.assign(Model=e1.Cell + " (256x2)")[["Model", "Test BPC", "Test PPL"]],
    pd.DataFrame([{"Model": f"Best overall: {best_run.Run}", "Test BPC": best_run["Test BPC"], "Test PPL": best_run["Test PPL"]}]),
], ignore_index=True).round(4)
A._tab(final, "final_test_comparison"); display(final)
print("Best model selected on validation BPC:", best_run.Run)
""")

md("### 5.5 E5 – Sampling strategies")
code("""
samp = json.loads((ROOT / "results/sampling.json").read_text())
fig, sdf = A.fig_sampling(samp); plt.show()
display(sdf)
print("Reference metrics on real held-out Austen text:", {k: round(v, 3) for k, v in samp["reference_test_text"].items()})
for k, v in samp["examples"].items():
    print(f"\\n=== {k} ===\\n" + textwrap.fill(v[:420], 110))
""")

md("""
### 5.6 Interactive generation with the best model
""")
code("""
from src.generate import sample
ck = torch.load(ROOT / f"checkpoints/{best_run.Run}.pt"); c = ck["config"]
best_model = CharRNN(meta["vocab_size"], c["cell"], c["emb_dim"], c["hidden"], c["layers"], c["dropout"])
best_model.load_state_dict(ck["state_dict"])
for prompt in ["Elizabeth looked at him", "\\"I cannot think"]:
    txt, _ = sample(best_model, stoi, itos, prompt, n=400, temperature=0.8, seed=7)
    print(textwrap.fill(txt, 110), "\\n")
""")

md("""
## 6. Summary
The conclusions are written in the report from the tables above; the key numbers are printed below for reference.
""")
code("""
summary = {
    "best_model": best_run.Run, "best_test_bpc": float(best_run["Test BPC"]), "best_test_ppl": float(best_run["Test PPL"]),
    "rnn_test_bpc": float(e1.loc[e1.Cell == "RNN", "Test BPC"].iloc[0]),
    "gru_test_bpc": float(e1.loc[e1.Cell == "GRU", "Test BPC"].iloc[0]),
    "lstm_test_bpc": float(e1.loc[e1.Cell == "LSTM", "Test BPC"].iloc[0]),
    "best_ngram_test_bpc": float(best_ng), "unigram_entropy": float(H),
}
(ROOT / "results/summary.json").write_text(json.dumps(summary, indent=1)); summary
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "CT1_Task3_RNN_Language_Modeling.ipynb")
print("notebook written with", len(C), "cells")
