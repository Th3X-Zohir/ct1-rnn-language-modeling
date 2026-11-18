# CT1 Task 3 — Character-Level Language Modeling with RNN, GRU and LSTM

**Course:** CSE638 Deep Learning, Daffodil International University · **Assignment:** Group Assignment 1 (CT1), Task 3 (RNN / Sequence Modeling)
**Group:** Recurrent Minds — Shafin Ahmed (262-25-035), Zohir Rayhan (262-25-036), Sheikh Ajwad (262-25-037)

We train character-level language models on six Jane Austen novels from Project Gutenberg and compare:

| Exp | Question | Runs |
|---|---|---|
| E1 | cell type | vanilla RNN vs GRU vs LSTM (hidden 256, 2 layers) |
| E2 | BPTT sequence length | LSTM with 32 / 64 / 128 / 256 |
| E3 | capacity / depth | LSTM hidden 128 / 256 / 384, layers 1 / 2 / 3 |
| E4 | non-neural baseline | Witten-Bell character n-gram (order 3 / 5 / 7) |
| E5 | sampling strategy | greedy, temperature, top-k, nucleus |
| E6 | final model | best GRU trained 3.2× longer (8,000 steps) |
| + | interpretability | hidden-unit correlations and linear probes |
| + | browser demo | the trained GRU running in plain JavaScript |

Metrics: bits per character (BPC) and character perplexity on a held-out test split.

## Key results (test set, lower BPC = better)

| Model | Params | Test BPC | Test PPL |
|---|---|---|---|
| Unigram entropy (no context) | – | 4.411 | 21.27 |
| Witten-Bell 5-gram (best n-gram) | – | 1.653 | 3.14 |
| Vanilla RNN 256×2 | 0.24 M | 1.766 | 3.40 |
| LSTM 256×2 | 0.88 M | 1.726 | 3.31 |
| LSTM 384×2 | 1.91 M | 1.599 | 3.03 |
| GRU 256×2 (best in the controlled study) | 0.67 M | 1.585 | 3.00 |
| **E6: GRU 256×2, 8,000 steps (final model)** | 0.67 M | **1.498** | **2.82** |

* **Cell type:** under an equal 2,500-step budget the GRU learns fastest and is best; the LSTM starts slowly but overtakes the vanilla RNN.
* **Sequence length:** 32 → 256 characters improves the LSTM only from 1.736 to 1.720 BPC.
* **Capacity:** hidden size is the strongest lever (128 → 384 units: 2.027 → 1.599); 2 layers beat 1, a 3rd layer does not pay off within the budget.
* **Sampling:** greedy decoding loops (78 % repeated 4-grams); nucleus (p = 0.9) gives the best spelling/diversity balance.
* **Final model (E6):** training the best GRU 3.2× longer lowers test BPC from 1.585 to **1.498** (9.4 % better than the best n-gram).
* **Interpretability:** one hidden unit acts as a *quotation detector* (|r| = 0.90); a linear probe on the hidden state predicts "inside a quotation" with 89 % accuracy vs 53 % from the current character.

## Live demo
Open [`demo/index.html`](demo/index.html) in any browser: the E6 model generates Austen-style text on your device (no server).
The JavaScript GRU is verified against PyTorch with `node demo/check_js.js`.

Full analysis: [`report/CT1_Task3_Report_Recurrent_Minds.pdf`](report/CT1_Task3_Report_Recurrent_Minds.pdf)

## Repository layout
```
CT1_Task3_RNN_Language_Modeling.ipynb   executed notebook (EDA, models, results, figures, generation)
run_experiments.py                      all training runs + n-gram baseline + sampling study (resumable)
build_notebook.py                       generates the notebook
src/data.py        download, cleaning, vocabulary, contiguous 90/5/5 split, batching
src/models.py      CharRNN (RNN / GRU / LSTM)
src/train.py       training loop, stateful evaluation, BPC / perplexity
src/ngram.py       Witten-Bell n-gram baseline
src/generate.py    greedy / temperature / top-k / nucleus sampling + text metrics
src/analysis.py    EDA statistics and all figures / tables
results/           per-run JSON, logs, CSV tables, PNG figures, sampling results
run_final_model.py E6: best GRU trained longer (results/final/)
src/interpret.py   interpretability: unit correlations, linear probes, activation plots
demo/              browser demo: index.html, gru.js, export_weights.py, check_js.js
tests/             16 unit tests
checkpoints/       best checkpoint of every run
report/            group report (DOCX + PDF) and its build script
```

## Reproduce
```bash
pip install -r requirements.txt
python run_experiments.py --workers 3 --threads 4   # ~3 h on a 16-core CPU, downloads data automatically
python run_final_model.py --threads 2                 # E6, ~90 min (2 threads is fastest for this small GRU)
python -m unittest discover -s tests                   # 16 unit tests
jupyter nbconvert --to notebook --execute --inplace CT1_Task3_RNN_Language_Modeling.ipynb
```
Seeds are fixed (42). Data: public-domain texts from Project Gutenberg (IDs 1342, 158, 161, 141, 121, 105).
