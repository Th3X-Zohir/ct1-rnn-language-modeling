# CT1 Task 3 — Character-Level Language Modeling with RNN, GRU and LSTM

**Course:** CSE638 Deep Learning, Daffodil International University · **Assignment:** Group Assignment 1 (CT1), Task 3 (RNN / Sequence Modeling)
**Group:** Recurrent Minds — Shafin Ahmed (262-25-035), Zohir Rayhan (262-25-036), Sheikh Ajwad (262-25-337)

We train character-level language models on six Jane Austen novels from Project Gutenberg and compare:

| Exp | Question | Runs |
|---|---|---|
| E1 | cell type | vanilla RNN vs GRU vs LSTM (hidden 256, 2 layers) |
| E2 | BPTT sequence length | LSTM with 32 / 64 / 128 / 256 |
| E3 | capacity / depth | LSTM hidden 128 / 256 / 384, layers 1 / 2 / 3 |
| E4 | non-neural baseline | Witten-Bell character n-gram (order 3 / 5 / 7) |
| E5 | sampling strategy | greedy, temperature, top-k, nucleus |

Metrics: bits per character (BPC) and character perplexity on a held-out test split.

<!-- RESULTS -->

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
checkpoints/       best checkpoint of every run
report/            group report (DOCX + PDF) and its build script
```

## Reproduce
```bash
pip install -r requirements.txt
python run_experiments.py --workers 3 --threads 4   # ~3 h on a 16-core CPU, downloads data automatically
jupyter nbconvert --to notebook --execute --inplace CT1_Task3_RNN_Language_Modeling.ipynb
```
Seeds are fixed (42). Data: public-domain texts from Project Gutenberg (IDs 1342, 158, 161, 141, 121, 105).
