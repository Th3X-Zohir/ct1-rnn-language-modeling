// Result-independent narrative for the report.
exports.limitations = () => [
  "**Fixed compute budget.** Every run was limited to 2,500 steps (≈ 5.6 epochs) on a CPU. Larger models, in particular, were still improving when training stopped, so the capacity comparison shows the best model *for this budget*, not the best achievable model.",
  "**Single seed.** Each configuration was trained once (seed 42). Differences of a few thousandths of a BPC are within run-to-run noise and should not be over-interpreted; the large differences (cell type, sequence length 32 vs 128) are well above it.",
  "**Equal width, not equal parameters, in E1.** GRU and LSTM have more parameters than the vanilla RNN at the same hidden size. E3 shows that size alone does not explain the gap, but a parameter-matched comparison would be cleaner.",
  "**One author, one domain.** The corpus is 19th-century English prose by a single author. The relative ranking of the cells is well established in the literature, but absolute BPC values do not transfer to other domains.",
  "**Automatic text metrics only.** The sampling study uses automatic proxies (valid-word rate, distinct-n, repetition). They capture spelling, diversity and looping but not coherence or style; no human evaluation was carried out.",
];
exports.future = () => [
  "Train the best configuration for much longer (e.g. 50 k steps on a GPU) with layer normalisation and weight tying to measure the achievable BPC on this corpus.",
  "Compare against a small character-level Transformer with the same parameter budget (Task 5 of the task sheet) to quantify the benefit of self-attention over recurrence.",
  "Repeat every configuration with three seeds and report mean ± standard deviation, together with significance tests.",
  "Add a word-level / sub-word (BPE) variant to study the effect of tokenisation on perplexity and generation quality.",
  "Run a small blind human evaluation of generated passages (fluency, coherence, “Austen-ness”) to complement the automatic metrics.",
];
