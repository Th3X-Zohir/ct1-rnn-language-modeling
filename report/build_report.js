// Build the CT1 group report (DOCX). Usage: node build_report.js  -> CT1_Task3_Report_Recurrent_Minds.docx
const fs = require("fs");
const path = require("path");
const L = require("./lib");
const { D, P, H1, H2, H3, bullet, numbered, pageBreak, equation, figure, table, csv, json, runs } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, Header, PageNumber, TableOfContents,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, TableLayoutType } = D;
const T = require("./text");   // result-dependent narrative (written after the experiments)

const META = {
  university: "Daffodil International University",
  dept: "Department of Computer Science and Engineering",
  course: "CSE638 — Deep Learning",
  assignment: "Group Assignment 1 (CT1)",
  task: "Task 3: RNN / Sequence Modeling",
  title: "Character-Level Language Modeling of Jane Austen's Novels with Vanilla RNN, GRU and LSTM",
  group: "Recurrent Minds",
  members: [
    ["Shafin Ahmed", "262-25-035", "262-25-035@diu.edu.bd"],
    ["Zohir Rayhan", "262-25-036", "262-25-036@diu.edu.bd"],
    ["Sheikh Ajwad", "262-25-037", "262-25-037@diu.edu.bd"],
  ],
  teacher: ["Sadat Hasan", "Adjunct Faculty", "Department of Computer Science and Engineering", "Daffodil International University"],
  date: "17.11.2026",
  repo: process.env.REPO_URL || "https://github.com/Th3X-Zohir/ct1-rnn-language-modeling",
};

const runsTab = csv("all_runs");
const byRun = Object.fromEntries(runsTab.map((r) => [r.Run, r]));
const ngram = json("results/ngram_baseline.json");
const samp = json("results/sampling.json");
const summary = json("results/summary.json");
const corpus = csv("corpus_statistics");
const f3 = (x) => Number(x).toFixed(3);
const f2 = (x) => Number(x).toFixed(2);
const n0 = (x) => Number(x).toLocaleString("en-US");

// ------------------------------------------------------------------ cover page
// Designed full-bleed cover (rendered by make_cover.py); the section has zero margins.
function cover() {
  return [new Paragraph({ spacing: { before: 0, after: 0, line: 240 }, children: [
    new D.ImageRun({ type: "png", data: fs.readFileSync(path.join(__dirname, "cover_page.png")),
      transformation: { width: 793, height: 1121 },
      altText: { title: "Cover page", description: `${META.course} ${META.assignment}: ${META.title}. Group ${META.group}.`, name: "cover" } })] })];
}

// Plain text cover in the class template layout (kept for reference, not used).
function coverTemplate() {
  // Same layout as the class's group-report cover template (DIU logo, course name/code, topic, group table, teacher).
  const F = L.BODY_FONT;
  const c = (parts, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: opts.before ?? 0, after: opts.after ?? 80 },
    children: parts.map(([text, o]) => new TextRun({ text, font: F, size: o.size, bold: o.bold, italics: o.italics, color: o.color })) });
  const logo = fs.readFileSync(path.join(__dirname, "diu_logo.png"));
  const memberRows = META.members.map((m, i) => [String(i + 1), m[0], m[1]]);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 360 }, children: [
      new D.ImageRun({ type: "png", data: logo, transformation: { width: 380, height: Math.round(380 * 233 / 840) },
        altText: { title: "Daffodil International University", description: "DIU logo", name: "diu_logo" } })] }),
    c([["Course Name: Deep Learning", { size: 32, bold: true }]], { after: 60 }),
    c([["Course Code: CSE638", { size: 32, bold: true }]], { after: 80 }),
    c([[`${META.assignment} — Group Report`, { size: 26 }]], { after: 360 }),
    c([[`Task 3: ${META.title}`, { size: 28, bold: true, color: "2F5496" }]], { after: 160 }),
    c([["Dataset: Project Gutenberg — six novels by Jane Austen (4.05 M characters)", { size: 24, italics: true }]], { after: 400 }),
    c([["Group Name: ", { size: 28 }], [META.group, { size: 28, bold: true }]], { after: 160 }),
    ...table(["#", "Student Name", "Student ID"], memberRows, null, [700, 4600, 3000], { size: 24, centerFirst: true }),
    c([["Submitted to: ", { size: 28 }], [META.teacher[0], { size: 28, bold: true }], [" [Adj. Faculty]", { size: 28 }]], { before: 240, after: 60 }),
    c([[META.teacher[3], { size: 28 }]], { after: 360 }),
    c([["Submission date: ", { size: 28, bold: true }], [META.date, { size: 28 }]], { after: 240 }),
    c([["Codebase: ", { size: 22, bold: true }], [META.repo, { size: 22, color: "2F5496" }]], { after: 0 }),
    pageBreak(),
  ];
}

// ------------------------------------------------------------------ sections
function abstractPart() {
  return [
    new Paragraph({ heading: D.HeadingLevel.HEADING_1, children: [new TextRun("Abstract")] }),
    P(T.abstract(summary, byRun, ngram)),
    P(T.extSummary(json("results/final/E6_gru_long.json"), json("results/interpretability.json"))),
    P(`**Keywords:** recurrent neural network, LSTM, GRU, character-level language model, bits per character, perplexity, text generation, sampling strategies.`),
  ];
}

// Static TOC; page numbers come from toc_pages.json written by the second build pass (finalize.py).
function tocPart() {
  const pf = path.join(__dirname, "toc_pages.json");
  const pages = fs.existsSync(pf) ? JSON.parse(fs.readFileSync(pf, "utf8")) : {};
  return [
    new Paragraph({ spacing: { before: 240, after: 120 }, children: [new TextRun({ text: "Table of Contents", bold: true, size: 28, font: L.HEAD_FONT, color: "1F4E79" })] }),
    ...L.tocLines(L.headings.map((h) => ({ ...h, page: pages[h.text] }))),
  ];
}

function introduction() {
  return [
    H1("1. Introduction", { newPage: true }),
    P("Language modeling — estimating the probability of the next symbol given everything that came before — is the core task behind text generation, autocompletion, speech recognition and modern large language models. Before Transformers, recurrent neural networks (RNNs) were the standard tool for this problem because they read a sequence one step at a time while carrying a hidden state that summarises the history."),
    P("This assignment addresses **Task 3 (RNN / Sequence Modeling)** of the CT1 task sheet: *train an RNN (vanilla, GRU, simple LSTM) for character-level or word-level language modeling / text generation, and compare different RNN variants, sequence length effects, sampling strategies, and perplexity.* We chose a **character-level** formulation: the vocabulary is tiny (75 symbols), there are no out-of-vocabulary words, and the model has to learn spelling, punctuation and grammar from raw characters, which makes the differences between recurrent cells clearly visible."),
    P("As data we use the six complete novels of Jane Austen from Project Gutenberg. A single-author corpus is stylistically consistent, so differences between models reflect modeling ability rather than domain shift, and the generated text can be judged easily by eye."),
    H2("1.1 Research questions"),
    numbered("**RQ1 – Cell type:** how much do gated cells (GRU, LSTM) improve on a vanilla RNN at the same width and the same compute budget?"),
    numbered("**RQ2 – Sequence length:** how does the truncated back-propagation-through-time (BPTT) window of 32–256 characters affect quality and cost?"),
    numbered("**RQ3 – Capacity:** how do hidden size and number of layers trade off against accuracy and training time?"),
    numbered("**RQ4 – Baseline:** do neural models actually beat a strong smoothed character n-gram model?"),
    numbered("**RQ5 – Sampling:** how do greedy, temperature, top-k and nucleus decoding change the quality and diversity of generated text?"),
    H2("1.2 Contributions"),
    bullet("A reproducible PyTorch pipeline (data download → cleaning → training → evaluation → generation) with a single entry point and fixed seeds."),
    bullet("Ten controlled training runs (identical optimiser, schedule, data and compute per step), a Witten-Bell n-gram baseline and a seven-strategy sampling study."),
    bullet("A stateful evaluation protocol that makes models trained with different sequence lengths directly comparable."),
  ];
}

function dataSection() {
  const rows = corpus.map((r) => [r.Book, r["Gutenberg ID"], n0(r["Raw chars"]), n0(r["Clean chars"]), r["Removed (%)"], n0(r.Words), n0(r["Unique words"])]);
  const sp = csv("ngram_sparsity");
  return [
    H1("2. Dataset and Data Analysis"),
    H2("2.1 Data source"),
    P("The corpus consists of six novels by Jane Austen downloaded as UTF-8 plain text from Project Gutenberg (public domain): *Pride and Prejudice* (ID 1342), *Emma* (158), *Sense and Sensibility* (161), *Mansfield Park* (141), *Northanger Abbey* (121) and *Persuasion* (105). The download is automated and cached in `src/data.py`, so the whole study can be regenerated from scratch."),
    ...table(["Book", "ID", "Raw chars", "Clean chars", "Removed %", "Words", "Unique words"], rows,
      "Corpus statistics before and after cleaning.", [2600, 700, 1150, 1150, 950, 1050, 1250],
      { highlight: (r) => r[0] === "TOTAL" }),
    H2("2.2 Pre-processing and cleaning"),
    P("Raw Gutenberg files contain licence text, transcription markup and hard line breaks that are not part of the novels. We apply five cleaning steps:"),
    numbered("Strip the Project Gutenberg header and footer (everything outside the `*** START OF` / `*** END OF` markers).", "numbers"),
    numbered("Normalise line endings and convert typographic Unicode punctuation (curly quotes, em-dashes, ellipses, accented letters) to ASCII equivalents."),
    numbered("Remove `[Illustration: …]` blocks and `_italic_` underscores used by transcribers."),
    numbered("Un-wrap hard line breaks inside paragraphs (Gutenberg wraps every ~70 characters) while keeping blank lines as paragraph separators."),
    numbered("Remove characters that occur fewer than 20 times in the whole corpus (e.g. `£`, `æ`, `&`, `*`), which would otherwise receive unreliable estimates."),
    P(`Cleaning removed only 3.1 % of the raw characters and leaves a vocabulary of **75 characters** (52 letters, digits, punctuation, space and newline). The text is then integer-encoded with a character-to-index map.`),
    H2("2.3 Train / validation / test split"),
    P("Each novel is split **contiguously** into 90 % training, 5 % validation and 5 % test text, and the pieces are concatenated per split. This gives 3,642,687 / 202,381 / 202,384 characters. Contiguous splitting guarantees that all six novels are represented in every split while no sentence can leak from training into evaluation (a random split of overlapping windows would leak heavily). The validation set is used only for model selection (best checkpoint); the test set is touched once per model for the final numbers."),
    H2("2.4 Exploratory data analysis"),
    ...figure("eda_book_sizes", "Size of each novel after cleaning. *Emma* and *Mansfield Park* contribute the most text.", 5.6),
    P("**Character distribution.** Letters make up 78.2 % of the training characters, space and newline 18.2 %, punctuation 3.6 % and digits only 0.01 %. The character frequencies follow a steep rank–frequency curve (Figure 2). The **unigram entropy is 4.41 bits/char**: this is the score of a model that ignores context entirely and is the first reference point any language model must beat (a uniform guess over 75 symbols would cost log2 75 = 6.23 bits)."),
    ...figure("eda_char_frequency", "Left: the 40 most frequent characters in the training split. Right: rank–frequency plot on log–log axes."),
    P("**Lengths.** The mean word length is 4.3 characters, so a 128-character training window covers about 25 words. Sentences have a median length of 13 words and a mean of 18.5 words with a long tail; paragraphs have a median of 239 characters but a 95th percentile of 1,192 characters. Long-range structure therefore exists (quotation marks that must be closed, dialogue turns, names that recur within a paragraph), which motivates the sequence-length experiment."),
    ...figure("eda_length_distributions", "Distributions of word, sentence and paragraph length in the training split (dashed line = median)."),
    P("**Vocabulary of words.** The training text contains 14,006 distinct word types, of which 33 % occur only once (hapax legomena). A word-level model would have to treat many of them as unknown, whereas a character-level model can still spell and generate them. The most frequent content words are character names (Elizabeth, Emma, Fanny, Elinor, Anne …), which the model has to learn to spell and to place in plausible contexts."),
    ...figure("eda_top_words", "Most frequent words and most frequent character names in the training text."),
    P(`**Data sparsity.** Figure 5 measures how many validation character n-grams never occur in the 3.6 M-character training text. Up to n = 5 almost everything has been seen (${f2(sp[4].unseen_val_ngrams_pct)} % unseen), but the share rises to ${f2(sp[6].unseen_val_ngrams_pct)} % for 7-grams, ${f2(sp[7].unseen_val_ngrams_pct)} % for 8-grams and ${f2(sp[9].unseen_val_ngrams_pct)} % for 10-grams. Count-based models therefore cannot use contexts longer than about 6–7 characters reliably, while a recurrent model compresses an unbounded history into a dense state and can generalise to unseen contexts. This is the central motivation for the neural models.`),
    ...figure("eda_ngram_sparsity", "Percentage of validation character n-grams that never appear in the training split.", 4.8),
  ];
}

function workflow() {
  return [
    H1("3. Workflow"),
    P("Figure 6 shows the end-to-end workflow. The top row is the data pipeline, the middle row is the training/evaluation loop that is shared by every neural run, and the bottom row lists the five experiment groups that are built on top of it."),
    ...figure("workflow_diagram", "End-to-end workflow of the study.", 6.4),
    numbered("**Acquire** – download the six novels from a Project Gutenberg mirror and cache them (`data/raw`).", "numbers"),
    numbered("**Clean & split** – apply the five cleaning steps, split each novel 90/5/5 contiguously, build the 75-symbol vocabulary (`data/processed`)."),
    numbered("**Analyse** – compute corpus statistics, entropy, length distributions and n-gram sparsity (Section 2.4)."),
    numbered("**Train** – for each configuration, sample random windows of `seq_len + 1` characters, train the CharRNN for 2,500 steps, and evaluate on validation every 250 steps."),
    numbered("**Select & test** – keep the checkpoint with the lowest validation loss and evaluate it once on the test split."),
    numbered("**Compare** – fit Witten-Bell n-gram baselines on the same training text and score them on the same test text."),
    numbered("**Generate** – sample text from the best model with seven decoding strategies and score it automatically."),
    P("All runs were executed by `run_experiments.py` on a 16-core Linux server (CPU only, PyTorch 2.15). Three runs were trained in parallel with 4 threads each; the notebook `CT1_Task3_RNN_Language_Modeling.ipynb` reproduces the analysis, tables and figures from the saved results."),
  ];
}

function modeling() {
  const params = csv("model_params");
  return [
    H1("4. Modeling"),
    H2("4.1 Architecture"),
    P("All neural models are instances of one PyTorch class, `CharRNN` (Figure 7): an embedding layer maps each of the 75 characters to a 64-dimensional vector, a stack of recurrent layers processes the sequence, dropout is applied to the top hidden states, and a linear layer produces logits over the 75 characters. Only the recurrent cell and its size change between runs, so every difference in the results can be attributed to that change."),
    ...figure("model_architecture", "CharRNN architecture. The recurrent state is carried from one time step to the next (and across chunks during evaluation).", 6.2),
    H2("4.2 Recurrent cells"),
    P("**Vanilla RNN.** The hidden state is updated with a single non-linearity:"),
    equation("h_t = tanh(W_x x_t + W_h h_{t−1} + b)"),
    P("Gradients flowing back through many multiplications by W_h tend to vanish or explode, which limits how far back the model can learn dependencies."),
    P("**GRU (Cho et al., 2014).** An update gate z_t and a reset gate r_t control how much of the old state is kept:"),
    equation("z_t = σ(W_z[x_t, h_{t−1}]),   r_t = σ(W_r[x_t, h_{t−1}]),   h̃_t = tanh(W[x_t, r_t ⊙ h_{t−1}]),   h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t"),
    P("**LSTM (Hochreiter & Schmidhuber, 1997).** A separate cell state c_t is updated additively under the control of input, forget and output gates, which gives gradients a nearly linear path through time:"),
    equation("c_t = f_t ⊙ c_{t−1} + i_t ⊙ tanh(W_c[x_t, h_{t−1}]),   h_t = o_t ⊙ tanh(c_t)"),
    ...table(["Cell", "Embedding", "Recurrent layers", "Output layer", "Total parameters"],
      params.map((r) => [r.Cell, n0(r.Embedding), n0(r.Recurrent), n0(r.Output), n0(r.Total)]),
      "Parameter count of the three cells at hidden size 256 and 2 layers (the E1 setting).", [1400, 1500, 2000, 1600, 2000]),
    P("At equal hidden size the GRU has three and the LSTM four weight blocks per layer, so they have about 2.8× and 3.7× the parameters of the vanilla RNN. We deliberately compare at **equal hidden size and equal compute per step** (the usual comparison in the literature); the parameter difference is reported next to every result and examined separately in E3."),
    H2("4.3 Training procedure"),
    ...table(["Setting", "Value", "Reason"], [
      ["Objective", "next-character cross-entropy", "maximum likelihood language modeling"],
      ["Optimiser", "AdamW, lr 2·10⁻³, no weight decay", "robust default for RNNs"],
      ["Schedule", "100-step linear warm-up, cosine decay to 0", "stable start, smooth convergence"],
      ["Steps", "2,500 (≈ 20.5 M characters ≈ 5.6 epochs)", "fixed compute budget for all runs"],
      ["Batch", "8,192 characters per step (batch = 8,192 / seq_len)", "equal compute per step across sequence lengths"],
      ["Gradient clipping", "global norm 1.0", "safety net against exploding gradients (see Figure 9)"],
      ["Regularisation", "dropout 0.2 between layers and before the output", "reduces over-fitting"],
      ["Model selection", "best validation loss (every 250 steps)", "early-stopping style checkpointing"],
      ["Seed", "42 (data sampling and initialisation)", "reproducibility"],
    ], "Training configuration shared by every neural run.", [1800, 3600, 3600]),
    H2("4.4 Evaluation metrics"),
    P("The models are scored by the average negative log-likelihood of the held-out text, reported as **bits per character (BPC)** and **character perplexity (PPL)**:"),
    equation("BPC = −(1/N) Σ log₂ p(x_t | x_<t),        PPL = 2^BPC = exp(mean NLL in nats)"),
    P("BPC is the average number of bits an ideal compressor using the model would need per character; lower is better. A perplexity of 3 means the model is, on average, as uncertain as if it had to choose uniformly among three characters. Evaluation is **stateful**: the validation/test text is cut into 32 parallel streams that are read in 256-character chunks while the hidden state is carried over, so every character (except the first of each stream) is predicted with its full history. This makes models trained with different BPTT lengths directly comparable."),
    H2("4.5 Experimental design"),
    ...table(["ID", "Question", "Runs (all other settings fixed)"], [
      ["E1", "Cell type", "RNN, GRU, LSTM — hidden 256, 2 layers, seq 128"],
      ["E2", "BPTT sequence length", "LSTM with seq 32, 64, 128 (= E1 LSTM), 256"],
      ["E3", "Capacity and depth", "LSTM hidden 128 / 256 / 384 (2 layers); layers 1 / 2 / 3 (hidden 256)"],
      ["E4", "Non-neural baseline", "Witten-Bell interpolated character n-gram, order 3 / 5 / 7"],
      ["E5", "Sampling strategy", "greedy, temperature 0.5 / 0.8 / 1.0 / 1.3, top-k (k = 5), nucleus (p = 0.9)"],
    ], "Experiments.", [700, 2300, 6000]),
    P("**Baseline.** The n-gram baseline uses interpolated Witten-Bell smoothing: P_k(c | h) = (C(hc) + T(h)·P_{k−1}(c | h′)) / (C(h) + T(h)), where T(h) is the number of distinct characters observed after context h and h′ drops the oldest character. It is a strong, tuning-free count-based model and tells us how much the neural networks gain from learned representations."),
  ];
}

function results() {
  const r = byRun;
  const row = (k, label) => [label, r[k]["Params (M)"], f3(r[k]["Train loss"] / Math.LN2), f3(r[k]["Val BPC"]), f3(r[k]["Test BPC"]), f3(r[k]["Test PPL"]), r[k]["Train time (min)"]];
  const H = ["Model", "Params (M)", "Train BPC*", "Val BPC", "Test BPC", "Test PPL", "Time (min)"];
  const W = [2300, 1050, 1150, 1050, 1050, 1050, 1050];
  const best = summary.best_model;
  const hl = (rr) => rr[0].includes("(best)");
  const lab = (k, s) => (k === best ? `${s} (best)` : s);
  const sRows = samp.rows.map((s) => [s.strategy, f3(s.valid_word_rate), f3(s.distinct_1), f3(s.distinct_2), f3(s.repeat_4gram_rate), f3(s.self_nll_bpc)]);
  const ref = samp.reference_test_text;
  return [
    H1("5. Performance Evaluation"),
    P(`Table 5 lists every neural run and the following subsections analyse each experiment. *Train BPC (in the per-experiment tables) is the training loss averaged over the last 50 steps with dropout active, so it is slightly pessimistic compared to the evaluation numbers. All BPC values are on held-out text; the best model (selected on validation BPC) is highlighted.`),
    ...table(["Run", "Cell", "H", "L", "Seq", "Params (M)", "Val BPC", "Test BPC", "Test PPL", "Time (min)"],
      runsTab.map((x) => [lab(x.Run, x.Run), x.Cell, x.Hidden, x.Layers, x["Seq len"], x["Params (M)"], f3(x["Val BPC"]), f3(x["Test BPC"]), f3(x["Test PPL"]), x["Train time (min)"]]),
      "All ten neural runs (2,500 steps each, identical data and optimiser).", [2000, 700, 550, 450, 550, 1000, 900, 900, 900, 900], { highlight: hl, size: 17 }),

    H2("5.1 E1 — Vanilla RNN vs GRU vs LSTM"),
    ...table(H, [row("E1_rnn", lab("E1_rnn", "Vanilla RNN")), row("E1_gru", lab("E1_gru", "GRU")), row("E1_lstm", lab("E1_lstm", "LSTM"))],
      "E1 results (hidden 256, 2 layers, sequence length 128).", W, { highlight: hl }),
    ...figure("e1_learning_curves", "E1 learning curves: training loss (left, with dropout) and validation loss (right) in bits per character."),
    ...T.e1(r, ngram).map((t) => P(t)),
    ...figure("e1_grad_norms", "Global gradient norm before clipping (log scale) for the three cells.", 4.8),
    ...T.e1grad(r).map((t) => P(t)),

    H2("5.2 E2 — Effect of the BPTT sequence length"),
    ...table(["Sequence length", "Batch", "Val BPC", "Test BPC", "Test PPL", "Time (min)"],
      ["E2_lstm_seq32", "E2_lstm_seq64", "E1_lstm", "E2_lstm_seq256"].map((k) => [lab(k, String(r[k]["Seq len"])), r[k].Batch, f3(r[k]["Val BPC"]), f3(r[k]["Test BPC"]), f3(r[k]["Test PPL"]), r[k]["Train time (min)"]]),
      "E2 results (LSTM, hidden 256, 2 layers; 8,192 characters per batch).", [2000, 1200, 1400, 1400, 1400, 1400], { highlight: hl }),
    ...figure("e2_learning_curves", "E2 learning curves for different BPTT lengths."),
    ...T.e2(r).map((t) => P(t)),

    H2("5.3 E3 — Capacity and depth"),
    ...table(H, [row("E3_lstm_h128", lab("E3_lstm_h128", "LSTM h128 × 2")), row("E1_lstm", lab("E1_lstm", "LSTM h256 × 2")), row("E3_lstm_h384", lab("E3_lstm_h384", "LSTM h384 × 2")),
      row("E3_lstm_L1", lab("E3_lstm_L1", "LSTM h256 × 1")), row("E3_lstm_L3", lab("E3_lstm_L3", "LSTM h256 × 3"))],
      "E3 results: hidden size (top three rows) and depth (last two rows plus h256 × 2).", W, { highlight: hl }),
    ...figure("e3_learning_curves", "E3 learning curves for different capacities."),
    ...T.e3(r).map((t) => P(t)),
    ...figure("tradeoff", "Test BPC against parameter count (left) and training time (right) for all runs. Colour = cell type."),
    ...T.tradeoff(r).map((t) => P(t)),

    H2("5.4 E4 — Comparison with the n-gram baseline"),
    ...table(["Model", "Val BPC", "Test BPC", "Test PPL"], [
      ["Unigram (context-free) entropy", "—", f3(summary.unigram_entropy), f3(2 ** summary.unigram_entropy)],
      ...ngram.map((g) => [g.model, f3(g.val_bpc), f3(g.test_bpc), f3(g.test_ppl)]),
      ["Vanilla RNN (256 × 2)", f3(r.E1_rnn["Val BPC"]), f3(r.E1_rnn["Test BPC"]), f3(r.E1_rnn["Test PPL"])],
      ["GRU (256 × 2)", f3(r.E1_gru["Val BPC"]), f3(r.E1_gru["Test BPC"]), f3(r.E1_gru["Test PPL"])],
      ["LSTM (256 × 2)", f3(r.E1_lstm["Val BPC"]), f3(r.E1_lstm["Test BPC"]), f3(r.E1_lstm["Test PPL"])],
      [`Best model: ${best} (best)`, f3(r[best]["Val BPC"]), f3(r[best]["Test BPC"]), f3(r[best]["Test PPL"])],
    ], "Neural models versus count-based baselines on the same test text.", [4000, 1600, 1600, 1600], { highlight: hl }),
    ...figure("e1_bpc_bar", "Validation and test BPC of the three cells; the dashed line marks the best n-gram model.", 4.9),
    ...T.e4(r, ngram, summary).map((t) => P(t)),

    H2("5.5 E5 — Sampling strategies"),
    P(`Text was generated from the best model (${best}) for four prompts × three random seeds × 800 characters per strategy and scored automatically. *Valid-word rate* is the share of generated words that exist in the training vocabulary (spelling quality), *distinct-n* is the ratio of unique word n-grams (diversity), *repeated 4-grams* measures looping, and *self-NLL* is how unlikely the sample is under the model itself. For reference, real held-out Austen text scores valid-word ${f3(ref.valid_word_rate)}, distinct-1 ${f3(ref.distinct_1)}, distinct-2 ${f3(ref.distinct_2)} and repeated 4-grams ${f3(ref.repeat_4gram_rate)} (computed on 20,000 test characters, so its distinct-n values are lower simply because the text is longer).`),
    ...table(["Strategy", "Valid-word", "Distinct-1", "Distinct-2", "Repeat 4-gram", "Self-NLL (BPC)"], sRows,
      "Automatic quality metrics of generated text (mean over 12 samples per strategy).", [1900, 1400, 1400, 1400, 1450, 1450]),
    ...figure("e5_sampling_metrics", "Spelling quality, diversity and repetition for each decoding strategy (dashed = real text)."),
    ...T.e5(samp).map((t) => P(t)),
    H3("Generated samples (prompt: “It is a truth universally acknowledged”)"),
    ...["greedy", "temp_0.5", "temp_0.8", "temp_1.3", "nucleus_0.9"].flatMap((k) => [
      new Paragraph({ spacing: { before: 100, after: 40 }, keepNext: true, children: [new TextRun({ text: k, bold: true, font: L.HEAD_FONT, size: 20, color: "2E75B6" })] }),
      new Paragraph({ spacing: { after: 100, line: 260 }, indent: { left: 360, right: 360 }, shading: { fill: "F5F7FA", type: ShadingType.CLEAR, color: "auto" },
        children: [new TextRun({ text: samp.examples[k].slice(0, 520).replace(/\s+/g, " ") + " …", font: "Courier New", size: 17 })] }),
    ]),
    ...extensions(),
  ];
}

// E6 (longer training), interpretability and the browser demo.
function extensions() {
  const fin = json("results/final/E6_gru_long.json");
  const it = json("results/interpretability.json");
  const probes = csv("interp_probes");
  const g = ngram.reduce((a, b) => (a.test_bpc < b.test_bpc ? a : b));
  return [
    H2("5.6 E6 — Final model: training the best GRU longer"),
    ...T.e6(byRun, fin, ngram).slice(0, 2).map((t) => P(t)),
    ...table(["Model", "Training steps", "Val BPC", "Test BPC", "Test PPL", "Time (min)"], [
      ["Best n-gram (Witten-Bell 5-gram)", "—", f3(g.val_bpc), f3(g.test_bpc), f3(g.test_ppl), "—"],
      ["E1: GRU 256 × 2", "2,500", f3(byRun.E1_gru["Val BPC"]), f3(byRun.E1_gru["Test BPC"]), f3(byRun.E1_gru["Test PPL"]), byRun.E1_gru["Train time (min)"]],
      ["E6: GRU 256 × 2 (final model)", fin.config.steps.toLocaleString("en-US"), f3(fin.best_val_bpc), f3(fin.test_bpc), f3(fin.test_ppl), (fin.train_time_s / 60).toFixed(1)],
    ], "E6: the best configuration trained longer, compared with the E1 run and the best baseline.", [3300, 1300, 1100, 1100, 1100, 1100],
    { highlight: (row) => row[0].startsWith("E6") }),
    ...figure("e6_final_model", "E6 validation curve (GRU trained longer) compared with the 2,500-step GRU of E1; the dotted line is the best n-gram test score.", 5.4),
    P(T.e6(byRun, fin, ngram)[2]),

    H2("5.7 What did the GRU learn? Interpretability"),
    ...T.interp(it).slice(0, 1).map((t) => P(t)),
    ...table(["Property", "Probe on hidden state", "Probe on current char", "Majority / zero", "Metric"],
      probes.map((p) => [p.property, Number(p["hidden state"]).toFixed(3), Number(p["current char only"]).toFixed(3), Number(p["majority baseline"]).toFixed(3), p.type]),
      "Linear probes on the GRU hidden state (held-out 30 % of 60,000 test characters).", [3000, 1700, 1700, 1500, 1100],
      { highlight: (row) => Number(row[1]) - Number(row[2]) > 0.1 }),
    ...figure("interp_unit_activations", "Activation of two individual hidden units over a passage of *Pride and Prejudice* (test split). Top: a quotation-tracking unit. Bottom: a word-boundary unit. Red = high, blue = low.", 6.3),
    ...T.interp(it).slice(1).map((t) => P(t)),
    ...figure("interp_probes", "Left: linear-probe accuracy for binary properties. Right: strongest single-unit correlation per property."),

    H2("5.8 Interactive browser demo"),
    ...T.demo().map((t) => P(t)),
    ...figure("demo_screenshot", "The browser demo generating text from the prompt “My dear Mr. Bennet,” with nucleus sampling.", 4.6),
  ];
}

function discussion() {
  return [
    H1("6. Discussion and Limitations"),
    ...T.discussion(byRun, ngram, summary).map((t) => P(t)),
    H2("6.1 Limitations and threats to validity"),
    ...T.limitations().map((t) => bullet(t)),
  ];
}

function summarySection() {
  return [
    H1("7. Summary and Future Work"),
    ...T.summary(byRun, ngram, summary).map((t) => P(t)),
    P(T.extSummary(json("results/final/E6_gru_long.json"), json("results/interpretability.json"))),
    H2("7.1 Answers to the research questions"),
    ...T.answers(byRun, ngram, summary).map((t) => bullet(t)),
    H2("7.2 Future work"),
    ...T.future().map((t) => bullet(t)),
  ];
}

function codebase() {
  return [
    H1("8. Codebase and Reproducibility"),
    P(`**Repository:** ${META.repo}`),
    ...table(["Path", "Content"], [
      ["CT1_Task3_RNN_Language_Modeling.ipynb", "executed notebook: data analysis, models, all results, figures and generation"],
      ["run_experiments.py", "runs all ten training runs, the n-gram baseline and the sampling study (resumable)"],
      ["src/data.py", "download, cleaning, vocabulary, contiguous split, batching"],
      ["src/models.py", "CharRNN (RNN / GRU / LSTM)"],
      ["src/train.py", "training loop, stateful evaluation, metrics"],
      ["src/ngram.py", "Witten-Bell character n-gram baseline"],
      ["src/generate.py", "greedy / temperature / top-k / nucleus sampling and text metrics"],
      ["src/analysis.py", "EDA statistics and every figure / table of this report"],
      ["results/", "per-run JSON (config, history, metrics), logs, tables (CSV), figures (PNG), sampling results"],
      ["checkpoints/", "best checkpoint of every run (PyTorch state dict)"],
      ["run_final_model.py", "E6: trains the best GRU for longer (results in results/final/)"],
      ["src/interpret.py", "interpretability: unit correlations, linear probes, activation plots"],
      ["demo/", "browser demo (index.html, JavaScript GRU, weight export, PyTorch check)"],
      ["tests/", "16 unit tests (python -m unittest discover -s tests)"],
      ["report/", "this report (DOCX + PDF) and the script that builds it"],
    ], "Repository structure.", [3400, 5600], { size: 18 }),
    P("**Reproduce:** `pip install -r requirements.txt` → `python run_experiments.py --workers 3 --threads 4` (≈ 3 hours on a 16-core CPU; downloads the data automatically) → open and run the notebook. All random seeds are fixed (42); results on different hardware may differ in the third decimal because of non-deterministic multi-threaded CPU kernels."),
    H2("8.1 Use of AI tools"),
    P("An AI coding assistant was used to help draft and debug parts of the code and to format this report. All experiments were run by the group, every number in this report is produced by the code in the repository, and all members reviewed and can explain the implementation."),
  ];
}

function references() {
  const refs = [
    "S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.",
    "K. Cho et al., “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” in *Proc. EMNLP*, 2014.",
    "J. L. Elman, “Finding Structure in Time,” *Cognitive Science*, vol. 14, no. 2, pp. 179–211, 1990.",
    "Y. Bengio, P. Simard and P. Frasconi, “Learning Long-Term Dependencies with Gradient Descent is Difficult,” *IEEE Trans. Neural Networks*, vol. 5, no. 2, 1994.",
    "R. Pascanu, T. Mikolov and Y. Bengio, “On the Difficulty of Training Recurrent Neural Networks,” in *Proc. ICML*, 2013.",
    "A. Karpathy, J. Johnson and L. Fei-Fei, “Visualizing and Understanding Recurrent Networks,” *arXiv:1506.02078*, 2015.",
    "J. Chung, C. Gulcehre, K. Cho and Y. Bengio, “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling,” *arXiv:1412.3555*, 2014.",
    "A. Graves, “Generating Sequences With Recurrent Neural Networks,” *arXiv:1308.0850*, 2013.",
    "I. H. Witten and T. C. Bell, “The Zero-Frequency Problem: Estimating the Probabilities of Novel Events in Adaptive Text Compression,” *IEEE Trans. Information Theory*, vol. 37, no. 4, 1991.",
    "A. Holtzman, J. Buys, L. Du, M. Forbes and Y. Choi, “The Curious Case of Neural Text Degeneration,” in *Proc. ICLR*, 2020.",
    "I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in *Proc. ICLR*, 2019.",
    "Project Gutenberg, “Works by Jane Austen,” https://www.gutenberg.org/ebooks/author/68 (accessed 17 Nov 2026).",
    "A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in *Proc. NeurIPS*, 2019.",
  ];
  return [H1("References"), ...refs.map((t) => new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(t), spacing: { after: 80, line: 276 } }))];
}

// ------------------------------------------------------------------ assemble
const abstractBlock = abstractPart();
const body = [...introduction(), ...dataSection(), ...workflow(), ...modeling(), ...results(),
  ...discussion(), ...summarySection(), ...codebase(), ...references()];
fs.writeFileSync(path.join(__dirname, "toc_headings.json"), JSON.stringify(L.headings.map((h) => h.text)));
const doc = new Document({
  creator: META.group, title: META.title, description: `${META.course} — ${META.assignment}`,
  styles: L.STYLES, numbering: L.NUMBERING, features: { updateFields: true },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 0, bottom: 0, left: 0, right: 0, header: 0, footer: 0 } } },
      children: cover() },
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 }, pageNumbers: { start: 1 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 2 } },
        children: [new TextRun({ text: `${META.course} · ${META.assignment} · Group ${META.group}`, size: 17, color: "7F7F7F", font: L.HEAD_FONT })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 18, color: "7F7F7F" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "7F7F7F" })] })] }) },
      children: [...abstractBlock, ...tocPart(), ...body],
    },
  ],
});

const out = path.join(__dirname, "CT1_Task3_Report_Recurrent_Minds.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
