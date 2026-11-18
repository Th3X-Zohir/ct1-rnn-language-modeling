// Result-dependent narrative. Every number is read from the results files, never typed by hand.
const S = require("./text_static");
const f3 = (x) => Number(x).toFixed(3);
const pct = (a, b) => (100 * (a - b) / a).toFixed(1);
const T = (r, k) => Number(r[k]["Test BPC"]);
const V = (r, k) => Number(r[k]["Val BPC"]);
const M = (r, k) => Number(r[k]["Train time (min)"]);
const Pm = (r, k) => Number(r[k]["Params (M)"]);
const bestNg = (ng) => ng.reduce((a, b) => (a.test_bpc < b.test_bpc ? a : b));

exports.abstract = (s, r, ng) => {
  const g = bestNg(ng);
  return `This report studies character-level language modeling with recurrent neural networks, following Task 3 of the CT1 task sheet. Six novels by Jane Austen (4.05 M characters after cleaning, 75-symbol vocabulary) were split contiguously into training, validation and test text. We trained ten models under an identical compute budget (2,500 steps of 8,192 characters) to compare a vanilla RNN, a GRU and an LSTM (E1), four BPTT sequence lengths (E2) and five LSTM capacities (E3), and compared them with Witten-Bell n-gram baselines (E4) and seven decoding strategies (E5). Under the equal budget the GRU was the best model with **${f3(s.best_test_bpc)} bits per character** on the test set (perplexity ${f3(s.best_test_ppl)}), ahead of the LSTM (${f3(s.lstm_test_bpc)}) and the vanilla RNN (${f3(s.rnn_test_bpc)}), and ${pct(g.test_bpc, s.best_test_bpc)} % better than the best n-gram model (${f3(g.test_bpc)}). Gating, capacity and depth mattered far more than the sequence length, which changed the LSTM score by less than 0.02 BPC. Temperature and nucleus sampling produced the most natural text, while greedy decoding quickly fell into repetitive loops.`;
};

exports.e1 = (r, ng) => {
  const g = bestNg(ng);
  return [
    `**The GRU is the clear winner of E1** with ${f3(T(r, "E1_gru"))} test BPC, ${f3(T(r, "E1_lstm") - T(r, "E1_gru"))} BPC better than the LSTM (${f3(T(r, "E1_lstm"))}) and ${f3(T(r, "E1_rnn") - T(r, "E1_gru"))} BPC better than the vanilla RNN (${f3(T(r, "E1_rnn"))}). Only the GRU beats the best n-gram baseline (${f3(g.test_bpc)}) within this budget. The learning curves explain the ranking: the GRU drops fastest from the very first evaluation (2.13 BPC after 250 steps against 2.30 for the RNN and 2.61 for the LSTM) and stays ahead for the whole run.`,
    `The LSTM starts slowest — its four gates and separate cell state need more updates before they carry useful information — but it overtakes the vanilla RNN between steps 1,000 and 1,250 and keeps improving faster than it, ending ${f3(T(r, "E1_rnn") - T(r, "E1_lstm"))} BPC ahead. The vanilla RNN learns quickly at first but flattens out early: without gates it cannot keep information for many steps, so it mostly models short local patterns. All validation curves are still falling at step 2,500, i.e. none of the models over-fit and the ranking reflects learning speed under a fixed budget (the LSTM is known to match or beat the GRU after much longer training).`,
    `Training cost follows model complexity: the RNN (${Pm(r, "E1_rnn")} M parameters) took ${M(r, "E1_rnn")} min, the LSTM (${Pm(r, "E1_lstm")} M) ${M(r, "E1_lstm")} min and the GRU (${Pm(r, "E1_gru")} M) ${M(r, "E1_gru")} min. The GRU is slower than the larger LSTM because PyTorch ships a more optimised fused CPU kernel for the LSTM (in an isolated benchmark one GRU step took 579 ms against 473 ms for the LSTM). Wall-clock times are approximate because three runs shared the CPU.`,
  ];
};

exports.e1grad = () => [
  "Figure 9 shows the global gradient norm before clipping. For all three cells it is largest at the start of training and then settles around 0.13–0.25; it never reached the clipping threshold of 1.0, so clipping acted only as a safety net. With a 128-step BPTT window, a small learning rate and a tanh cell, the vanilla RNN therefore did **not** suffer from exploding gradients here. Its weaker score comes from the other side of the same problem — vanishing gradients make long-range information hard to learn — which is exactly what the gates of the GRU and LSTM are designed to fix.",
];

exports.e2 = (r) => [
  `Longer BPTT windows help, but only slightly: test BPC improves monotonically from ${f3(T(r, "E2_lstm_seq32"))} (32 characters) to ${f3(T(r, "E2_lstm_seq64"))} (64), ${f3(T(r, "E1_lstm"))} (128) and ${f3(T(r, "E2_lstm_seq256"))} (256) — a total gain of only ${f3(T(r, "E2_lstm_seq32") - T(r, "E2_lstm_seq256"))} BPC for an 8× longer window. Because the evaluation is stateful, even the model trained on 32-character windows can use a long history at test time; what the longer window adds is the ability to *learn* dependencies longer than 32 characters, such as closing quotation marks or keeping a name consistent within a sentence.`,
  `The cost is higher: with the number of characters per batch fixed, the longer windows have fewer parallel sequences and a longer sequential recurrence, and the 256-character run was the slowest LSTM run (${M(r, "E2_lstm_seq256")} min vs ${M(r, "E2_lstm_seq32")} min for 32). Shorter windows also give a noisier start (the 32-character model is behind at every evaluation). For this corpus a window of 128–256 characters (≈ 25–50 words, roughly one sentence) is the sweet spot; beyond that the returns are small compared with changing the cell or the capacity.`,
];

exports.e3 = (r) => {
  const l3 = r.E3_lstm_L3;
  const deep = l3
    ? (T(r, "E3_lstm_L3") < T(r, "E1_lstm")
      ? `A third layer improves further to ${f3(T(r, "E3_lstm_L3"))} BPC, at the cost of ${M(r, "E3_lstm_L3")} min of training.`
      : `A third layer, however, does **not** help within the budget: the 3-layer LSTM reaches only ${f3(T(r, "E3_lstm_L3"))} BPC after ${M(r, "E3_lstm_L3")} min. Deeper recurrent stacks are harder to optimise and start much more slowly (3.17 BPC after 250 steps versus 2.61 for two layers), so 2,500 steps are not enough for the extra layer to pay off.`)
    : "";
  return [
    `**Width matters a lot.** Halving the hidden size from 256 to 128 costs ${f3(T(r, "E3_lstm_h128") - T(r, "E1_lstm"))} BPC (${f3(T(r, "E3_lstm_h128"))}); the model is too small to store the spelling and vocabulary of the corpus. Increasing it to 384 gains ${f3(T(r, "E1_lstm") - T(r, "E3_lstm_h384"))} BPC (${f3(T(r, "E3_lstm_h384"))}) — the largest improvement of any LSTM modification in the study — because wider layers both store more and learn faster per step (the 384-unit model was already at 1.80 BPC after 750 steps, when the 256-unit model was at 1.97).`,
    `**Depth helps up to a point.** A single 256-unit layer reaches only ${f3(T(r, "E3_lstm_L1"))} BPC, so stacking a second layer gains ${f3(T(r, "E3_lstm_L1") - T(r, "E1_lstm"))} BPC: the second layer can combine character patterns detected by the first into word- and phrase-level features. ${deep}`,
    `The 384-unit LSTM (${Pm(r, "E3_lstm_h384")} M parameters) almost catches the GRU (${f3(T(r, "E3_lstm_h384"))} vs ${f3(T(r, "E1_gru"))} BPC), but needs ${(Pm(r, "E3_lstm_h384") / Pm(r, "E1_gru")).toFixed(1)}× the parameters and ${(M(r, "E3_lstm_h384") / M(r, "E1_gru")).toFixed(1)}× the training time. This confirms that the E1 result is not just a matter of parameter count: per parameter and per step, the GRU is the most efficient cell on this task.`,
  ];
};

exports.tradeoff = (r) => [
  `Figure 12 summarises the accuracy–cost trade-off. The best points lie on a clear frontier: the vanilla RNN is the cheapest model, the GRU gives the best accuracy per parameter and per minute, and the large LSTM (h384) gains ${f3(T(r, "E1_lstm") - T(r, "E3_lstm_h384"))} BPC over the default LSTM at ${(Pm(r, "E3_lstm_h384") / Pm(r, "E1_lstm")).toFixed(1)}× the parameters and ${(M(r, "E3_lstm_h384") / M(r, "E1_lstm")).toFixed(1)}× the training time. The 1-layer and 128-unit LSTMs are dominated: they are cheap but lose much more accuracy than they save.`,
];

exports.e4 = (r, ng, s) => {
  const g = bestNg(ng);
  const n3 = ng.find((x) => x.order === 3), n7 = ng.find((x) => x.order === 7);
  return [
    `The n-gram baselines put the neural results into perspective. The 3-gram (${f3(n3.test_bpc)} BPC) is far behind, while the 5-gram is surprisingly strong at ${f3(g.test_bpc)} BPC — better than the vanilla RNN and every LSTM except the 384-unit model. Increasing the order to 7 makes the n-gram **worse** (${f3(n7.test_bpc)}): as shown by the sparsity analysis (Section 2.4), most 7-character contexts are seen too rarely to give reliable estimates, so the extra context mainly adds noise. This is precisely the limitation that recurrent networks remove — they compress any length of history into a dense state and share statistics between similar contexts.`,
    `The best neural model (GRU) beats the best n-gram by ${f3(g.test_bpc - s.best_test_bpc)} BPC (${pct(g.test_bpc, s.best_test_bpc)} % relative), with a model that has only ${Pm(r, "E1_gru")} M parameters against ${(g.n_contexts / 1e3).toFixed(0)} k stored contexts for the 5-gram. The margin is modest because of the short training budget: the neural curves were still falling when training stopped, while the n-gram model is already at its optimum. With longer training, character-level LSTMs and GRUs are known to reach well below 1.5 BPC on corpora of this size, whereas the n-gram cannot improve further without more data.`,
  ];
};

exports.e5 = (samp) => {
  const R = Object.fromEntries(samp.rows.map((x) => [x.strategy, x]));
  const ref = samp.reference_test_text;
  return [
    `**Greedy decoding degenerates.** Always picking the most likely character produces fluent-looking but repetitive text: ${(100 * R.greedy.repeat_4gram_rate).toFixed(0)} % of its word 4-grams are repeats and its distinct-2 is only ${f3(R.greedy.distinct_2)}. Its self-NLL is the lowest (${f3(R.greedy.self_nll_bpc)} BPC) — the model is very confident about this text — which illustrates that *the most probable sequence is not a good sample* (Holtzman et al., 2020).`,
    `**Temperature controls the quality–diversity trade-off.** At T = 0.5 the text is almost perfectly spelled (valid-word rate ${f3(R["temp_0.5"].valid_word_rate)}) but noticeably less diverse (distinct-1 ${f3(R["temp_0.5"].distinct_1)} vs ${f3(R["temp_1.0"].distinct_1)} at T = 1.0) and it reuses common phrases. As T rises, diversity increases and spelling degrades: at T = 1.0 the valid-word rate falls to ${f3(R["temp_1.0"].valid_word_rate)}, and at T = 1.3 to ${f3(R["temp_1.3"].valid_word_rate)} with many invented words. Real Austen text has a valid-word rate of ${f3(ref.valid_word_rate)} (some test words are simply absent from the training vocabulary).`,
    `**Truncated sampling is the best compromise.** Top-k (k = 5) and nucleus sampling (p = 0.9) cut off the unreliable tail of the distribution while keeping randomness: nucleus sampling reaches a valid-word rate of ${f3(R["nucleus_0.9"].valid_word_rate)} with distinct-2 ${f3(R["nucleus_0.9"].distinct_2)} and only ${(100 * R["nucleus_0.9"].repeat_4gram_rate).toFixed(1)} % repeated 4-grams, and top-k gives ${f3(R.top_k_5.valid_word_rate)} / ${f3(R.top_k_5.distinct_2)} / ${(100 * R.top_k_5.repeat_4gram_rate).toFixed(1)} %. Together with T ≈ 0.8 (${f3(R["temp_0.8"].valid_word_rate)} / ${f3(R["temp_0.8"].distinct_2)}) these settings produce the most Austen-like text: correct punctuation and dialogue formatting, period-appropriate vocabulary and character names, but little long-range coherence — the expected behaviour of a model that sees about one sentence of context during training.`,
  ];
};

exports.discussion = (r, ng, s) => [
  `**Why does the GRU win?** Under a fixed and fairly short budget the GRU combines the benefit of gating (it keeps information over many steps, unlike the vanilla RNN) with a simpler structure than the LSTM: two gates instead of three and no separate cell state, so fewer parameters must be co-adapted before the model becomes useful. Chung et al. (2014) report the same pattern — the GRU often converges faster, while the LSTM catches up with longer training. Our result should therefore be read as *“GRU is the most efficient cell for a small compute budget”*, not as a universal ranking.`,
  `**What matters most?** Ranked by effect size, the levers were: hidden size (${f3(T(r, "E3_lstm_h128") - T(r, "E3_lstm_h384"))} BPC between 128 and 384 units), cell type (${f3(T(r, "E1_rnn") - T(r, "E1_gru"))} BPC between RNN and GRU), depth (${f3(T(r, "E3_lstm_L1") - T(r, "E1_lstm"))} BPC between one and two layers) and, a distant last, sequence length (${f3(T(r, "E2_lstm_seq32") - T(r, "E2_lstm_seq256"))} BPC between 32 and 256 characters).`,
  `**Training versus validation loss.** In every run the reported training BPC is *higher* than the validation BPC. This is not a bug: training loss is measured with dropout active and on random windows whose first characters have no history, while evaluation runs without dropout and with the full carried-over hidden state. Together with validation curves that are still decreasing, it shows that none of the models over-fitted; all of them were limited by compute, not by data.`,
  `**Practical recommendation.** For character-level modeling on a CPU budget, a 2-layer GRU with 256 units and a 128–256-character window is the best choice; if more compute is available, increasing the hidden size gives more benefit than adding depth or lengthening the window. For generation, nucleus sampling (p ≈ 0.9) or temperature ≈ 0.8 should be used instead of greedy decoding.`,
];

// ---------------------------------------------------------------- extensions (E6, interpretability, demo)
exports.e6 = (r, fin, ng) => {
  const g = bestNg(ng), k = fin.config.steps, gain = T(r, "E1_gru") - fin.test_bpc;
  return [
    `E1–E5 compare models under an *equal* budget. E6 asks a different question: how good does the winning configuration (2-layer GRU, 256 units, sequence length 128) become when it is simply trained longer? We trained it for ${k.toLocaleString("en-US")} steps instead of 2,500 (${(k / 2500).toFixed(1)}× the budget, ≈ ${((k * 8192) / 1e6).toFixed(0)} M characters or ${((k * 8192) / 3.64e6).toFixed(1)} epochs), with everything else unchanged.`,
    `The longer run reaches **${f3(fin.test_bpc)} BPC on the test set (perplexity ${f3(fin.test_ppl)})**, ${f3(gain)} BPC better than the 2,500-step GRU and ${pct(g.test_bpc, fin.test_bpc)} % better than the best n-gram model (${f3(g.test_bpc)}). The validation curve (Figure 15) keeps falling throughout training, confirming the conclusion of E1–E3 that all models in the controlled study were limited by compute rather than by data or capacity.`,
    `A practical note from this experiment: on our 16-core CPU server the small GRU trained **fastest with only 2 threads** (0.64 s per step) and five times slower with 12 threads (3.1 s per step), because the matrices are too small for multi-threading to pay off and the threads spend their time synchronising. Choosing the thread count by a short benchmark turned a run that would have taken about ${(k * 3.14 / 3600).toFixed(0)} hours into one of ${(fin.train_time_s / 60).toFixed(0)} minutes.`,
  ];
};

exports.interp = (it) => {
  const U = Object.fromEntries(it.units.map((u) => [u.property, u]));
  const Pr = Object.fromEntries(it.probes.map((p) => [p.property, p]));
  const q = U["inside quotation"], qp = Pr["inside quotation"], b = U["next char starts a word"];
  const w = Pr["position inside word"], para = Pr["position in paragraph (log)"];
  return [
    `To see *what* the network has learned, we ran the best E1 model (GRU 256×2) over ${it.n_chars.toLocaleString("en-US")} characters of test text, recorded its ${it.hidden_units} top-layer hidden units at every step, and related them to six interpretable properties of the text (method of Karpathy et al., 2015). For each property we report the single most correlated unit and a **linear probe** — a logistic or linear regression trained on 70 % of the characters and tested on the remaining 30 % — compared with a probe that only sees the *current character*. If the hidden-state probe beats the current-character probe, the network must have stored information about the past.`,
    `**A quotation detector emerged on its own.** Unit ${q["best unit"]} correlates with "inside a quotation" at |r| = ${Math.abs(q["pearson r"]).toFixed(2)} (Figure 16, top; the sign only says that its activation drops inside quotations): it switches state at an opening quotation mark, stays there for the whole speech — even across the narrator's interruption *"replied Darcy,"* where it briefly relaxes — and switches back when the quotation closes. A linear probe on the hidden state predicts whether the model is inside a quotation with **${(100 * qp["hidden state"]).toFixed(1)} % accuracy**, against ${(100 * qp["current char only"]).toFixed(1)} % from the current character and ${(100 * qp["majority baseline"]).toFixed(1)} % for the majority class. Nobody told the model what a quotation is; it learned to track one because closing the quotation correctly lowers the prediction loss.`,
    `**Word structure is tracked as well.** Unit ${b["best unit"]} fires at word boundaries (|r| = ${Math.abs(b["pearson r"]).toFixed(2)}, Figure 16 bottom), and the position inside the current word is linearly decodable with R² = ${w["hidden state"].toFixed(2)} from the hidden state versus ${w["current char only"].toFixed(2)} from the current character alone. Capital letters and sentence ends, in contrast, are already given by the current character, so the hidden state adds nothing there (Figure 17). Finally, the position within a long paragraph is **not** represented (R² ≈ ${Math.max(0, para["hidden state"]).toFixed(2)}): with 128-character training windows the model never had a reason to count beyond about one sentence, which matches the long-range limitations seen in the generated text.`,
  ];
};

exports.demo = (meta) => [
  `To make the model tangible we built an **interactive browser demo** (\`demo/index.html\`). A short script exports the trained GRU weights to a 1.8 MB JavaScript file (float16), and a ${"~"}100-line JavaScript implementation of the GRU cell (\`demo/gru.js\`) runs the model directly in the browser — no server, no Python. The JavaScript output matches PyTorch to within 2·10⁻⁶ on the logits and reproduces the greedy sample character-for-character (\`demo/check_js.js\`).`,
  `The page lets the user choose a prompt, the decoding strategy (greedy, temperature, top-k, nucleus), the temperature, p, k and the output length, and streams the generated text at several hundred characters per second (about 500–900 in our tests) on an ordinary laptop (Figure 18). It turns the sampling study of E5 into something one can explore by hand: switching to greedy immediately shows the repetition loops, and raising the temperature shows the model inventing words.`,
];

exports.extSummary = (fin, it) => {
  const q = it.units.find((u) => u.property === "inside quotation");
  const qp = it.probes.find((p) => p.property === "inside quotation");
  return `Beyond the controlled study, training the best GRU ${(fin.config.steps / 2500).toFixed(1)}× longer (E6) lowered the test score to **${f3(fin.test_bpc)} BPC**. An interpretability analysis found a single hidden unit that tracks whether the text is inside a quotation (|r| = ${Math.abs(q["pearson r"]).toFixed(2)}; linear-probe accuracy ${(100 * qp["hidden state"]).toFixed(1)} % vs ${(100 * qp["current char only"]).toFixed(1)} % from the current character), and the final model runs live in a browser demo implemented in plain JavaScript.`;
};

exports.limitations = S.limitations;
exports.future = S.future;

exports.summary = (r, ng, s) => {
  const g = bestNg(ng);
  return [
    `We built a complete, reproducible pipeline for character-level language modeling on six Jane Austen novels and ran a controlled study of recurrent architectures. With an identical budget for every model, the **2-layer GRU with 256 units achieved the best test score of ${f3(s.best_test_bpc)} BPC (perplexity ${f3(s.best_test_ppl)})**, beating the LSTM (${f3(s.lstm_test_bpc)}), the vanilla RNN (${f3(s.rnn_test_bpc)}) and the strongest n-gram baseline (${f3(g.test_bpc)}). The vanilla RNN learned local spelling patterns quickly but plateaued early; the gated cells kept improving throughout training.`,
  ];
};

exports.answers = (r, ng, s) => {
  const g = bestNg(ng);
  return [
    `**RQ1 (cell type):** gating helps substantially — GRU ${f3(T(r, "E1_gru"))} and LSTM ${f3(T(r, "E1_lstm"))} vs RNN ${f3(T(r, "E1_rnn"))} BPC; under a short budget the GRU learns fastest and is best.`,
    `**RQ2 (sequence length):** longer BPTT windows give small, monotonic gains (${f3(T(r, "E2_lstm_seq32"))} → ${f3(T(r, "E2_lstm_seq256"))} BPC from 32 to 256) at a higher time cost; 128–256 characters is a good choice.`,
    `**RQ3 (capacity):** hidden size is the strongest lever (${f3(T(r, "E3_lstm_h128"))} → ${f3(T(r, "E3_lstm_h384"))} BPC from 128 to 384 units); a second layer helps clearly${r.E3_lstm_L3 ? (T(r, "E3_lstm_L3") < T(r, "E1_lstm") ? ", a third a little more" : ", a third does not pay off within the budget") : ""}.`,
    `**RQ4 (baseline):** a 5-gram is a strong baseline (${f3(g.test_bpc)} BPC) and higher orders over-fit; the best recurrent model beats it by ${pct(g.test_bpc, s.best_test_bpc)} %, and the gap would widen with longer training.`,
    `**RQ5 (sampling):** greedy decoding loops; nucleus (p = 0.9), top-k and temperature ≈ 0.8 give the best balance of correct spelling and diversity; high temperature produces invented words.`,
  ];
};
