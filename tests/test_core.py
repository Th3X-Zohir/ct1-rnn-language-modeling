"""Unit tests for the data pipeline, models, evaluation, n-gram baseline and sampling.

Run:  python -m unittest discover -s tests -v
The tests use small synthetic inputs only, so they run in a few seconds and need no download.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import clean, drop_rare, encode, random_batch, strip_gutenberg  # noqa: E402
from src.generate import sample, text_metrics  # noqa: E402
from src.models import CharRNN, n_params  # noqa: E402
from src.ngram import WittenBellLM  # noqa: E402
from src.train import evaluate  # noqa: E402

TEXT = ("It is a truth universally acknowledged, that a single man in possession of a good fortune, "
        "must be in want of a wife. ") * 20


class TestData(unittest.TestCase):
    def test_strip_gutenberg_keeps_only_body(self):
        raw = "licence text\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBODY\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nmore"
        self.assertEqual(strip_gutenberg(raw).strip(), "BODY")

    def test_clean_normalises_quotes_and_unwraps_lines(self):
        out = clean("“Hello,” said\nshe.\r\n\r\n_Next_ paragraph.")
        self.assertEqual(out, '"Hello," said she.\n\nNext paragraph.')

    def test_drop_rare_removes_infrequent_characters(self):
        text, removed = drop_rare("aaaa£", min_count=2)
        self.assertEqual(text, "aaaa"); self.assertIn("£", removed)

    def test_drop_rare_never_removes_book_separator(self):
        text, _ = drop_rare("ab\x00cd", min_count=5)
        self.assertIn("\x00", text)

    def test_encode_round_trip(self):
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        itos = {i: c for c, i in stoi.items()}
        ids = encode(TEXT, stoi)
        self.assertEqual("".join(itos[int(i)] for i in ids), TEXT)

    def test_random_batch_targets_are_shifted_inputs(self):
        data = torch.arange(1000)
        x, y = random_batch(data, batch=4, seq_len=16, gen=torch.Generator().manual_seed(0))
        self.assertEqual(tuple(x.shape), (4, 16))
        self.assertTrue(torch.equal(y, x + 1))


class TestModels(unittest.TestCase):
    def test_output_shapes_for_every_cell(self):
        for cell in ("rnn", "gru", "lstm"):
            m = CharRNN(30, cell, emb_dim=8, hidden=16, layers=2)
            logits, state = m(torch.randint(0, 30, (3, 7)))
            self.assertEqual(tuple(logits.shape), (3, 7, 30))
            h = state[0] if cell == "lstm" else state
            self.assertEqual(tuple(h.shape), (2, 3, 16))

    def test_parameter_ordering_rnn_gru_lstm(self):
        p = {c: n_params(CharRNN(75, c)) for c in ("rnn", "gru", "lstm")}
        self.assertLess(p["rnn"], p["gru"]); self.assertLess(p["gru"], p["lstm"])

    def test_stateful_evaluation_matches_single_pass(self):
        torch.manual_seed(0)
        m = CharRNN(10, "gru", emb_dim=8, hidden=16, layers=1, dropout=0.0)
        data = torch.randint(0, 10, (513,))
        chunked = evaluate(m, data, n_streams=1, chunk=64)
        one_pass = evaluate(m, data, n_streams=1, chunk=10_000)
        self.assertAlmostEqual(chunked, one_pass, places=4)

    def test_untrained_model_is_close_to_uniform(self):
        torch.manual_seed(0)
        m = CharRNN(20, "lstm", emb_dim=8, hidden=16, layers=1, dropout=0.0)
        nll = evaluate(m, torch.randint(0, 20, (2001,)), n_streams=4, chunk=100)
        self.assertLess(abs(nll - math.log(20)), 0.2)


class TestNgram(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        lm = WittenBellLM(3).fit(TEXT)
        for h in ("", "a", "th", "zzz"):
            total = sum(lm.prob(h, c) for c in set(TEXT))
            self.assertAlmostEqual(total, 1.0, delta=1e-6)

    def test_higher_order_fits_training_text_better(self):
        nll = [WittenBellLM(k).fit(TEXT).nll(TEXT[:500]) for k in (1, 3)]
        self.assertLess(nll[1], nll[0])


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        self.itos = {i: c for c, i in self.stoi.items()}
        torch.manual_seed(0)
        self.m = CharRNN(len(self.stoi), "gru", emb_dim=8, hidden=16, layers=1)

    def test_greedy_is_deterministic(self):
        a, _ = sample(self.m, self.stoi, self.itos, "It", n=30, strategy="greedy", seed=1)
        b, _ = sample(self.m, self.stoi, self.itos, "It", n=30, strategy="greedy", seed=2)
        self.assertEqual(a, b)

    def test_sampling_respects_length_and_prompt(self):
        txt, nll = sample(self.m, self.stoi, self.itos, "It is", n=40, temperature=0.8, top_p=0.9, seed=0)
        self.assertTrue(txt.startswith("It is")); self.assertEqual(len(txt), 45); self.assertGreater(nll, 0)

    def test_top_k_one_equals_greedy(self):
        a, _ = sample(self.m, self.stoi, self.itos, "It", n=25, strategy="greedy")
        b, _ = sample(self.m, self.stoi, self.itos, "It", n=25, top_k=1, seed=3)
        self.assertEqual(a, b)

    def test_text_metrics(self):
        m = text_metrics("the cat the cat the cat", {"the", "cat"})
        self.assertEqual(m["valid_word_rate"], 1.0)
        self.assertAlmostEqual(m["distinct_1"], 2 / 6)
        self.assertGreater(m["repeat_4gram_rate"], 0)


if __name__ == "__main__":
    unittest.main()
