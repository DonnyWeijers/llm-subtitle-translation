"""Unit tests for the parts of subtrans that need no network: parsing and the free quality checks.

  python3 -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "subtrans"))
import subtrans  # noqa: E402

# Made-up dialogue of varied length, so line lengths carry a signal for the shift check
EN = [f"Line {i}: " + "word " * (2 + (i * 7) % 11) for i in range(60)]
LINES = list(enumerate(EN))


def dutch(text):
    return text.replace("Line", "Regel").replace("word", "woord")


class Parse(unittest.TestCase):
    def test_keeps_wanted_indices_only(self):
        text = 'Sure: [{"index": "1", "content": "Hallo"}, {"index": "9", "content": "extra"}, {"index": "2"}]'
        self.assertEqual(subtrans.parse(text, ["1", "2"]), {"1": "Hallo"})

    def test_garbage(self):
        self.assertEqual(subtrans.parse("no json here", ["1"]), {})


class Checks(unittest.TestCase):
    def test_good_translation_passes(self):
        res, failed = subtrans.checks(LINES, {p: dutch(t) for p, t in LINES})
        self.assertEqual(failed, [])
        self.assertEqual(res["missing"], 0)

    def test_missing_lines(self):
        tr = {p: dutch(t) for p, t in LINES if p % 10}
        self.assertIn("missing", subtrans.checks(LINES, tr)[1])

    def test_english_left(self):
        tr = {p: (t if p < 10 else dutch(t)) for p, t in LINES}
        self.assertIn("untranslated", subtrans.checks(LINES, tr)[1])

    def test_text_shifted_one_line(self):
        tr = {p: dutch(EN[p + 1]) if p + 1 < len(EN) else "" for p, _ in LINES}
        self.assertIn("shifted", subtrans.checks(LINES, tr)[1])


if __name__ == "__main__":
    unittest.main()
