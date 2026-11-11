"""Tests for search_engine.py.

Run with: python -m unittest test_search_engine
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_engine import build_index, find_snippet, load_documents, search, tokenize


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])


class TestBuildIndex(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "a.txt": "The cat sat on the mat. The cat slept.",
            "b.txt": "A dog sat on the rug.",
            "c.txt": "The cat and the dog played.",
        }
        self.index, self.doc_lengths = build_index(self.documents)

    def test_term_in_right_docs(self):
        self.assertEqual(set(self.index["cat"]), {"a.txt", "c.txt"})
        self.assertEqual(set(self.index["dog"]), {"b.txt", "c.txt"})
        self.assertNotIn("mat", self.index["dog"])

    def test_positions_recorded(self):
        self.assertEqual(self.index["cat"]["a.txt"], [1, 7])

    def test_doc_lengths(self):
        self.assertEqual(self.doc_lengths["a.txt"], 9)
        self.assertEqual(self.doc_lengths["b.txt"], 6)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "a.txt": "The cat sat on the mat. The cat slept.",
            "b.txt": "A dog sat on the rug.",
            "c.txt": "The cat and the dog played.",
        }
        self.index, self.doc_lengths = build_index(self.documents)

    def test_single_term_search(self):
        results = search(self.index, self.doc_lengths, "cat")
        self.assertEqual([doc for doc, _ in results], ["a.txt", "c.txt"])

    def test_multi_term_and_search(self):
        results = search(self.index, self.doc_lengths, "cat dog")
        self.assertEqual([doc for doc, _ in results], ["c.txt"])

    def test_and_with_missing_term_returns_nothing(self):
        results = search(self.index, self.doc_lengths, "cat zebra")
        self.assertEqual(results, [])

    def test_ranking_more_occurrences_first(self):
        documents = {
            "x.txt": "apple banana apple",
            "y.txt": "apple banana cherry",
        }
        index, doc_lengths = build_index(documents)
        results = search(index, doc_lengths, "apple")
        self.assertEqual([doc for doc, _ in results], ["x.txt", "y.txt"])

    def test_missing_term(self):
        self.assertEqual(search(self.index, self.doc_lengths, "zebra"), [])

    def test_empty_query(self):
        self.assertEqual(search(self.index, self.doc_lengths, ""), [])
        self.assertEqual(search(self.index, self.doc_lengths, "   "), [])

    def test_case_insensitive(self):
        lower = search(self.index, self.doc_lengths, "cat")
        upper = search(self.index, self.doc_lengths, "CAT")
        self.assertEqual(lower, upper)

    def test_phrase_search(self):
        documents = {
            "p.txt": "the quick brown fox",
            "q.txt": "brown quick the fox",
        }
        index, doc_lengths = build_index(documents)
        self.assertEqual(
            [doc for doc, _ in search(index, doc_lengths, '"quick brown"')],
            ["p.txt"])
        self.assertEqual(
            [doc for doc, _ in search(index, doc_lengths, '"brown quick"')],
            ["q.txt"])

    def test_phrase_missing(self):
        self.assertEqual(
            search(self.index, self.doc_lengths, '"purple elephant"'), [])


class TestSnippet(unittest.TestCase):
    def test_snippet_contains_term(self):
        snippet = find_snippet("The cat sat on the mat.", ["cat"])
        self.assertIn("cat", snippet)

    def test_snippet_empty_when_term_absent(self):
        self.assertEqual(find_snippet("The cat sat.", ["zebra"]), "")


class TestLoadDocuments(unittest.TestCase):
    def test_reads_only_txt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, content in [("a.txt", "hello"), ("b.txt", "world"),
                                  ("notes.md", "not text")]:
                with open(os.path.join(directory, name), "w") as handle:
                    handle.write(content)
            documents = load_documents(directory)
        self.assertEqual(documents, {"a.txt": "hello", "b.txt": "world"})


if __name__ == "__main__":
    unittest.main()
