"""A small command-line search engine built on an inverted index.

An inverted index maps each word to the documents that contain it, so a
query never scans every document: it looks up each query word and works
only with the documents that contain it.

Commands:
    python search_engine.py index <corpus_dir> [--output index.json]
    python search_engine.py search "<query>" [--index index.json] [--corpus corpus]

The index step reads every .txt file in the corpus directory and writes a
JSON index file. The search step loads the index, finds the documents that
contain every query word, ranks them, and prints a short matching snippet
for each hit.

Ranking: a document scores the total number of times the query words
appear in it, divided by the square root of the document's length in
words. That rewards documents that mention the query a lot while giving
shorter documents a fair chance.

Queries are case-insensitive. A query in double quotes is a phrase search:
only documents containing the exact phrase, in order, match.

Python standard library only.
"""

import argparse
import json
import math
import os
import re
import sys


def tokenize(text):
    """Split text into lowercase words.

    Anything that is not a letter or a digit breaks a word, so punctuation
    and case do not matter: "Alice," and "alice" become the same token.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(documents):
    """Build an inverted index from {document name: document text}.

    Returns (index, doc_lengths). The index maps each term to
    {document name: [positions]}, where positions are token offsets
    counted from 0. doc_lengths maps each document name to its number
    of tokens.
    """
    index = {}
    doc_lengths = {}
    for name, text in documents.items():
        tokens = tokenize(text)
        doc_lengths[name] = len(tokens)
        for position, term in enumerate(tokens):
            index.setdefault(term, {}).setdefault(name, []).append(position)
    return index, doc_lengths


def _phrase_occurrences(position_sets):
    """Count occurrences of a phrase in one document.

    position_sets is a list of sets of token positions, one set per phrase
    word. A phrase occurs at position p when the first word is at p and
    each following word is at p + 1, p + 2, and so on.
    """
    count = 0
    for p in position_sets[0]:
        if all(p + i in position_sets[i] for i in range(1, len(position_sets))):
            count += 1
    return count


def _ranked(results, doc_lengths):
    """Sort (document, score) pairs best first, breaking ties by name."""
    return sorted(results, key=lambda item: (-item[1], item[0]))


def _and_search(index, doc_lengths, terms):
    """Documents containing every term, ranked by total term frequency
    divided by the square root of the document length."""
    postings = [index.get(term) for term in terms]
    if any(posting is None for posting in postings):
        return []
    common = set(postings[0])
    for posting in postings[1:]:
        common &= set(posting)
    results = []
    for doc in common:
        total = sum(len(postings[i][doc]) for i in range(len(terms)))
        score = total / math.sqrt(doc_lengths[doc]) if doc_lengths[doc] else 0.0
        results.append((doc, score))
    return _ranked(results, doc_lengths)


def _phrase_search(index, doc_lengths, terms):
    """Documents containing the exact phrase, ranked by phrase frequency
    divided by the square root of the document length."""
    postings = [index.get(term) for term in terms]
    if any(posting is None for posting in postings):
        return []
    common = set(postings[0])
    for posting in postings[1:]:
        common &= set(posting)
    results = []
    for doc in common:
        sets = [set(postings[i][doc]) for i in range(len(terms))]
        count = _phrase_occurrences(sets)
        if count:
            score = count / math.sqrt(doc_lengths[doc]) if doc_lengths[doc] else 0.0
            results.append((doc, score))
    return _ranked(results, doc_lengths)


def search(index, doc_lengths, query):
    """Search the index. Returns a list of (document name, score) pairs,
    best score first.

    A query in double quotes is a phrase search. Otherwise every query
    word must appear in a document (AND). An empty query, or a word that
    appears nowhere, returns no results.
    """
    query = query.strip()
    if not query:
        return []
    if query.startswith('"') and query.endswith('"') and len(query) >= 2:
        terms = tokenize(query[1:-1])
        if not terms:
            return []
        return _phrase_search(index, doc_lengths, terms)
    terms = tokenize(query)
    if not terms:
        return []
    return _and_search(index, doc_lengths, terms)


def find_snippet(text, terms, width=45):
    """Return a short excerpt of text around the first occurrence of any
    of the given terms. Returns "" when no term occurs."""
    lowered = text.lower()
    first = None
    for term in terms:
        at = lowered.find(term)
        if at != -1 and (first is None or at < first):
            first = at
    if first is None:
        return ""
    start = max(0, first - width)
    end = min(len(text), first + width)
    snippet = " ".join(text[start:end].split())
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def load_documents(directory):
    """Read every .txt file in a directory. Returns {file name: text}."""
    documents = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            documents[name] = handle.read()
    return documents


def save_index(path, index, doc_lengths):
    """Write the index and document lengths to a JSON file."""
    data = {"index": index, "doc_lengths": doc_lengths}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def load_index(path):
    """Read a JSON index file. Returns (index, doc_lengths)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["index"], data["doc_lengths"]


def main(argv=None):
    """Command-line entry point. Returns an exit code."""
    parser = argparse.ArgumentParser(
        description="A small search engine built on an inverted index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="build an index from a directory of .txt files")
    index_parser.add_argument("corpus_dir", help="directory of .txt files")
    index_parser.add_argument("--output", default="index.json",
                              help="where to write the index (default: index.json)")

    search_parser = subparsers.add_parser(
        "search", help="search the index and print ranked results")
    search_parser.add_argument(
        "query", help='search words, or a phrase in double quotes')
    search_parser.add_argument("--index", default="index.json",
                               help="index file to load (default: index.json)")
    search_parser.add_argument("--corpus", default="corpus",
                               help="corpus directory, for snippets (default: corpus)")

    args = parser.parse_args(argv)

    if args.command == "index":
        documents = load_documents(args.corpus_dir)
        if not documents:
            print(f"No .txt files found in {args.corpus_dir}.")
            return 1
        index, doc_lengths = build_index(documents)
        save_index(args.output, index, doc_lengths)
        print(f"Indexed {len(documents)} documents, "
              f"{len(index)} unique terms. Wrote {args.output}.")
        return 0

    try:
        index, doc_lengths = load_index(args.index)
    except FileNotFoundError:
        print(f"Index file {args.index} not found. "
              "Run 'python search_engine.py index <corpus_dir>' first.")
        return 1

    if not args.query.strip():
        print("Empty query: type one or more words to search for.")
        return 0
    results = search(index, doc_lengths, args.query)
    display_query = args.query.strip()
    if display_query.startswith('"') and display_query.endswith('"'):
        display_query = display_query[1:-1]
    if not results:
        print(f'No documents match "{display_query}".')
        return 0
    documents = load_documents(args.corpus)
    if args.query.strip().startswith('"'):
        snippet_terms = [args.query.strip().strip('"')]
    else:
        snippet_terms = tokenize(args.query)
    print(f'{len(results)} result(s) for "{display_query}":')
    for rank, (doc, score) in enumerate(results, start=1):
        snippet = find_snippet(documents.get(doc, ""), snippet_terms)
        print(f"{rank}. {doc} (score {score:.2f})")
        if snippet:
            print(f"   {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
