"""Inverted index, TF-IDF weighting with cosine length normalisation, and
cosine similarity -- implemented by hand (no sklearn / gensim vectorizers).

Weighting scheme (SMART notation  l t c):
    tf  weight  w_tf  = 1 + log10(tf)          (tf > 0)
    idf weight  idf   = log10(N / df)
    raw weight  w     = w_tf * idf
    normalised  w'    = w / sqrt(sum_t w_t^2)   (cosine / L2 length normalisation)
"""
import math
from collections import Counter, defaultdict

import numpy as np


class InvertedIndex:
    """term -> postings list [(doc_idx, term_frequency), ...] sorted by doc_idx."""

    def __init__(self, token_lists, min_df=1, max_df_ratio=1.0):
        self.N = len(token_lists)
        postings = defaultdict(list)
        for d, tokens in enumerate(token_lists):
            for term, tf in Counter(tokens).items():
                postings[term].append((d, tf))
        max_df = max_df_ratio * self.N
        self.postings = {
            t: p for t, p in postings.items() if min_df <= len(p) <= max_df
        }
        self.vocab = sorted(self.postings)
        self.term_id = {t: i for i, t in enumerate(self.vocab)}
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.idf = {t: math.log10(self.N / df) for t, df in self.df.items()}
        self.pruned = len(postings) - len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def lookup(self, term):
        return self.postings.get(term, [])


class TfidfModel:
    def __init__(self, index: InvertedIndex):
        self.index = index
        N, V = index.N, len(index)
        raw = np.zeros((N, V), dtype=np.float64)
        for term, plist in index.postings.items():
            j = index.term_id[term]
            idf = index.idf[term]
            for d, tf in plist:
                raw[d, j] = (1.0 + math.log10(tf)) * idf
        self.raw = raw
        norms = np.sqrt((raw ** 2).sum(axis=1))
        self.lengths = norms
        safe = np.where(norms == 0, 1.0, norms)
        self.matrix = raw / safe[:, None]  # rows have unit length

    def vectorize(self, tokens):
        """TF-IDF vector for an unseen query / document using corpus idf."""
        v = np.zeros(len(self.index), dtype=np.float64)
        for term, tf in Counter(tokens).items():
            j = self.index.term_id.get(term)
            if j is not None:
                v[j] = (1.0 + math.log10(tf)) * self.index.idf[term]
        n = math.sqrt(float((v ** 2).sum()))
        return v / n if n > 0 else v

    def top_terms(self, doc_idx, k=10):
        row = self.matrix[doc_idx]
        idx = np.argsort(-row)[:k]
        return [(self.index.vocab[j], float(row[j])) for j in idx if row[j] > 0]


def cosine_similarity(a, b):
    """Cosine of two vectors (works on un-normalised vectors as well)."""
    na = math.sqrt(float(np.dot(a, a)))
    nb = math.sqrt(float(np.dot(b, b)))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(M):
    """All-pairs cosine similarity. Rows of M are L2-normalised, so cos = dot."""
    norms = np.sqrt((M ** 2).sum(axis=1))
    safe = np.where(norms == 0, 1.0, norms)
    U = M / safe[:, None]
    return U @ U.T


def score_query_with_index(index: InvertedIndex, model: TfidfModel, query_tokens):
    """Term-at-a-time scoring through the postings lists (classic IR retrieval)."""
    q = model.vectorize(query_tokens)
    scores = defaultdict(float)
    for term in set(query_tokens):
        j = index.term_id.get(term)
        if j is None:
            continue
        for d, _tf in index.lookup(term):
            scores[d] += q[j] * model.matrix[d, j]
    return sorted(scores.items(), key=lambda x: -x[1])
