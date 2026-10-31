"""Exact and near-duplicate detection.

* exact duplicates      : identical text after normalisation (hash)
* near-duplicates       : word 5-shingle Jaccard >= j_thresh  OR  TF-IDF cosine >= c_thresh
                          - "near"     : Jaccard >= 0.35  (re-issued / re-formatted copy)
                          - "template" : high cosine, low Jaccard (recurring report built
                                         from the same wording template, different figures)
Groups are formed with union-find so duplicates can be kept together or dropped.
"""
import hashlib
import re

import numpy as np

_WS = re.compile(r"[^a-z0-9]+")


def normalise(text):
    text = re.sub(r"<[^>]+>", " ", str(text)).lower()
    return _WS.sub(" ", text).strip()


def shingles(text, k=5):
    w = normalise(text).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def find_duplicates(texts, sim_matrix, c_thresh=0.90, j_thresh=0.60):
    n = len(texts)
    hashes = [hashlib.md5(normalise(t).encode()).hexdigest() for t in texts]
    sh = [shingles(t) for t in texts]
    pairs = []
    iu = np.triu_indices(n, 1)
    cand = np.where(sim_matrix[iu] >= min(c_thresh, 0.5))[0]  # cheap pre-filter
    for c in cand:
        i, j = int(iu[0][c]), int(iu[1][c])
        cos = float(sim_matrix[i, j])
        jac = jaccard(sh[i], sh[j])
        exact = hashes[i] == hashes[j]
        if exact or cos >= c_thresh or jac >= j_thresh:
            if exact:
                kind = "exact"
            elif jac >= 0.35:
                kind = "near"  # re-issued / re-formatted copy of the same story
            else:
                kind = "template"  # recurring report: same wording template, different facts
            pairs.append(dict(i=i, j=j, cosine=cos, jaccard=jac, kind=kind))
    return pairs, group_pairs(n, pairs, kinds=("exact", "near"))


def group_pairs(n, pairs, kinds=("exact", "near")):
    uf = _UF(n)
    for p in pairs:
        if p["kind"] in kinds:
            uf.union(p["i"], p["j"])
    return [uf.find(i) for i in range(n)]


def keep_mask(groups):
    """Keep the first member of every duplicate group."""
    seen, mask = set(), []
    for g in groups:
        mask.append(g not in seen)
        seen.add(g)
    return np.array(mask)
