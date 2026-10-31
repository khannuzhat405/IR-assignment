"""Content-based Top-K recommendation on manual TF-IDF vectors."""
import numpy as np


def recommend_similar(sim_matrix, doc_idx, k=5, exclude=None, min_score=0.0):
    """Top-K documents most similar (cosine) to document `doc_idx`.

    exclude : optional set of indices to skip (e.g. the document's duplicates)
    """
    scores = sim_matrix[doc_idx].copy()
    scores[doc_idx] = -1
    if exclude:
        for e in exclude:
            scores[e] = -1
    order = np.argsort(-scores)
    out = [(int(i), float(scores[i])) for i in order if scores[i] > min_score]
    return out[:k]


def recommend_for_vector(model, qvec, k=5):
    scores = model.matrix @ qvec
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def explain_overlap(model, a, b, k=6):
    """Terms contributing most to cos(a,b): element-wise product of weights."""
    contrib = model.matrix[a] * model.matrix[b]
    idx = np.argsort(-contrib)[:k]
    return [(model.index.vocab[j], float(contrib[j])) for j in idx if contrib[j] > 0]


def precision_at_k(sim_matrix, labels, k=5, exclude_groups=None):
    """Mean fraction of Top-K recommendations sharing the query's category.
    A proxy offline evaluation for a content-based recommender."""
    vals = []
    for i in range(len(labels)):
        excl = None
        if exclude_groups is not None:
            excl = {j for j, g in enumerate(exclude_groups) if g == exclude_groups[i] and j != i}
        recs = recommend_similar(sim_matrix, i, k, exclude=excl)
        if recs:
            vals.append(sum(labels[j] == labels[i] for j, _ in recs) / len(recs))
    return float(np.mean(vals))
