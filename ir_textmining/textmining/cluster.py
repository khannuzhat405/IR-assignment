"""K-Means (with k-means++ seeding and full step logging), Purity, Rand Index
and a PCA projection for 2-D plotting -- all implemented manually with numpy."""
from collections import Counter
from itertools import combinations

import numpy as np


class KMeans:
    def __init__(self, k, max_iter=100, tol=1e-6, init="k-means++", n_init=5, seed=42):
        self.k, self.max_iter, self.tol = k, max_iter, tol
        self.init, self.n_init, self.seed = init, n_init, seed

    # ---------- initialisation ----------
    def _init_centroids(self, X, rng):
        n = X.shape[0]
        if self.init == "random":
            seeds = list(rng.choice(n, self.k, replace=False))
        else:  # k-means++ : next seed sampled ∝ squared distance to nearest seed
            seeds = [int(rng.integers(n))]
            d2 = ((X - X[seeds[0]]) ** 2).sum(axis=1)
            for _ in range(1, self.k):
                probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
                nxt = int(rng.choice(n, p=probs))
                seeds.append(nxt)
                d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(axis=1))
        return X[seeds].copy(), [int(s) for s in seeds]

    @staticmethod
    def _sq_dists(X, C):
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
        return (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]

    def _single_run(self, X, rng):
        C, seed_idx = self._init_centroids(X, rng)
        log = []
        labels = None
        stop_reason = f"reached max_iter={self.max_iter}"
        for it in range(1, self.max_iter + 1):
            # assignment step
            D = self._sq_dists(X, C)
            new_labels = D.argmin(axis=1)
            changed = int(self.k if labels is None else (new_labels != labels).sum())
            labels = new_labels
            # update step
            newC = C.copy()
            for j in range(self.k):
                members = X[labels == j]
                if len(members):
                    newC[j] = members.mean(axis=0)
                else:  # empty cluster -> re-seed with the farthest point
                    newC[j] = X[D.min(axis=1).argmax()]
            shift = float(np.sqrt(((newC - C) ** 2).sum(axis=1)).max())
            C = newC
            sse = float(self._sq_dists(X, C)[np.arange(len(X)), labels].sum())
            log.append(dict(iteration=it, reassigned=changed, max_centroid_shift=shift, sse=sse,
                            sizes=np.bincount(labels, minlength=self.k).tolist()))
            if it > 1 and changed == 0:
                stop_reason = "no document changed cluster (assignments converged)"
                break
            if shift < self.tol:
                stop_reason = f"max centroid shift {shift:.2e} < tol {self.tol}"
                break
        return dict(labels=labels, centroids=C, log=log, sse=log[-1]["sse"],
                    seed_idx=seed_idx, stop_reason=stop_reason)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        runs = [self._single_run(X, rng) for _ in range(self.n_init)]
        best = min(runs, key=lambda r: r["sse"])
        self.labels_ = best["labels"]
        self.centroids_ = best["centroids"]
        self.log_ = best["log"]
        self.sse_ = best["sse"]
        self.seed_idx_ = best["seed_idx"]
        self.stop_reason_ = best["stop_reason"]
        self.n_iter_ = len(best["log"])
        self.all_run_sse_ = [r["sse"] for r in runs]
        return self


def purity(clusters, labels):
    """(1/N) Σ_k max_j |ω_k ∩ c_j|"""
    total = 0
    for k in set(clusters):
        members = [labels[i] for i in range(len(labels)) if clusters[i] == k]
        total += Counter(members).most_common(1)[0][1]
    return total / len(labels)


def rand_index(clusters, labels):
    """RI = (TP + TN) / (TP + FP + FN + TN) over all N(N-1)/2 document pairs."""
    tp = fp = fn = tn = 0
    for i, j in combinations(range(len(labels)), 2):
        same_cluster = clusters[i] == clusters[j]
        same_class = labels[i] == labels[j]
        if same_cluster and same_class:
            tp += 1
        elif same_cluster:
            fp += 1
        elif same_class:
            fn += 1
        else:
            tn += 1
    ri = (tp + tn) / (tp + fp + fn + tn)
    return ri, dict(TP=tp, FP=fp, FN=fn, TN=tn)


def contingency(clusters, labels):
    ks = sorted(set(clusters))
    cs = sorted(set(labels))
    M = np.zeros((len(ks), len(cs)), dtype=int)
    for k, c in zip(clusters, labels):
        M[ks.index(k), cs.index(c)] += 1
    return M, ks, cs


def pca_2d(X):
    """Project onto the first two principal components (numpy SVD).
    Used for visualisation only."""
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var = (S ** 2) / (S ** 2).sum()
    return coords, var[:2]
