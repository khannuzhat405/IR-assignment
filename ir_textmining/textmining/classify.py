"""Multinomial Naive Bayes, stratified split and classification metrics,
all implemented manually."""
import math
import random
from collections import Counter, defaultdict

import numpy as np


def stratified_split(labels, test_ratio=0.3, seed=42, groups=None):
    """Return (train_idx, test_idx), preserving class proportions.

    If `groups` is given (e.g. near-duplicate group ids), all members of a group
    are kept on the same side of the split to avoid train/test leakage.
    """
    rng = random.Random(seed)
    if groups is None:
        groups = list(range(len(labels)))
    group_members = defaultdict(list)
    for i, g in enumerate(groups):
        group_members[g].append(i)
    by_class = defaultdict(list)
    for g, members in group_members.items():
        by_class[labels[members[0]]].append(g)
    train, test = [], []
    for cls in sorted(by_class):
        gs = by_class[cls]
        rng.shuffle(gs)
        n_test = max(1, round(len(gs) * test_ratio))
        for g in gs[:n_test]:
            test.extend(group_members[g])
        for g in gs[n_test:]:
            train.extend(group_members[g])
    return sorted(train), sorted(test)


class MultinomialNaiveBayes:
    """P(c|d) ∝ P(c) * Π_t P(t|c)^tf(t,d), with Laplace (add-alpha) smoothing.

    P(t|c) = (count(t,c) + alpha) / (Σ_t' count(t',c) + alpha*|V|)
    Computation is done in log space to avoid underflow.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, token_lists, labels):
        self.classes = sorted(set(labels))
        n = len(labels)
        class_counts = Counter(labels)
        self.log_prior = {c: math.log(class_counts[c] / n) for c in self.classes}
        self.term_counts = {c: Counter() for c in self.classes}
        for tokens, y in zip(token_lists, labels):
            self.term_counts[y].update(tokens)
        self.vocab = set()
        for c in self.classes:
            self.vocab.update(self.term_counts[c])
        V = len(self.vocab)
        self.total = {c: sum(self.term_counts[c].values()) for c in self.classes}
        self.denom = {c: self.total[c] + self.alpha * V for c in self.classes}
        self.log_unseen = {c: math.log(self.alpha / self.denom[c]) for c in self.classes}
        return self

    def log_likelihood(self, term, c):
        cnt = self.term_counts[c].get(term, 0)
        return math.log((cnt + self.alpha) / self.denom[c])

    def joint_log_scores(self, tokens):
        tf = Counter(t for t in tokens if t in self.vocab)  # ignore OOV terms
        return {
            c: self.log_prior[c] + sum(n * self.log_likelihood(t, c) for t, n in tf.items())
            for c in self.classes
        }

    def predict_one(self, tokens):
        s = self.joint_log_scores(tokens)
        return max(s, key=s.get)

    def predict_proba_one(self, tokens):
        s = self.joint_log_scores(tokens)
        m = max(s.values())
        exp = {c: math.exp(v - m) for c, v in s.items()}
        z = sum(exp.values())
        return {c: v / z for c, v in exp.items()}

    def predict(self, token_lists):
        return [self.predict_one(t) for t in token_lists]

    def top_features(self, c, k=10):
        """Terms with the highest log-ratio P(t|c) / P(t|not c)."""
        others = [o for o in self.classes if o != c]
        rest = Counter()
        for o in others:
            rest.update(self.term_counts[o])
        rest_total = sum(rest.values()) + self.alpha * len(self.vocab)
        scores = []
        for t, cnt in self.term_counts[c].items():
            if cnt < 3:
                continue
            p_c = (cnt + self.alpha) / self.denom[c]
            p_o = (rest.get(t, 0) + self.alpha) / rest_total
            scores.append((t, math.log(p_c / p_o)))
        return sorted(scores, key=lambda x: -x[1])[:k]


def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M  # rows = actual, columns = predicted


def classification_report(y_true, y_pred, classes):
    M = confusion_matrix(y_true, y_pred, classes)
    per_class = {}
    for i, c in enumerate(classes):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = dict(precision=prec, recall=rec, f1=f1, support=int(M[i, :].sum()))
    n = len(y_true)
    acc = float(np.trace(M)) / n if n else 0.0
    macro = {k: float(np.mean([per_class[c][k] for c in classes])) for k in ("precision", "recall", "f1")}
    weighted = {
        k: float(sum(per_class[c][k] * per_class[c]["support"] for c in classes) / n)
        for k in ("precision", "recall", "f1")
    }
    return dict(accuracy=acc, macro=macro, weighted=weighted, per_class=per_class, confusion=M)
