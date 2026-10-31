"""End-to-end pipeline used by both the Streamlit app and experiments.py."""
import time
from collections import Counter

import numpy as np

from .classify import MultinomialNaiveBayes, classification_report, stratified_split
from .cluster import KMeans, contingency, pca_2d, purity, rand_index
from .dedup import find_duplicates, group_pairs, keep_mask
from .features import InvertedIndex, TfidfModel, cosine_matrix
from .preprocess import RAW_CONFIG, preprocess_corpus
from .recommend import precision_at_k


def build_features(texts, cfg, min_df=2, max_df_ratio=0.9):
    t0 = time.time()
    tokens = preprocess_corpus(texts, cfg)
    index = InvertedIndex(tokens, min_df=min_df, max_df_ratio=max_df_ratio)
    model = TfidfModel(index)
    sim = cosine_matrix(model.matrix)
    return dict(tokens=tokens, index=index, model=model, sim=sim, seconds=time.time() - t0)


def vocab_stats(token_lists):
    c = Counter(t for doc in token_lists for t in doc)
    lens = [len(d) for d in token_lists]
    return dict(
        vocab_size=len(c), total_tokens=sum(c.values()),
        hapax=sum(1 for v in c.values() if v == 1),
        mean_doc_len=float(np.mean(lens)), median_doc_len=float(np.median(lens)),
        type_token_ratio=len(c) / max(1, sum(c.values())), counter=c,
    )


def run_all(df, cfg, k=None, test_ratio=0.3, seed=42, dedup=True, alpha=1.0,
            min_df=2, max_df_ratio=0.9, kmeans_init="k-means++", n_init=5, rec_k=5,
            group_split=True, drop_templates=False):
    texts = df["text"].tolist()
    labels = df["category"].tolist()
    feats = build_features(texts, cfg, min_df, max_df_ratio)
    pairs, groups = find_duplicates(texts, feats["sim"])
    if drop_templates:
        groups = group_pairs(len(texts), pairs, kinds=("exact", "near", "template"))

    # ---- de-duplication ----
    mask = keep_mask(groups) if dedup else np.ones(len(df), dtype=bool)
    keep = np.where(mask)[0]
    if dedup:
        sub = df.iloc[keep].reset_index(drop=True)
        feats = build_features(sub["text"].tolist(), cfg, min_df, max_df_ratio)
        groups_sub = [groups[i] for i in keep]
    else:
        sub = df.reset_index(drop=True)
        groups_sub = groups
    y = sub["category"].tolist()
    tokens = feats["tokens"]

    # ---- classification ----
    tr, te = stratified_split(y, test_ratio, seed, groups=groups_sub if group_split else None)
    nb = MultinomialNaiveBayes(alpha).fit([tokens[i] for i in tr], [y[i] for i in tr])
    pred = nb.predict([tokens[i] for i in te])
    classes = sorted(set(y))
    rep = classification_report([y[i] for i in te], pred, classes)

    # ---- clustering ----
    k = k or len(classes)
    km = KMeans(k, init=kmeans_init, n_init=n_init, seed=seed).fit(feats["model"].matrix)
    clusters = km.labels_.tolist()
    pur = purity(clusters, y)
    ri, ri_counts = rand_index(clusters, y)
    cont, ks, cs = contingency(clusters, y)
    coords, var = pca_2d(feats["model"].matrix)

    # ---- recommendation ----
    p_at_k = precision_at_k(feats["sim"], y, rec_k)

    return dict(
        df=sub, labels=y, feats=feats, dup_pairs=pairs, dup_groups_full=groups,
        groups=groups_sub, n_removed=int(len(df) - len(sub)),
        train_idx=tr, test_idx=te, nb=nb, pred=pred, classes=classes, report=rep,
        kmeans=km, clusters=clusters, purity=pur, rand_index=ri, ri_counts=ri_counts,
        contingency=(cont, ks, cs), coords=coords, pca_var=var,
        precision_at_k=p_at_k, rec_k=rec_k, cfg=cfg,
        raw_stats=vocab_stats(preprocess_corpus(sub["text"].tolist(), RAW_CONFIG)),
        proc_stats=vocab_stats(tokens),
    )


def cluster_label_map(clusters, labels):
    """Name each cluster by its majority true label (for display/comparison only)."""
    out = {}
    for k in sorted(set(clusters)):
        members = [labels[i] for i in range(len(labels)) if clusters[i] == k]
        lab, cnt = Counter(members).most_common(1)[0]
        out[k] = (lab, cnt / len(members))
    return out
