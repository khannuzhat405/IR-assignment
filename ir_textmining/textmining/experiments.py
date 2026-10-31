"""Controlled experiments used for the 'Performance analytics' page and the report.
Every metric is averaged over several random seeds (train/test split and
K-Means seeding) so that differences are not artefacts of one lucky split."""
from collections import Counter

import numpy as np

from .classify import MultinomialNaiveBayes, stratified_split
from .cluster import KMeans, purity, rand_index
from .pipeline import build_features, run_all
from .preprocess import PreprocessConfig, preprocess_corpus

CONFIGS = {
    "A. Raw tokens (no cleaning)": PreprocessConfig(
        lowercase=False, strip_markup=False, remove_numbers=False, remove_stopwords=False,
        domain_stopwords=False, normalizer="none", min_token_len=1),
    "B. Lowercase + markup/numbers removed": PreprocessConfig(
        remove_stopwords=False, domain_stopwords=False, normalizer="none"),
    "C. B + stop-words (generic + domain)": PreprocessConfig(normalizer="none"),
    "D. C + Porter stemming": PreprocessConfig(normalizer="stem"),
    "E. C + WordNet lemmatization": PreprocessConfig(normalizer="lemma"),
    "F. E + bigrams": PreprocessConfig(normalizer="lemma", ngram_max=2),
}


def _summ(vals):
    return float(np.mean(vals)), float(np.std(vals))


def preprocessing_experiment(df, seeds=(0, 1, 2, 3, 4), configs=CONFIGS, progress=None):
    rows = []
    for n, (name, cfg) in enumerate(configs.items()):
        accs, f1s, purs, ris, pks = [], [], [], [], []
        last = None
        for s in seeds:
            r = run_all(df, cfg, seed=s)
            accs.append(r["report"]["accuracy"]); f1s.append(r["report"]["macro"]["f1"])
            purs.append(r["purity"]); ris.append(r["rand_index"]); pks.append(r["precision_at_k"])
            last = r
        rows.append(dict(
            config=name, description=cfg.describe(),
            vocab_before_pruning=last["proc_stats"]["vocab_size"],
            indexed_vocab=len(last["feats"]["index"]),
            tokens=last["proc_stats"]["total_tokens"],
            nb_accuracy=_summ(accs), nb_macro_f1=_summ(f1s),
            km_purity=_summ(purs), km_rand_index=_summ(ris), rec_precision_at_5=_summ(pks),
        ))
        if progress:
            progress((n + 1) / len(configs), name)
    return rows


def dedup_experiment(df, cfg=None, seeds=(0, 1, 2, 3, 4)):
    cfg = cfg or PreprocessConfig()
    modes = {
        "Keep all (no de-duplication)": dict(dedup=False, group_split=False),
        "Remove exact + near duplicates": dict(dedup=True),
        "Also remove template duplicates": dict(dedup=True, drop_templates=True),
    }
    rows = []
    for name, kw in modes.items():
        accs, f1s, purs, ris, pks, ndocs = [], [], [], [], [], []
        for s in seeds:
            r = run_all(df, cfg, seed=s, **kw)
            accs.append(r["report"]["accuracy"]); f1s.append(r["report"]["macro"]["f1"])
            purs.append(r["purity"]); ris.append(r["rand_index"]); pks.append(r["precision_at_k"])
            ndocs.append(len(r["df"]))
        rows.append(dict(mode=name, docs=ndocs[0], nb_accuracy=_summ(accs), nb_macro_f1=_summ(f1s),
                         km_purity=_summ(purs), km_rand_index=_summ(ris),
                         rec_precision_at_5=_summ(pks)))
    return rows


def duplicate_leakage(df, cfg=None, seeds=range(10)):
    """Effect of duplicates on each task when they are NOT removed.

    * classification: accuracy on test docs whose duplicate sits in the training set
    * recommendation: share of documents (that have a duplicate) whose Top-1
      recommendation is merely its own duplicate
    """
    cfg = cfg or PreprocessConfig()
    texts, y = df["text"].tolist(), df["category"].tolist()
    feats = build_features(texts, cfg)
    from .dedup import find_duplicates
    pairs, groups = find_duplicates(texts, feats["sim"])
    tokens = feats["tokens"]
    gsize = Counter(groups)
    leaked_hits = leaked_n = clean_hits = clean_n = 0
    for s in seeds:
        tr, te = stratified_split(y, 0.3, s)  # naive split, duplicates may straddle
        nb = MultinomialNaiveBayes().fit([tokens[i] for i in tr], [y[i] for i in tr])
        trs = set(tr)
        train_groups = {groups[i] for i in tr}
        for i in te:
            ok = nb.predict_one(tokens[i]) == y[i]
            has_twin_in_train = gsize[groups[i]] > 1 and groups[i] in train_groups
            if has_twin_in_train:
                leaked_hits += ok; leaked_n += 1
            else:
                clean_hits += ok; clean_n += 1
        del trs
    sim = feats["sim"]
    dup_docs = [i for i in range(len(y)) if gsize[groups[i]] > 1]
    top1_dup = 0
    for i in dup_docs:
        s = sim[i].copy(); s[i] = -1
        j = int(s.argmax())
        top1_dup += groups[j] == groups[i]
    km_full = KMeans(len(set(y)), seed=0).fit(feats["model"].matrix)
    same_cluster = sum(km_full.labels_[p["i"]] == km_full.labels_[p["j"]] for p in pairs)
    return dict(
        n_pairs=len(pairs), kinds=dict(Counter(p["kind"] for p in pairs)),
        n_docs_with_dup=len(dup_docs),
        leaked_test_docs=leaked_n, leaked_accuracy=leaked_hits / leaked_n if leaked_n else None,
        clean_test_docs=clean_n, clean_accuracy=clean_hits / clean_n,
        top1_is_duplicate=top1_dup / len(dup_docs) if dup_docs else 0,
        dup_pairs_same_cluster=same_cluster / len(pairs) if pairs else 0,
    )


def k_sweep(df, cfg=None, ks=(3, 4, 5, 6, 7, 8, 10, 12), seeds=(0, 1, 2)):
    cfg = cfg or PreprocessConfig()
    r = run_all(df, cfg, seed=0)
    X, y = r["feats"]["model"].matrix, r["labels"]
    rows = []
    for k in ks:
        pu, ri, sse = [], [], []
        for s in seeds:
            km = KMeans(k, seed=s).fit(X)
            pu.append(purity(km.labels_.tolist(), y))
            ri.append(rand_index(km.labels_.tolist(), y)[0])
            sse.append(km.sse_)
        rows.append(dict(k=k, purity=float(np.mean(pu)), rand_index=float(np.mean(ri)), sse=float(np.mean(sse))))
    return rows


def vocab_growth(df, configs=CONFIGS):
    out = {}
    for name, cfg in configs.items():
        toks = preprocess_corpus(df["text"].tolist(), cfg)
        c = Counter(t for d in toks for t in d)
        out[name] = dict(vocab=len(c), tokens=sum(c.values()), top=c.most_common(15))
    return out
