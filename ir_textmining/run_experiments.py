"""Reproduces all numbers and figures used in the report.
Usage: python run_experiments.py      -> writes results/results.json and results/fig_*.png
"""
import html
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from textmining.experiments import (dedup_experiment, duplicate_leakage, k_sweep,
                                    preprocessing_experiment)
from textmining.pipeline import cluster_label_map, run_all
from textmining.preprocess import PreprocessConfig
from textmining.recommend import explain_overlap, recommend_similar

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, ORANGE, GREY = "#2E6F95", "#E09F3E", "#8A9BA8"
CAT_COLORS = dict(zip(["acq", "crude", "interest", "money-fx", "ship", "trade"],
                      ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499"]))

df = pd.read_csv("data/reuters_finance_news.csv").fillna("")
df["title"] = df["title"].map(html.unescape)
res = {}

# ---------------- main run (default configuration, seed 42) ----------------
cfg = PreprocessConfig()
R = run_all(df, cfg, seed=42)
sub, y, classes = R["df"], R["labels"], R["classes"]
rep = R["report"]
res["main"] = dict(
    n_docs=len(df), n_after_dedup=len(sub), removed=R["n_removed"],
    raw=dict((k, v) for k, v in R["raw_stats"].items() if k != "counter"),
    processed=dict((k, v) for k, v in R["proc_stats"].items() if k != "counter"),
    indexed_vocab=len(R["feats"]["index"]), pruned=R["feats"]["index"].pruned,
    raw_top=R["raw_stats"]["counter"].most_common(15), proc_top=R["proc_stats"]["counter"].most_common(15),
    train=len(R["train_idx"]), test=len(R["test_idx"]),
    accuracy=rep["accuracy"], macro=rep["macro"], weighted=rep["weighted"],
    per_class=rep["per_class"], confusion=rep["confusion"].tolist(), classes=classes,
    purity=R["purity"], rand_index=R["rand_index"], ri_counts=R["ri_counts"],
    contingency=R["contingency"][0].tolist(),
    kmeans_log=R["kmeans"].log_, kmeans_stop=R["kmeans"].stop_reason_,
    kmeans_seeds=[sub.loc[s, "doc_id"] for s in R["kmeans"].seed_idx_],
    kmeans_restart_sse=R["kmeans"].all_run_sse_, pca_var=R["pca_var"].tolist(),
    precision_at_5=R["precision_at_k"],
    top_features={c: R["nb"].top_features(c, 8) for c in classes},
    centroid_terms=[[R["feats"]["index"].vocab[t] for t in np.argsort(-c)[:8]] for c in R["kmeans"].centroids_],
    cluster_map={int(k): v for k, v in cluster_label_map(R["clusters"], y).items()},
)
res["main"]["predictions"] = [
    dict(doc_id=sub.loc[i, "doc_id"], title=sub.loc[i, "title"], actual=y[i], predicted=p)
    for i, p in zip(R["test_idx"], R["pred"])]
res["main"]["membership"] = [dict(doc_id=sub.loc[i, "doc_id"], title=sub.loc[i, "title"], label=y[i],
                                  cluster=int(R["clusters"][i])) for i in range(len(sub))]
res["dup_pairs"] = [dict(a=df.loc[p["i"], "doc_id"], b=df.loc[p["j"], "doc_id"], ta=df.loc[p["i"], "title"],
                         cosine=p["cosine"], jaccard=p["jaccard"], kind=p["kind"]) for p in R["dup_pairs"]]

# recommendation examples
model, sim = R["feats"]["model"], R["feats"]["sim"]
recs = {}
for qid in ["RTR-06301", "RTR-08710", "RTR-11349"]:
    if (sub["doc_id"] == qid).any():
        qi = int(np.where(sub["doc_id"] == qid)[0][0])
        excl = {j for j in range(len(sub)) if (R["groups"][j] == R["groups"][qi] or sim[qi, j] >= 0.90) and j != qi}
        raw_top = recommend_similar(sim, qi, 1)[0]
        recs[qid] = dict(title=sub.loc[qi, "title"], category=y[qi],
                         unfiltered_top1=dict(doc_id=sub.loc[raw_top[0], "doc_id"], score=raw_top[1],
                                              title=sub.loc[raw_top[0], "title"]),
                         recs=[
            dict(doc_id=sub.loc[j, "doc_id"], title=sub.loc[j, "title"], category=y[j], score=s,
                 terms=[t for t, _ in explain_overlap(model, qi, j, 4)])
            for j, s in recommend_similar(sim, qi, 5, exclude=excl)])
res["recommendations"] = recs

# ---------------- experiments (5 seeds) ----------------
print("preprocessing experiment …")
res["preprocessing"] = preprocessing_experiment(df)
print("dedup experiment …")
res["dedup"] = dedup_experiment(df)
res["leakage"] = duplicate_leakage(df)
print("k sweep …")
res["k_sweep"] = k_sweep(df)


def to_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


with open(f"{OUT}/results.json", "w") as f:
    json.dump(res, f, indent=1, default=to_json)

# ---------------- figures ----------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)


# corpus
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
vc = df["category"].value_counts().sort_index()
ax[0].bar(vc.index, vc.values, color=[CAT_COLORS[c] for c in vc.index]); ax[0].set_title("Documents per category")
ax[1].hist(df["text"].str.split().str.len(), bins=40, color=BLUE); ax[1].set_title("Document length (words)")
save(fig, "fig_corpus")

# vocab before/after
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, stats, t, c in [(ax[0], R["raw_stats"], "Top terms - before", ORANGE), (ax[1], R["proc_stats"], "Top terms - after", BLUE)]:
    top = stats["counter"].most_common(15)[::-1]
    a.barh([w for w, _ in top], [n for _, n in top], color=c); a.set_title(t)
for stats, lab, c in [(R["raw_stats"], "before", ORANGE), (R["proc_stats"], "after", BLUE)]:
    f = sorted(stats["counter"].values(), reverse=True)
    ax[2].loglog(range(1, len(f) + 1), f, label=lab, color=c)
ax[2].set_title("Rank-frequency (Zipf)"); ax[2].set_xlabel("rank"); ax[2].legend()
save(fig, "fig_vocab")

# classification
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
actual = [y[i] for i in R["test_idx"]]
x = np.arange(len(classes))
ax[0].bar(x - 0.2, [actual.count(c) for c in classes], 0.4, label="Actual", color=BLUE)
ax[0].bar(x + 0.2, [R["pred"].count(c) for c in classes], 0.4, label="Predicted", color=ORANGE)
ax[0].set_xticks(x, classes, rotation=30); ax[0].legend(); ax[0].set_title("Actual vs predicted (test set)")
M = rep["confusion"]
ax[1].imshow(M, cmap="Blues")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax[1].text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
ax[1].set_xticks(x, classes, rotation=30); ax[1].set_yticks(x, classes)
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual"); ax[1].set_title("Confusion matrix")
save(fig, "fig_classification")

# clustering
cmap = cluster_label_map(R["clusters"], y)
coords = R["coords"]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
pal = plt.cm.tab10.colors
for k in sorted(set(R["clusters"])):
    m = np.array(R["clusters"]) == k
    ax[0].scatter(coords[m, 0], coords[m, 1], s=10, color=pal[k], label=f"C{k} ({cmap[k][0]})", alpha=0.8)
ax[0].legend(fontsize=7, markerscale=1.5); ax[0].set_title("K-Means clusters (PCA projection, display only)")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
for c in classes:
    m = np.array(y) == c
    ax[1].scatter(coords[m, 0], coords[m, 1], s=10, color=CAT_COLORS.get(c, GREY), label=c, alpha=0.8)
ax[1].legend(fontsize=7, markerscale=1.5); ax[1].set_title("Same projection, coloured by true category")
ax[1].set_xlabel("PC1")
save(fig, "fig_cluster_scatter")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1, 1.2, 1]))
sizes = Counter(R["clusters"])
ks = sorted(sizes)
ax[0].bar([f"C{k}" for k in ks], [sizes[k] for k in ks], color=[pal[k] for k in ks])
for i, k in enumerate(ks):
    ax[0].text(i, sizes[k] + 1, sizes[k], ha="center", fontsize=8)
ax[0].set_title("Cluster sizes")
cont = R["contingency"][0]
ax[1].imshow(cont, cmap="Greens")
for i in range(cont.shape[0]):
    for j in range(cont.shape[1]):
        ax[1].text(j, i, cont[i, j], ha="center", va="center", fontsize=8,
                   color="white" if cont[i, j] > cont.max() / 2 else "black")
ax[1].set_xticks(range(len(classes)), classes, rotation=30); ax[1].set_yticks(range(len(ks)), [f"C{k}" for k in ks])
ax[1].set_title("Cluster x category")
log = R["kmeans"].log_
ax[2].plot([l["iteration"] for l in log], [l["sse"] for l in log], "o-", color=BLUE)
ax2 = ax[2].twinx(); ax2.bar([l["iteration"] for l in log], [l["reassigned"] for l in log], color=ORANGE, alpha=0.4)
ax[2].set_title("Convergence: SSE (line), reassigned (bars)"); ax[2].set_xlabel("iteration")
save(fig, "fig_cluster_stats")

# comparison
fig, ax = plt.subplots(figsize=(7, 3))
names = ["Accuracy", "Macro P", "Macro R", "Macro F1", "Purity", "Rand Index"]
vals = [rep["accuracy"], rep["macro"]["precision"], rep["macro"]["recall"], rep["macro"]["f1"], R["purity"], R["rand_index"]]
bars = ax.bar(names, vals, color=[BLUE] * 4 + [ORANGE] * 2)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.1); ax.set_title("Naive Bayes (blue) vs K-Means (orange), seed 42")
save(fig, "fig_comparison")

# preprocessing experiment
pe = res["preprocessing"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[2, 1]))
labs = [r["config"][0] for r in pe]
metrics = [("nb_accuracy", "NB accuracy"), ("nb_macro_f1", "NB macro-F1"), ("km_purity", "Purity"),
           ("km_rand_index", "Rand Index"), ("rec_precision_at_5", "Rec. P@5")]
w = 0.16
for i, (m, lab) in enumerate(metrics):
    ax[0].bar(np.arange(len(pe)) + (i - 2) * w, [r[m][0] for r in pe], w, yerr=[r[m][1] for r in pe],
              label=lab, capsize=2, color=plt.cm.Set2.colors[i])
ax[0].set_xticks(range(len(pe)), labs); ax[0].set_ylim(0.4, 1); ax[0].legend(fontsize=7, ncol=5, loc="upper center")
ax[0].set_title("Metric by preprocessing configuration (mean ± std, 5 seeds)")
ax[1].bar(np.arange(len(pe)) - 0.2, [r["vocab_before_pruning"] for r in pe], 0.4, label="vocabulary", color=ORANGE)
ax[1].bar(np.arange(len(pe)) + 0.2, [r["indexed_vocab"] for r in pe], 0.4, label="indexed (df>=2)", color=BLUE)
ax[1].set_xticks(range(len(pe)), labs); ax[1].legend(fontsize=7); ax[1].set_title("Vocabulary size")
save(fig, "fig_preprocessing")

# k sweep
ksw = res["k_sweep"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot([r["k"] for r in ksw], [r["purity"] for r in ksw], "o-", label="Purity", color=BLUE)
ax[0].plot([r["k"] for r in ksw], [r["rand_index"] for r in ksw], "s-", label="Rand Index", color=ORANGE)
ax[0].legend(); ax[0].set_xlabel("K"); ax[0].set_title("Purity / RI vs K")
ax[1].plot([r["k"] for r in ksw], [r["sse"] for r in ksw], "o-", color=GREY); ax[1].set_xlabel("K"); ax[1].set_title("SSE vs K")
save(fig, "fig_ksweep")
print("done ->", OUT)
