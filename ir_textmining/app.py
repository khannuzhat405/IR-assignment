"""
FinNews Intelligence -- Streamlit Text Mining & Document Intelligence System
Business context: financial-newswire analysis for an investment-research desk.

Run:  streamlit run app.py
"""
import html
import io
import json
import os
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from textmining.cluster import pca_2d
from textmining.experiments import (CONFIGS, dedup_experiment, duplicate_leakage,
                                    k_sweep, preprocessing_experiment)
from textmining.features import cosine_similarity, score_query_with_index
from textmining.pipeline import cluster_label_map, run_all
from textmining.preprocess import RAW_CONFIG, PreprocessConfig, clean, preprocess, tokenize
from textmining.recommend import explain_overlap, recommend_for_vector, recommend_similar

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(APP_DIR, "data", "reuters_finance_news.csv")
PALETTE = px.colors.qualitative.Safe

st.set_page_config(page_title="FinNews Intelligence", page_icon="📰", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {background: rgba(46,111,149,0.07); border: 1px solid rgba(46,111,149,0.18);
        border-radius: 10px; padding: 10px 14px;}
    .small-note {font-size: 0.86rem; opacity: 0.8;}
    </style>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data helpers
def standardise(df, id_col, text_col, label_col, title_col):
    out = pd.DataFrame()
    out["doc_id"] = df[id_col].astype(str) if id_col != "(auto)" else [f"DOC-{i:05d}" for i in range(len(df))]
    out["title"] = df[title_col].astype(str) if title_col != "(none)" else df[text_col].astype(str).str.slice(0, 70)
    out["text"] = df[text_col].fillna("").astype(str)
    out["category"] = df[label_col].astype(str) if label_col != "(none)" else "unlabelled"
    for c in df.columns:
        if c not in {id_col, text_col, label_col, title_col} and c not in out.columns:
            out[c] = df[c]
    return out


@st.cache_data(show_spinner=False)
def load_default():
    d = pd.read_csv(DEFAULT_DATA).fillna("")
    d["title"] = d["title"].map(html.unescape)  # display only; body text keeps its noise
    return d


def read_upload(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".json") or name.endswith(".jsonl"):
        raw = file.read().decode("utf-8", errors="ignore")
        try:
            data = json.loads(raw)
            return pd.json_normalize(data if isinstance(data, list) else data.get("documents", data))
        except json.JSONDecodeError:
            return pd.read_json(io.StringIO(raw), lines=True)
    if name.endswith(".zip"):  # folder of .txt files; sub-folder name = label
        rows = []
        with zipfile.ZipFile(file) as z:
            for member in z.namelist():
                if member.endswith(".txt") and not member.startswith("__MACOSX"):
                    txt = z.read(member).decode("utf-8", errors="ignore")
                    parts = member.strip("/").split("/")
                    rows.append(dict(doc_id=os.path.splitext(parts[-1])[0],
                                     title=txt.strip().split("\n")[0][:120], text=txt,
                                     category=parts[-2] if len(parts) > 1 else "unlabelled",
                                     source=member))
        return pd.DataFrame(rows)
    raise ValueError("Unsupported file type")


def data_key(df):
    return str(pd.util.hash_pandas_object(df[["doc_id", "text", "category"]], index=False).sum())


@st.cache_resource(show_spinner=False, max_entries=6)
def cached_run(_df, key, cfg, k, test_ratio, seed, dedup, drop_templates, alpha, min_df, max_df, init, n_init):
    return run_all(_df, cfg, k=k, test_ratio=test_ratio, seed=seed, dedup=dedup, drop_templates=drop_templates,
                   alpha=alpha, min_df=min_df, max_df_ratio=max_df, kmeans_init=init, n_init=n_init)


@st.cache_data(show_spinner=False)
def cached_experiments(key, _df, which):
    if which == "prep":
        return preprocessing_experiment(_df, seeds=(0, 1, 2))
    if which == "dedup":
        return dedup_experiment(_df, seeds=(0, 1, 2))
    if which == "leak":
        return duplicate_leakage(_df)
    if which == "k":
        return k_sweep(_df, ks=(3, 4, 5, 6, 7, 8, 10), seeds=(0, 1))


# ------------------------------------------------------------------ session state
if "df" not in st.session_state:
    st.session_state.df = load_default()
    st.session_state.data_name = "Reuters-21578 financial newswire (bundled)"
if "params" not in st.session_state:
    st.session_state.params = dict(lowercase=True, strip_markup=True, remove_numbers=True, remove_stopwords=True,
                                   domain_stopwords=True, normalizer="lemma", ngram_max=1, min_token_len=2,
                                   k=0, test_ratio=0.3, seed=42, dedup=True, drop_templates=False, alpha=1.0,
                                   min_df=2, max_df=0.9, init="k-means++", n_init=5)

df = st.session_state.df
P = st.session_state.params
# one fixed colour per category so every chart uses the same legend
CAT_COLOR = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(df["category"].unique()))}


def current_cfg():
    return PreprocessConfig(lowercase=P["lowercase"], strip_markup=P["strip_markup"],
                            remove_numbers=P["remove_numbers"], remove_stopwords=P["remove_stopwords"],
                            domain_stopwords=P["domain_stopwords"], normalizer=P["normalizer"],
                            ngram_max=P["ngram_max"], min_token_len=P["min_token_len"])


def get_results():
    n_classes = df["category"].nunique()
    k = P["k"] or n_classes
    with st.spinner("Running pipeline: preprocessing → inverted index → TF-IDF → NB → K-Means …"):
        return cached_run(df, data_key(df), current_cfg(), k, P["test_ratio"], P["seed"], P["dedup"],
                          P["drop_templates"], P["alpha"], P["min_df"], P["max_df"], P["init"], P["n_init"])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📰 FinNews Intelligence")
st.sidebar.caption("Text Mining & Document Intelligence System")
PAGES = ["🏠 Dashboard", "📥 Data ingestion", "⚙️ Preprocessing & features", "🔬 Text mining hub",
         "🔗 Recommendations", "📈 Performance analytics"]
page = st.sidebar.radio("Workflow", PAGES)
st.sidebar.divider()
st.sidebar.markdown(f"**Dataset:** {st.session_state.data_name}")
st.sidebar.markdown(f"**Documents:** {len(df)} · **Classes:** {df['category'].nunique()}")
st.sidebar.markdown(f"**Pipeline:** {current_cfg().describe()}")
st.sidebar.caption("TF-IDF, inverted index, Naive Bayes, K-Means, Purity and Rand Index are implemented from scratch.")


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (label, value, help_) in zip(cols, items):
        c.metric(label, value, help=help_)


# ================================================================== DASHBOARD
if page == PAGES[0]:
    st.title("Financial News Intelligence Dashboard")
    st.markdown(
        "An investment-research desk receives hundreds of newswire stories a day. This system **tags** each story "
        "by market topic (Naive Bayes), **groups** stories without labels (K-Means), **flags duplicates and "
        "re-issues**, and **links** each story to related coverage (content-based recommendation).")
    wc = df["text"].str.split().str.len()
    kpi_row([("Documents", f"{len(df):,}", None), ("Categories", df["category"].nunique(), None),
             ("Raw text", f"{df['text'].str.len().sum() / 1e6:.2f} MB", "Characters of body text"),
             ("Mean length", f"{wc.mean():.0f} words", None),
             ("Unique IDs", "✓" if df["doc_id"].is_unique else "✗", "Every document must have a unique ID")])
    c1, c2 = st.columns(2)
    with c1:
        counts = df["category"].value_counts().reset_index()
        counts.columns = ["category", "documents"]
        fig = px.bar(counts, x="category", y="documents", color="category", color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE,
                     title="Documents per category")
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df.assign(words=wc), x="words", color="category", nbins=40, barmode="stack",
                           color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE, title="Document length distribution (words)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    R = get_results()
    st.subheader("Pipeline results at a glance")
    rep = R["report"]
    kpi_row([("NB accuracy", f"{rep['accuracy']:.3f}", "Test-set accuracy of manual Naive Bayes"),
             ("NB macro-F1", f"{rep['macro']['f1']:.3f}", None),
             ("K-Means purity", f"{R['purity']:.3f}", None), ("Rand index", f"{R['rand_index']:.3f}", None),
             (f"Rec. precision@{R['rec_k']}", f"{R['precision_at_k']:.3f}",
              "Share of Top-K recommendations that share the query story's category"),
             ("Duplicates removed", R["n_removed"], None)])
    st.markdown(
        '<p class="small-note">Use the sidebar to walk through the workflow: ingest data → configure preprocessing '
        "→ inspect features → classify / cluster / evaluate → recommend → compare configurations.</p>",
        unsafe_allow_html=True)

# ================================================================== INGESTION
elif page == PAGES[1]:
    st.title("Data ingestion")
    src = st.radio("Source", ["Bundled dataset (Reuters-21578 finance subset)", "Upload my own (CSV / JSON / ZIP of .txt)"],
                   horizontal=True)
    if src.startswith("Upload"):
        up = st.file_uploader("Upload corpus", type=["csv", "json", "jsonl", "zip"])
        st.caption("ZIP: one .txt file per document, placed in a sub-folder named after its category.")
        if up is not None:
            try:
                raw = read_upload(up)
                st.write(f"Read **{len(raw)}** rows and columns: {', '.join(map(str, raw.columns))}")
                cols = list(raw.columns)
                guess = lambda names, default: next((c for c in cols if c.lower() in names), default)
                c1, c2, c3, c4 = st.columns(4)
                id_col = c1.selectbox("Document ID", ["(auto)"] + cols,
                                      index=(["(auto)"] + cols).index(guess({"doc_id", "id"}, "(auto)")))
                text_col = c2.selectbox("Text", cols, index=cols.index(guess({"text", "body", "content"}, cols[0])))
                label_col = c3.selectbox("Category label", ["(none)"] + cols,
                                         index=(["(none)"] + cols).index(guess({"category", "label", "topic"}, "(none)")))
                title_col = c4.selectbox("Title", ["(none)"] + cols,
                                         index=(["(none)"] + cols).index(guess({"title", "headline"}, "(none)")))
                if st.button("Load this dataset", type="primary"):
                    new = standardise(raw, id_col, text_col, label_col, title_col)
                    if not new["doc_id"].is_unique:
                        st.error("Document IDs are not unique - choose another ID column or '(auto)'.")
                    elif new["category"].nunique() < 2:
                        st.error("At least two categories are needed for classification and clustering evaluation.")
                    else:
                        st.session_state.df = new.fillna("")
                        st.session_state.data_name = up.name
                        st.success(f"Loaded {len(new)} documents.")
                        st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read file: {e}")
    else:
        if st.button("Reload bundled dataset"):
            st.session_state.df = load_default()
            st.session_state.data_name = "Reuters-21578 financial newswire (bundled)"
            st.rerun()

    st.subheader("Corpus statistics")
    wc = df["text"].str.split().str.len()
    chars = df["text"].str.len()
    kpi_row([("Documents", len(df), None), ("Categories", df["category"].nunique(), None),
             ("Total words", f"{wc.sum():,}", None), ("Total size", f"{chars.sum() / 1e6:.2f} MB", None),
             ("Shortest / longest", f"{wc.min()} / {wc.max()} words", None)])
    stats = df.assign(words=wc, chars=chars).groupby("category").agg(
        documents=("doc_id", "count"), mean_words=("words", "mean"), median_words=("words", "median"),
        total_chars=("chars", "sum")).round(1)
    st.dataframe(stats, use_container_width=True)

    noisy = df["text"].str.contains(r"<[^>]+>|&lt;|&amp;", regex=True).sum()
    upper = (df["text"].str.count(r"[A-Z]") / df["text"].str.len().clip(lower=1) > 0.5).sum()
    st.markdown(f"**Noise detected:** {noisy} documents contain markup/HTML entities, {upper} are mostly upper-case. "
                "Domain jargon (`mln`, `dlrs`, `pct`, ticker symbols like `<GY>`) is common.")

    st.subheader("Metadata (kept separate from text)")
    meta_cols = [c for c in df.columns if c != "text"]
    st.dataframe(df[meta_cols], use_container_width=True, height=280)
    st.subheader("Document text viewer")
    pick = st.selectbox("Document", df["doc_id"] + " — " + df["title"].str.slice(0, 80))
    row = df[df["doc_id"] == pick.split(" — ")[0]].iloc[0]
    st.markdown(f"**{row['title']}**  \n`{row['doc_id']}` · {row['category']}")
    st.text_area("Raw text", row["text"], height=220)

# ================================================================== PREPROCESSING
elif page == PAGES[2]:
    st.title("Text preprocessing & feature engineering")
    with st.form("prep"):
        st.subheader("Preprocessing configuration")
        c1, c2, c3 = st.columns(3)
        lowercase = c1.checkbox("Lowercase", P["lowercase"])
        strip_markup = c1.checkbox("Strip HTML / markup", P["strip_markup"])
        remove_numbers = c1.checkbox("Remove numbers", P["remove_numbers"])
        remove_stop = c2.checkbox("Remove English stop-words", P["remove_stopwords"])
        domain_stop = c2.checkbox("Remove newswire stop-words (said, mln, dlrs, pct, reuter…)", P["domain_stopwords"])
        normalizer = c2.radio("Normalisation", ["none", "stem", "lemma"], index=["none", "stem", "lemma"].index(P["normalizer"]),
                              format_func={"none": "None", "stem": "Porter stemming", "lemma": "WordNet lemmatization"}.get,
                              horizontal=True)
        ngram = c3.radio("N-grams", [1, 2, 3], index=P["ngram_max"] - 1,
                         format_func={1: "Unigrams", 2: "Uni + bigrams", 3: "Up to trigrams"}.get, horizontal=True)
        min_len = c3.slider("Minimum token length", 1, 4, P["min_token_len"])
        min_df = c3.slider("Min document frequency (index pruning)", 1, 5, P["min_df"])
        max_df = c3.slider("Max document frequency (ratio)", 0.5, 1.0, float(P["max_df"]), 0.05)
        if st.form_submit_button("Apply & rebuild features", type="primary"):
            P.update(lowercase=lowercase, strip_markup=strip_markup, remove_numbers=remove_numbers,
                     remove_stopwords=remove_stop, domain_stopwords=domain_stop, normalizer=normalizer,
                     ngram_max=ngram, min_token_len=min_len, min_df=min_df, max_df=max_df)
            st.rerun()

    R = get_results()
    feats, idx, model = R["feats"], R["feats"]["index"], R["feats"]["model"]
    sub = R["df"]

    st.subheader("Pipeline on one document")
    pick = st.selectbox("Example document", sub["doc_id"] + " — " + sub["title"].str.slice(0, 80), index=0)
    d = int(sub.index[sub["doc_id"] == pick.split(" — ")[0]][0])
    cfg = current_cfg()
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Raw text**")
    c1.code(sub.loc[d, "text"][:600], language=None)
    c2.markdown("**2. Tokenised (before preprocessing)**")
    c2.code(" | ".join(tokenize(clean(sub.loc[d, "text"], RAW_CONFIG))[:70]), language=None)
    c3.markdown("**3. After preprocessing**")
    c3.code(" | ".join(preprocess(sub.loc[d, "text"], cfg)[:70]), language=None)

    st.subheader("Vocabulary before vs after preprocessing")
    rs, ps = R["raw_stats"], R["proc_stats"]
    comp = pd.DataFrame({
        "Before (raw tokens)": [rs["vocab_size"], rs["total_tokens"], rs["hapax"], rs["mean_doc_len"], rs["type_token_ratio"]],
        "After preprocessing": [ps["vocab_size"], ps["total_tokens"], ps["hapax"], ps["mean_doc_len"], ps["type_token_ratio"]],
    }, index=["Vocabulary size", "Total tokens", "Hapax legomena (freq = 1)", "Mean tokens / doc", "Type-token ratio"])
    comp["Change"] = ((comp.iloc[:, 1] / comp.iloc[:, 0] - 1) * 100).map(lambda x: f"{x:+.1f}%")
    st.dataframe(comp.style.format("{:,.3f}", subset=pd.IndexSlice[["Type-token ratio"], comp.columns[:2]])
                 .format("{:,.0f}", subset=pd.IndexSlice[["Vocabulary size", "Total tokens", "Hapax legomena (freq = 1)"], comp.columns[:2]])
                 .format("{:,.1f}", subset=pd.IndexSlice[["Mean tokens / doc"], comp.columns[:2]]),
                 use_container_width=True)
    st.caption(f"Inverted index keeps terms with df ≥ {P['min_df']} and df ≤ {P['max_df']:.0%} of documents: "
               f"**{len(idx):,} indexed terms** ({idx.pruned:,} rare/ubiquitous terms pruned).")
    c1, c2 = st.columns(2)
    for col, stats, title, color in [(c1, rs, "Top 20 terms — before", "#B5651D"), (c2, ps, "Top 20 terms — after", "#2E6F95")]:
        top = pd.DataFrame(stats["counter"].most_common(20), columns=["term", "frequency"])
        fig = px.bar(top[::-1], x="frequency", y="term", orientation="h", title=title)
        fig.update_traces(marker_color=color)
        fig.update_layout(height=480)
        col.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    zipf = []
    for name, stats in [("before", rs), ("after", ps)]:
        f = sorted(stats["counter"].values(), reverse=True)
        zipf.append(pd.DataFrame({"rank": np.arange(1, len(f) + 1), "frequency": f, "stage": name}))
    fig = px.line(pd.concat(zipf), x="rank", y="frequency", color="stage", log_x=True, log_y=True,
                  title="Rank–frequency (Zipf) distribution", color_discrete_sequence=["#B5651D", "#2E6F95"])
    c1.plotly_chart(fig, use_container_width=True)
    lens = pd.DataFrame({"before": [len(tokenize(t)) for t in sub["text"]], "after": [len(t) for t in feats["tokens"]]})
    fig = px.histogram(lens.melt(var_name="stage", value_name="tokens per document"), x="tokens per document",
                       color="stage", barmode="overlay", nbins=50, title="Document length before / after",
                       color_discrete_sequence=["#B5651D", "#2E6F95"])
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Inverted index explorer")
    c1, c2 = st.columns([1, 2])
    df_sorted = sorted(idx.df.items(), key=lambda x: -x[1])
    c1.dataframe(pd.DataFrame([(t, n, round(idx.idf[t], 4)) for t, n in df_sorted[:300]],
                              columns=["term", "df", "idf = log10(N/df)"]), height=330, use_container_width=True)
    term = c2.text_input("Look up a term (after preprocessing)", value=preprocess("oil", cfg)[0] if preprocess("oil", cfg) else "oil")
    q = preprocess(term, cfg)
    t = q[0] if q else term
    post = idx.lookup(t)
    if post:
        c2.markdown(f"`{t}` → df = **{idx.df[t]}**, idf = **{idx.idf[t]:.4f}**, postings (first 25):")
        c2.dataframe(pd.DataFrame([(sub.loc[d_, "doc_id"], sub.loc[d_, "title"][:60], tf,
                                    round((1 + np.log10(tf)) * idx.idf[t], 4), round(model.matrix[d_, idx.term_id[t]], 4))
                                   for d_, tf in post[:25]],
                                  columns=["doc_id", "title", "tf", "w = (1+log tf)·idf", "normalised w"]),
                     use_container_width=True, height=280)
    else:
        c2.info(f"`{t}` is not in the index.")

    st.subheader("TF-IDF document vectors")
    st.latex(r"w_{t,d}=\frac{(1+\log_{10} tf_{t,d})\cdot \log_{10}\frac{N}{df_t}}{\sqrt{\sum_{t'}\left[(1+\log_{10} tf_{t',d})\cdot\log_{10}\frac{N}{df_{t'}}\right]^2}}")
    c1, c2 = st.columns(2)
    tt = pd.DataFrame(model.top_terms(d, 12), columns=["term", "tf-idf"])
    fig = px.bar(tt[::-1], x="tf-idf", y="term", orientation="h", title=f"Top weighted terms in {sub.loc[d, 'doc_id']}")
    fig.update_traces(marker_color="#2E6F95")
    c1.plotly_chart(fig, use_container_width=True)
    c1.caption(f"Vector length before normalisation: {model.lengths[d]:.3f}; after: "
               f"{np.linalg.norm(model.matrix[d]):.3f}. Non-zero terms: {(model.matrix[d] > 0).sum()}.")
    nz = (model.matrix > 0).sum(1)
    fig = px.histogram(pd.DataFrame({"non-zero terms per document": nz}), x="non-zero terms per document", nbins=40,
                       title=f"Feature density (sparsity {(1 - (model.matrix > 0).mean()):.2%})")
    fig.update_traces(marker_color="#6C8EAD")
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Cosine similarity")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Document A", sub["doc_id"], index=0)
    b = c2.selectbox("Document B", sub["doc_id"], index=1)
    ia, ib = int(sub.index[sub["doc_id"] == a][0]), int(sub.index[sub["doc_id"] == b][0])
    st.metric("cos(A, B)", f"{cosine_similarity(model.matrix[ia], model.matrix[ib]):.4f}")
    sample = sub.groupby("category", group_keys=False).head(5).index.tolist()
    S = feats["sim"][np.ix_(sample, sample)]
    fig = px.imshow(S, x=[sub.loc[i, "category"] + ":" + sub.loc[i, "doc_id"][-5:] for i in sample],
                    y=[sub.loc[i, "category"] + ":" + sub.loc[i, "doc_id"][-5:] for i in sample],
                    color_continuous_scale="Blues", zmin=0, zmax=1,
                    title="Cosine similarity heat-map (5 documents per category)")
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== MINING HUB
elif page == PAGES[3]:
    st.title("Text mining & corpus analytics hub")
    with st.expander("Model settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        test_ratio = c1.slider("Test-set ratio", 0.1, 0.5, float(P["test_ratio"]), 0.05)
        alpha = c1.select_slider("Laplace α", [0.1, 0.25, 0.5, 1.0, 2.0], P["alpha"])
        k = c2.slider("K (clusters, 0 = number of classes)", 0, 12, P["k"])
        init = c2.radio("Centroid initialisation", ["k-means++", "random"], index=["k-means++", "random"].index(P["init"]))
        n_init = c3.slider("K-Means restarts (best SSE kept)", 1, 10, P["n_init"])
        seed = c3.number_input("Random seed", 0, 9999, P["seed"])
        dedup = c4.checkbox("Remove exact / near duplicates", P["dedup"])
        drop_t = c4.checkbox("Also remove template duplicates", P["drop_templates"])
        if st.button("Apply settings", type="primary"):
            P.update(test_ratio=test_ratio, alpha=alpha, k=k, init=init, n_init=n_init, seed=int(seed),
                     dedup=dedup, drop_templates=drop_t)
            st.rerun()

    R = get_results()
    sub, y, classes = R["df"], R["labels"], R["classes"]
    t1, t2, t3, t4 = st.tabs(["🧮 Naive Bayes classification", "🧩 K-Means clustering",
                              "⚖️ Evaluation & comparison", "♊ Duplicates"])

    # ---------------- classification
    with t1:
        rep = R["report"]
        st.markdown(f"Multinomial Naive Bayes trained on **{len(R['train_idx'])}** documents, tested on "
                    f"**{len(R['test_idx'])}** (stratified split; duplicate groups kept on one side). "
                    f"Laplace smoothing α = {P['alpha']}.")
        st.latex(r"\hat c=\arg\max_c\Big[\log P(c)+\sum_{t\in d} tf_{t,d}\,\log\frac{count(t,c)+\alpha}{\sum_{t'}count(t',c)+\alpha|V|}\Big]")
        kpi_row([("Accuracy", f"{rep['accuracy']:.3f}", None),
                 ("Macro precision", f"{rep['macro']['precision']:.3f}", None),
                 ("Macro recall", f"{rep['macro']['recall']:.3f}", None),
                 ("Macro F1", f"{rep['macro']['f1']:.3f}", None),
                 ("Weighted F1", f"{rep['weighted']['f1']:.3f}", None)])
        c1, c2 = st.columns(2)
        actual = [y[i] for i in R["test_idx"]]
        dist = pd.DataFrame({"category": classes,
                             "Actual": [actual.count(c) for c in classes],
                             "Predicted": [R["pred"].count(c) for c in classes]}).melt("category", var_name="series",
                                                                                         value_name="documents")
        fig = px.bar(dist, x="category", y="documents", color="series", barmode="group",
                     title="Actual vs predicted class distribution (test set)",
                     color_discrete_sequence=["#2E6F95", "#E09F3E"])
        c1.plotly_chart(fig, use_container_width=True)
        M = rep["confusion"]
        fig = px.imshow(M, x=classes, y=classes, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="docs"), title="Confusion matrix")
        c2.plotly_chart(fig, use_container_width=True)
        pc = pd.DataFrame(rep["per_class"]).T
        pc.loc["macro avg"] = [rep["macro"]["precision"], rep["macro"]["recall"], rep["macro"]["f1"], len(actual)]
        pc.loc["weighted avg"] = [rep["weighted"]["precision"], rep["weighted"]["recall"], rep["weighted"]["f1"], len(actual)]
        st.dataframe(pc.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "support": "{:.0f}"}),
                     use_container_width=True)
        st.markdown("**Test-set predictions**")
        tokens = R["feats"]["tokens"]
        rows = []
        for i, p in zip(R["test_idx"], R["pred"]):
            prob = R["nb"].predict_proba_one(tokens[i])
            rows.append(dict(doc_id=sub.loc[i, "doc_id"], title=sub.loc[i, "title"][:70], actual=y[i], predicted=p,
                             confidence=round(prob[p], 3), correct="✅" if p == y[i] else "❌"))
        pdf = pd.DataFrame(rows)
        only_err = st.checkbox("Show misclassifications only")
        st.dataframe(pdf[pdf.correct == "❌"] if only_err else pdf, use_container_width=True, height=300)
        st.markdown("**Most discriminative terms per class** (log P(t|c) / P(t|¬c))")
        cols = st.columns(len(classes))
        for col, c in zip(cols, classes):
            col.markdown(f"**{c}**")
            col.write(", ".join(t for t, _ in R["nb"].top_features(c, 8)))
        st.markdown("**Classify a new story**")
        new = st.text_area("Paste a headline or story", "OPEC ministers agree to cut crude output to support prices")
        if new.strip():
            prob = R["nb"].predict_proba_one(preprocess(new, current_cfg()))
            pr = pd.DataFrame(sorted(prob.items(), key=lambda x: -x[1]), columns=["category", "posterior"])
            st.plotly_chart(px.bar(pr, x="category", y="posterior", color="category",
                                   color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE).update_layout(showlegend=False, height=280),
                            use_container_width=True)

    # ---------------- clustering
    with t2:
        km = R["kmeans"]
        st.markdown(f"K-Means with **K = {km.k}** on L2-normalised TF-IDF vectors (squared Euclidean distance, which "
                    "for unit vectors equals 2 − 2·cosine). Best of "
                    f"**{km.n_init}** restarts by SSE.")
        st.markdown("**Step 1 — centroid initialisation** "
                    f"({'k-means++: each new seed sampled with probability ∝ squared distance to nearest seed' if km.init == 'k-means++' else 'random documents'})")
        st.dataframe(pd.DataFrame([dict(centroid=j, seed_doc=sub.loc[s, "doc_id"], title=sub.loc[s, "title"][:70],
                                        true_label=y[s]) for j, s in enumerate(km.seed_idx_)]),
                     use_container_width=True)
        st.markdown("**Steps 2–3 — assign documents to nearest centroid, recompute centroids as cluster means**")
        log = pd.DataFrame(km.log_)
        c1, c2 = st.columns([3, 2])
        c1.dataframe(log.assign(sizes=log["sizes"].astype(str)).style.format(
            {"max_centroid_shift": "{:.5f}", "sse": "{:.3f}"}), use_container_width=True, height=280)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log["iteration"], y=log["sse"], name="SSE", mode="lines+markers", line_color="#2E6F95"))
        fig.add_trace(go.Bar(x=log["iteration"], y=log["reassigned"], name="docs reassigned", yaxis="y2",
                             marker_color="rgba(224,159,62,0.6)"))
        fig.update_layout(title="Convergence", yaxis=dict(title="SSE"),
                          yaxis2=dict(title="reassigned", overlaying="y", side="right"), height=300,
                          legend=dict(orientation="h", y=-0.25))
        c2.plotly_chart(fig, use_container_width=True)
        st.success(f"**Step 4 — stopping criterion:** stopped after {km.n_iter_} iterations because "
                   f"{km.stop_reason_}. Restart SSEs: {', '.join(f'{s:.2f}' for s in km.all_run_sse_)}.")

        cmap = cluster_label_map(R["clusters"], y)
        coords, var = R["coords"], R["pca_var"]
        color_by = st.radio("Colour points by", ["K-Means cluster", "True category"], horizontal=True)
        pts = pd.DataFrame(dict(x=coords[:, 0], y=coords[:, 1], doc_id=sub["doc_id"], title=sub["title"].str.slice(0, 60),
                                category=y, cluster=[f"C{c} ({cmap[c][0]})" for c in R["clusters"]]))
        C2 = (km.centroids_ - R["feats"]["model"].matrix.mean(0))
        _, _, Vt = np.linalg.svd(R["feats"]["model"].matrix - R["feats"]["model"].matrix.mean(0), full_matrices=False)
        cen = C2 @ Vt[:2].T
        fig = px.scatter(pts, x="x", y="y", color="cluster" if color_by.startswith("K") else "category",
                         hover_data=["doc_id", "title", "category", "cluster"], color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE,
                         title=f"Documents in 2-D (PCA for display only; {var.sum():.1%} of variance)",
                         category_orders={"cluster": sorted(pts["cluster"].unique())})
        if color_by.startswith("K"):
            fig.add_trace(go.Scatter(x=cen[:, 0], y=cen[:, 1], mode="markers+text", text=[f"C{j}" for j in range(km.k)],
                                     textposition="top center", marker=dict(symbol="x", size=14, color="black"),
                                     name="centroids"))
        fig.update_layout(height=560, xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        sizes = pd.Series(R["clusters"]).value_counts().sort_index()
        bar = pd.DataFrame({"cluster": [f"C{c}" for c in sizes.index], "documents": sizes.values,
                            "majority label": [cmap[c][0] for c in sizes.index]})
        c1.plotly_chart(px.bar(bar, x="cluster", y="documents", color="majority label", text="documents",
                               color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE, title="Cluster sizes"), use_container_width=True)
        cont, ks, cs = R["contingency"]
        c2.plotly_chart(px.imshow(cont, x=cs, y=[f"C{k_}" for k_ in ks], text_auto=True, color_continuous_scale="Teal",
                                  labels=dict(x="True category", y="Cluster", color="docs"),
                                  title="Cluster × category contingency"), use_container_width=True)
        st.markdown("**Top terms per centroid**")
        vocab = R["feats"]["index"].vocab
        st.dataframe(pd.DataFrame([dict(cluster=f"C{j}", size=int(sizes.get(j, 0)), majority=cmap.get(j, ("-", 0))[0],
                                        top_terms=", ".join(vocab[t] for t in np.argsort(-km.centroids_[j])[:10]))
                                   for j in range(km.k)]), use_container_width=True)
        st.markdown("**Cluster membership**")
        mem = pd.DataFrame(dict(doc_id=sub["doc_id"], title=sub["title"], true_label=y,
                                cluster=[f"C{c}" for c in R["clusters"]],
                                cluster_majority=[cmap[c][0] for c in R["clusters"]]))
        f = st.multiselect("Filter clusters", sorted(mem["cluster"].unique()))
        st.dataframe(mem[mem.cluster.isin(f)] if f else mem, use_container_width=True, height=320)

    # ---------------- evaluation
    with t3:
        rep = R["report"]
        st.markdown("Purity and Rand Index are computed with hand-written functions over cluster assignments and true labels.")
        st.latex(r"\text{Purity}=\frac1N\sum_k\max_j|\omega_k\cap c_j|\qquad RI=\frac{TP+TN}{TP+FP+FN+TN}")
        rc = R["ri_counts"]
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame([[rc["TP"], rc["FN"]], [rc["FP"], rc["TN"]]],
                                  index=["same cluster", "different cluster"], columns=["same class", "different class"]),
                     use_container_width=True)
        c1.caption(f"{sum(rc.values()):,} document pairs = N(N−1)/2 with N = {len(y)}.")
        cont, ks, cs = R["contingency"]
        c2.dataframe(pd.DataFrame({"cluster": [f"C{k_}" for k_ in ks], "size": cont.sum(1), "majority count": cont.max(1),
                                   "majority label": [cs[i] for i in cont.argmax(1)]}), use_container_width=True)
        st.subheader("Supervised (Naive Bayes) vs unsupervised (K-Means)")
        comp = pd.DataFrame([
            ["Naive Bayes", "Accuracy", rep["accuracy"]], ["Naive Bayes", "Macro precision", rep["macro"]["precision"]],
            ["Naive Bayes", "Macro recall", rep["macro"]["recall"]], ["Naive Bayes", "Macro F1", rep["macro"]["f1"]],
            ["K-Means", "Purity", R["purity"]], ["K-Means", "Rand Index", R["rand_index"]],
        ], columns=["model", "metric", "value"])
        st.plotly_chart(px.bar(comp, x="metric", y="value", color="model", text=comp["value"].round(3),
                               color_discrete_sequence=["#2E6F95", "#E09F3E"], range_y=[0, 1.05],
                               title="Evaluation metrics"), use_container_width=True)
        s_same = sum((n / len(y)) ** 2 for n in Counter(y).values())  # P(two docs share a class)
        random_ri = s_same / km.k + (1 - s_same) * (1 - 1 / km.k)  # expected RI of a random K-way split
        maj = max(Counter(y).values()) / len(y)
        merged = [f"C{k_} mixes " + " + ".join(cs[j] for j in np.argsort(-cont[i])[:2])
                  for i, k_ in enumerate(ks) if cont[i].max() / cont[i].sum() < 0.75 and cont[i].sum() > 5]
        worst = min(rep["per_class"].items(), key=lambda x: x[1]["f1"])
        st.info(
            f"**Interpretation.** Naive Bayes reaches **{rep['accuracy']:.1%} accuracy** (macro-F1 {rep['macro']['f1']:.3f}) "
            f"because it learns the label definitions from {len(R['train_idx'])} labelled stories; its weakest class is "
            f"**{worst[0]}** (F1 {worst[1]['f1']:.2f}). K-Means sees no labels and reaches **purity {R['purity']:.3f}** — "
            f"well above the {maj:.1%} you would get by putting everything in one cluster — and **Rand Index "
            f"{R['rand_index']:.3f}**. The Rand Index is inflated by the many 'different class, different cluster' pairs "
            f"(TN = {rc['TN']:,}); a random {km.k}-way split would already score about {random_ri:.2f}. "
            + (f"Impure clusters: {'; '.join(merged)} — these topics share vocabulary (e.g. rates and currencies, or tankers and oil), "
               "so an unsupervised grouping by word usage does not match the editorial taxonomy exactly. " if merged else "")
            + "The two are complementary: NB is the right tool when a stable taxonomy exists; K-Means helps discover "
              "themes and sanity-check the taxonomy.")

    # ---------------- duplicates
    with t4:
        pairs = R["dup_pairs"]
        full = st.session_state.df
        kinds = Counter(p["kind"] for p in pairs)
        kpi_row([("Duplicate pairs", len(pairs), None), ("Exact", kinds.get("exact", 0), None),
                 ("Near (re-issued)", kinds.get("near", 0), "5-shingle Jaccard ≥ 0.35 and cosine ≥ 0.90, or Jaccard ≥ 0.60"),
                 ("Template", kinds.get("template", 0), "cosine ≥ 0.90 but Jaccard < 0.35: same report template, different figures"),
                 ("Removed from analysis", R["n_removed"], None)])
        st.markdown("Detection: MD5 hash of normalised text (exact), word 5-shingle **Jaccard** and TF-IDF **cosine** "
                    "(near). Duplicates are grouped with union-find; one representative per group is kept, and during "
                    "the train/test split a whole group always goes to the same side to prevent leakage.")
        if pairs:
            tab = pd.DataFrame([dict(doc_a=full.loc[p["i"], "doc_id"], title_a=full.loc[p["i"], "title"][:55],
                                     doc_b=full.loc[p["j"], "doc_id"], title_b=full.loc[p["j"], "title"][:55],
                                     cosine=round(p["cosine"], 3), jaccard=round(p["jaccard"], 3), kind=p["kind"])
                                for p in pairs]).sort_values(["kind", "cosine"], ascending=[True, False])
            st.dataframe(tab, use_container_width=True, height=380)
            st.plotly_chart(px.scatter(tab, x="jaccard", y="cosine", color="kind", hover_data=["doc_a", "doc_b"],
                                       title="Duplicate pairs: shingle overlap vs TF-IDF cosine",
                                       color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE), use_container_width=True)
            if "injected_dup_of" in full.columns:
                inj = full[full["injected_dup_of"].astype(str) != ""]
                found = {(full.loc[p["i"], "doc_id"], full.loc[p["j"], "doc_id"]) for p in pairs}
                found |= {(b, a) for a, b in found}
                hit = sum((r.doc_id, r.injected_dup_of) in found for r in inj.itertuples())
                st.caption(f"Recall on the {len(inj)} injected near-duplicates: {hit}/{len(inj)}.")

# ================================================================== RECOMMEND
elif page == PAGES[4]:
    st.title("Document recommendation")
    R = get_results()
    sub, model, sim = R["df"], R["feats"]["model"], R["feats"]["sim"]
    st.markdown("**Method: content-based Top-K.** Every story is a length-normalised TF-IDF vector; related stories "
                "are those with the highest cosine similarity. Free-text queries are scored term-at-a-time through "
                "the inverted index.")
    c1, c2, c3 = st.columns([3, 1, 1])
    mode = c1.radio("Query by", ["Select a document", "Enter free text"], horizontal=True)
    k = c2.slider("Top-K", 1, 20, 5)
    hide_dup = c3.checkbox("Hide duplicates of query", True)
    if mode.startswith("Select"):
        options = sub["doc_id"] + " — " + sub["title"].str.slice(0, 90)
        default = int(np.where(sub["doc_id"] == "RTR-06301")[0][0]) if (sub["doc_id"] == "RTR-06301").any() else 0
        pick = st.selectbox("Query document", options, index=default)
        qi = int(sub.index[sub["doc_id"] == pick.split(" — ")[0]][0])
        st.markdown(f"**Query:** {sub.loc[qi, 'title']}  \n`{sub.loc[qi, 'doc_id']}` · category **{sub.loc[qi, 'category']}** · "
                    f"{sub.loc[qi, 'source']}")
        with st.expander("Query text"):
            st.write(sub.loc[qi, "text"])
        excl = None
        if hide_dup:
            g = R["groups"]
            excl = {j for j in range(len(sub)) if g[j] == g[qi] and j != qi}
            excl |= {j for j in range(len(sub)) if sim[qi, j] >= 0.90 and j != qi}  # template duplicates too
        recs = recommend_similar(sim, qi, k, exclude=excl)
    else:
        text = st.text_area("Describe what you are looking for", "Japan trade surplus with the United States and yen intervention")
        qv = model.vectorize(preprocess(text, current_cfg()))
        recs = recommend_for_vector(model, qv, k)
        qi = None
        idx_hits = score_query_with_index(R["feats"]["index"], model, preprocess(text, current_cfg()))
        st.caption(f"Inverted-index scoring touched {len(idx_hits)} candidate documents out of {len(sub)}.")
    if not recs:
        st.warning("No related documents found.")
    else:
        rows = []
        for rank, (j, s) in enumerate(recs, 1):
            why = explain_overlap(model, qi, j, 5) if qi is not None else []
            rows.append(dict(rank=rank, doc_id=sub.loc[j, "doc_id"], title=sub.loc[j, "title"], category=sub.loc[j, "category"],
                             similarity=round(s, 4), source=sub.loc[j, "source"],
                             shared_terms=", ".join(t for t, _ in why),
                             same_category=("✅" if qi is not None and sub.loc[j, "category"] == sub.loc[qi, "category"] else
                                            ("—" if qi is None else "❌"))))
        rec = pd.DataFrame(rows)
        st.dataframe(rec, use_container_width=True, hide_index=True)
        fig = px.bar(rec[::-1], x="similarity", y=rec["rank"].astype(str)[::-1] + ". " + rec["title"].str.slice(0, 45)[::-1],
                     orientation="h", color="category", color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE, title="Recommendation scores")
        fig.update_layout(yaxis_title=None, height=120 + 38 * len(rec))
        st.plotly_chart(fig, use_container_width=True)
        if qi is not None:
            prec = (rec["same_category"] == "✅").mean()
            st.caption(f"{prec:.0%} of these {len(rec)} recommendations share the query's category. Corpus-wide "
                       f"precision@5 = {R['precision_at_k']:.3f}.")
            st.markdown("**Similarity of the query to every document**")
            dist = pd.DataFrame(dict(similarity=np.delete(sim[qi], qi), category=np.delete(np.array(R["labels"]), qi)))
            st.plotly_chart(px.box(dist, x="category", y="similarity", color="category", points="outliers",
                                   color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE).update_layout(showlegend=False, height=330),
                            use_container_width=True)
    with st.expander("How it works & limitations", expanded=False):
        st.markdown("""
* **How:** query vector *q* and document vectors *d* are unit-length TF-IDF vectors, so `score = q · d = cos(q, d)`.
  The Top-K highest scores (excluding the query and its duplicates) are returned with the terms that contribute
  most to the dot product.
* **Strengths:** works from day one with no user history (no cold-start for new stories), transparent explanations,
  cheap to update.
* **Limitations on this dataset:**
  * purely lexical — "yen intervention" and "Bank of Japan buys dollars" only match if they share words; no synonyms;
  * templated reports (weekly API inventories, FHLBB rate tables) look identical after number removal, so they
    crowd each other's lists even though the figures differ;
  * re-issued stories score ~1.0 and must be filtered or they waste recommendation slots;
  * no notion of recency, source authority or individual analyst interests — a collaborative or hybrid
    model would need analyst–story interaction logs, which this corpus does not include.
""")

# ================================================================== ANALYTICS
elif page == PAGES[5]:
    st.title("Performance analytics")
    st.markdown("Controlled experiments that re-run the complete pipeline under different settings. Each figure is "
                "the mean over 3 random seeds (train/test split + K-Means seeding); error bars show ±1 std.")
    key = data_key(df)
    tab1, tab2, tab3 = st.tabs(["Preprocessing comparison", "Duplicates", "Choice of K"])
    with tab1:
        if st.button("Run preprocessing experiment (≈1 min)", type="primary") or st.session_state.get("prep_done"):
            st.session_state.prep_done = True
            with st.spinner("Running 6 configurations × 3 seeds…"):
                rows = cached_experiments(key, df, "prep")
            tab = pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, tuple)},
                                 **{k: v[0] for k, v in r.items() if isinstance(v, tuple)}} for r in rows])
            st.dataframe(tab.drop(columns=["description"]).style.format(precision=3), use_container_width=True)
            long = []
            for r in rows:
                for m, lab in [("nb_accuracy", "NB accuracy"), ("nb_macro_f1", "NB macro-F1"), ("km_purity", "K-Means purity"),
                               ("km_rand_index", "Rand Index"), ("rec_precision_at_5", "Rec. precision@5")]:
                    long.append(dict(config=r["config"][:2].strip(". "), metric=lab, mean=r[m][0], std=r[m][1]))
            long = pd.DataFrame(long)
            st.plotly_chart(px.bar(long, x="config", y="mean", error_y="std", color="metric", barmode="group",
                                   range_y=[0, 1], color_discrete_map=CAT_COLOR, color_discrete_sequence=PALETTE,
                                   title="Effect of preprocessing (A = raw … F = lemma + bigrams)"), use_container_width=True)
            st.plotly_chart(px.bar(tab, x=tab["config"].str.slice(0, 2), y=["vocab_before_pruning", "indexed_vocab"],
                                   barmode="group", title="Vocabulary size per configuration",
                                   color_discrete_sequence=["#B5651D", "#2E6F95"]), use_container_width=True)
            for r in rows:
                st.caption(f"**{r['config']}** — {r['description']}")
    with tab2:
        if st.button("Run duplicate experiments") or st.session_state.get("dup_done"):
            st.session_state.dup_done = True
            with st.spinner("Running…"):
                rows = cached_experiments(key, df, "dedup")
                leak = cached_experiments(key, df, "leak")
            tab = pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, tuple)},
                                 **{k: v[0] for k, v in r.items() if isinstance(v, tuple)}} for r in rows])
            st.dataframe(tab.style.format(precision=3), use_container_width=True)
            kpi_row([("Accuracy on test docs whose twin is in training", f"{leak['leaked_accuracy']:.1%}",
                      f"{leak['leaked_test_docs']} test cases across 10 naive splits"),
                     ("Accuracy on other test docs", f"{leak['clean_accuracy']:.1%}", None),
                     ("Top-1 recommendation is just a duplicate", f"{leak['top1_is_duplicate']:.0%}",
                      f"of the {leak['n_docs_with_dup']} documents that have a duplicate"),
                     ("Duplicate pairs in same cluster", f"{leak['dup_pairs_same_cluster']:.0%}", None)])
    with tab3:
        if st.button("Run K sweep") or st.session_state.get("k_done"):
            st.session_state.k_done = True
            with st.spinner("Running…"):
                rows = pd.DataFrame(cached_experiments(key, df, "k"))
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(rows, x="k", y=["purity", "rand_index"], markers=True, title="Purity / RI vs K",
                                    color_discrete_sequence=["#2E6F95", "#E09F3E"]), use_container_width=True)
            c2.plotly_chart(px.line(rows, x="k", y="sse", markers=True, title="SSE vs K (elbow)",
                                    color_discrete_sequence=["#6C8EAD"]), use_container_width=True)
