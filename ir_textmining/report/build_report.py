"""Builds report/IR_Text_Mining_Report.pdf from results/results.json, results/fig_*.png
and report/screenshots/*.png.   Usage:  python report/build_report.py"""
import json
import os
from collections import Counter

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# ---- edit these lines with your group details -------------------------------------------
COURSE = "AIMLZG537 - Information Retrieval"
AUTHORS = [("Nuzhat Fatima Khan", "2025AG05164")]  # add group members: ("Name", "BITS ID")
GROUP = ""  # e.g. "Group 12"
# -------------------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
M = R["main"]
DF = pd.read_csv(os.path.join(ROOT, "data", "reuters_finance_news.csv")).fillna("")
FIG = lambda n: os.path.join(ROOT, "results", f"fig_{n}.png")
SHOT = lambda n: os.path.join(ROOT, "report", "screenshots", f"{n}.png")
BUILD = os.path.join(ROOT, "report", "_build"); os.makedirs(BUILD, exist_ok=True)
OUT = os.path.join(ROOT, "report", "IR_Text_Mining_Report.pdf")

BLUE = colors.HexColor("#2E6F95"); LIGHT = colors.HexColor("#EEF3F7"); DARK = colors.HexColor("#1F2A36")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=BLUE, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=DARK, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.4, leading=13.2, spaceAfter=5)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=3, spaceAfter=2.5)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, textColor=colors.HexColor("#5A6B7A"), alignment=TA_CENTER, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=7.8, leading=9.6, spaceAfter=0)
TCB = ParagraphStyle("TCB", parent=TC, fontName="Helvetica-Bold", textColor=colors.white)
EQ = ParagraphStyle("EQ", parent=B, alignment=TA_CENTER, fontName="Helvetica-Oblique", spaceBefore=3, spaceAfter=6)
W = A4[0] - 3.6 * cm

story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))
def f3(x): return f"{x:.3f}"
def ms(pair): return f"{pair[0]:.3f} ± {pair[1]:.3f}"


def table(rows, widths=None, header=True, zebra=True, align_right_from=None):
    data = [[Paragraph(str(c), TCB if (header and i == 0) else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DC")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))


def image(path, width=W, caption=None, max_h=22 * cm):
    im = Image.open(path)
    w, h = im.size
    disp_w = width
    disp_h = disp_w * h / w
    if disp_h > max_h:
        disp_h = max_h; disp_w = disp_h * w / h
    items = [RLImage(path, width=disp_w, height=disp_h)]
    if caption:
        items.append(Paragraph(caption, CAP))
    story.append(KeepTogether(items))


def screenshot(name, caption, crop=None, part_h=1650, max_parts=3, max_h=23.5 * cm):
    """Crop the Streamlit sidebar away and split long pages into page-sized parts."""
    im = Image.open(SHOT(name))
    x0 = 300 if im.size[0] >= 1400 else 0
    im = im.crop((x0, 0, im.size[0], im.size[1]))
    if crop:
        im = im.crop((0, crop[0], im.size[0], min(crop[1], im.size[1])))
    from PIL import ImageOps
    bbox = ImageOps.invert(im.convert("L")).point(lambda v: 255 if v > 12 else 0).getbbox()
    if bbox:
        im = im.crop((0, max(0, bbox[1] - 20), im.size[0], min(im.size[1], bbox[3] + 20)))
    parts = []
    y = 0
    while y < im.size[1] and len(parts) < max_parts:
        piece = im.crop((0, y, im.size[0], min(y + part_h, im.size[1])))
        ink = ImageOps.invert(piece.convert("L")).point(lambda v: 255 if v > 12 else 0).getbbox()
        if piece.size[1] > 120 and ink and ink[3] - ink[1] > 150:
            parts.append(piece)
        y += part_h
    for i, piece in enumerate(parts):
        p = os.path.join(BUILD, f"{name}_{i}.png")
        piece.save(p)
        cap = caption + (f" (part {i + 1}/{len(parts)})" if len(parts) > 1 else "")
        image(p, caption=cap, max_h=max_h)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#7A8A99"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "FinNews Intelligence - Text Mining & Document Intelligence System")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


pre = {r["config"][0]: r for r in R["preprocessing"]}
dd = R["dedup"]; lk = R["leakage"]; ksw = R["k_sweep"]
E = pre["E"]
classes = M["classes"]
s_same = sum((n / M["n_after_dedup"]) ** 2 for n in Counter(m["label"] for m in M["membership"]).values())
random_ri = s_same / 6 + (1 - s_same) * (5 / 6)
kinds = Counter(p["kind"] for p in R["dup_pairs"])
cont = M["contingency"]

# =========================================================== TITLE
story.append(Spacer(1, 3.2 * cm))
story.append(Paragraph("FinNews Intelligence", ParagraphStyle("T", parent=H1, fontSize=26, alignment=TA_CENTER, textColor=BLUE)))
story.append(Paragraph("A Streamlit Text Mining &amp; Document Intelligence System for Financial Newswire Analysis",
                       ParagraphStyle("ST", parent=B, fontSize=13, leading=17, alignment=TA_CENTER)))
story.append(Spacer(1, 0.6 * cm))
story.append(Paragraph(COURSE + " &mdash; Text Mining Assignment", ParagraphStyle("c", parent=B, alignment=TA_CENTER)))
story.append(Spacer(1, 1.2 * cm))
rows = [["Name", "BITS ID"]] + [[n, i] for n, i in AUTHORS]
t = Table(rows, colWidths=[7 * cm, 4.5 * cm])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                       ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DC")), ("FONTSIZE", (0, 0), (-1, -1), 10),
                       ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story.append(t)
if GROUP:
    story.append(Spacer(1, 0.3 * cm)); story.append(Paragraph(GROUP, ParagraphStyle("g", parent=B, alignment=TA_CENTER)))
story.append(Spacer(1, 1.5 * cm))
summary = [
    ["Corpus", f"{M['n_docs']} Reuters-21578 stories, 6 topics ({M['n_after_dedup']} after de-duplication)"],
    ["Naive Bayes (5 seeds)", f"accuracy {ms(E['nb_accuracy'])}, macro-F1 {ms(E['nb_macro_f1'])}"],
    ["K-Means, K = 6 (5 seeds)", f"Purity {ms(E['km_purity'])}, Rand Index {ms(E['km_rand_index'])}"],
    ["Recommendation", f"content-based Top-K; precision@5 = {E['rec_precision_at_5'][0]:.3f}"],
    ["Duplicates", f"{len(R['dup_pairs'])} pairs found ({kinds['exact']} exact, {kinds['near']} near, {kinds['template']} template); all 12 injected copies detected"],
]
table([["Headline results", ""]] + summary, widths=[4.2 * cm, W - 4.2 * cm])
story.append(PageBreak())

# =========================================================== 1 CONTEXT
P("1. Selected business context", H1)
P("An <b>investment-research desk</b> at an asset manager receives a continuous stream of financial newswire "
  "stories. Analysts need every story <b>tagged</b> with a market topic so it reaches the right desk (FX, rates, "
  "energy, M&amp;A, trade policy, shipping); they want to <b>discover</b> themes that cut across the editorial taxonomy; "
  "they lose time on <b>re-issued and corrected copies</b> of the same story; and when reading a story they want the "
  "<b>related coverage</b> at hand. The system built here addresses each need: manual Multinomial Naive Bayes for "
  "tagging, manual K-Means for theme discovery, shingle/cosine duplicate detection, and a content-based Top-K "
  "recommender, all driven from a Streamlit front end.")

# =========================================================== 2 DATASET
P("2. Dataset description", H1)
P("The corpus is drawn from <b>Reuters-21578</b> (ApteMod distribution, NLTK), the standard benchmark of real 1987 "
  "Reuters financial newswire. <font face='Courier'>data/prepare_dataset.py</font> selects stories that carry exactly one "
  "topic label and at least 80 words, samples ~60 per topic with a fixed seed, separates the headline from the body, "
  "and writes one CSV in which text and metadata are separate columns.")
counts = DF["category"].value_counts()
wc = DF["text"].str.split().str.len()
rows = [["Topic", "Desk", "Documents", "Mean words", "Median words"]]
names = dict(zip(DF["category"], DF["category_name"]))
for c in sorted(counts.index):
    w_ = wc[DF["category"] == c]
    rows.append([c, names[c], counts[c], f"{w_.mean():.0f}", f"{w_.median():.0f}"])
rows.append(["<b>Total</b>", "", f"<b>{len(DF)}</b>", f"{wc.mean():.0f}", f"{wc.median():.0f}"])
table(rows, widths=[2.4 * cm, 5 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm])
bullets([
    f"<b>Size:</b> {len(DF)} documents, {wc.sum():,} words, {DF['text'].str.len().sum() / 1e6:.2f} MB of body text "
    "(requirement: at least 50-75 documents).",
    "<b>Fields:</b> <i>doc_id</i> (unique, e.g. RTR-06301), <i>title</i>, <i>text</i>, <i>category</i> (label), "
    "<i>category_name</i>, <i>source</i>, <i>split</i> (original ApteMod train/test), <i>word_count</i>, "
    "<i>injected_dup_of</i>. The dashboard and ingestion page display metadata and text separately.",
    "<b>Natural noise:</b> all-capital headlines, HTML entities and ticker markup (<font face='Courier'>&amp;lt;GY&gt;</font>), "
    "newswire jargon (<i>mln, dlrs, pct, bpd, stg</i>), tables of figures, inconsistent spacing.",
    "<b>Duplicates:</b> the corpus contains genuine exact duplicates and re-issued stories, plus templated weekly "
    "reports. To make duplicate handling measurable, 12 synthetic near-duplicates (DUP-001..012) were added that mimic "
    "real syndication patterns: an 'UPDATE 1-' re-issue with the last sentence dropped, an upper-cased copy wrapped in "
    "HTML, and a 'CORRECTED' copy with repeated sentences and punctuation noise. They are flagged in <i>injected_dup_of</i>.",
    "<b>Supports every task:</b> labels for Naive Bayes and for Purity/Rand Index, and text for content-based "
    "recommendation. Collaborative filtering is not used because the corpus has no analyst-story interaction log (Section 9.3).",
])
image(FIG("corpus"), width=W * 0.9, caption="Figure 1 - Category distribution and document length.")

# =========================================================== 3 IMPLEMENTATION
P("3. Implementation approach", H1)
P("The code is split into a small library (<font face='Courier'>textmining/</font>) and the Streamlit front end "
  "(<font face='Courier'>app.py</font>). Every scored algorithm is written by hand; numpy is used only for array "
  "arithmetic and NLTK only for its stop-word list, Porter stemmer and WordNet lemmatizer. scikit-learn is not imported.")
table([
    ["Module", "Implemented manually"],
    ["preprocess.py", "markup stripping, regex tokenizer, lowercasing, number removal, generic + domain stop-words, stemming / lemmatization, n-gram generation, minimum token length"],
    ["features.py", "inverted index (term -> postings [(doc, tf)]), df / idf, TF-IDF weights with cosine length normalisation, cosine similarity, term-at-a-time query scoring through the postings"],
    ["classify.py", "Multinomial Naive Bayes with Laplace smoothing in log space, stratified (duplicate-group aware) train/test split, confusion matrix, precision / recall / F1, macro and weighted averages"],
    ["cluster.py", "K-Means with k-means++ or random seeding, logged assignment / update steps, two stopping rules, multiple restarts; Purity; Rand Index over all document pairs; PCA via SVD (display only)"],
    ["dedup.py", "MD5 for exact duplicates, word 5-shingle Jaccard and TF-IDF cosine for near/template duplicates, union-find grouping"],
    ["recommend.py", "content-based Top-K by cosine, duplicate exclusion, term-level explanations, precision@K"],
    ["experiments.py", "controlled comparisons over preprocessing settings, de-duplication modes and K, averaged over random seeds"],
], widths=[2.8 * cm, W - 2.8 * cm])
P("<b>Streamlit workflow.</b> Six pages in the sidebar follow the analyst's workflow: <i>Dashboard</i> (corpus "
  "overview and headline metrics) &rarr; <i>Data ingestion</i> (bundled corpus or upload of CSV / JSON / ZIP of .txt "
  "with column mapping, ID-uniqueness check and corpus statistics) &rarr; <i>Preprocessing &amp; features</i> "
  "(configuration form, before/after vocabulary, inverted-index explorer, TF-IDF vectors, similarity heat-map) &rarr; "
  "<i>Text mining hub</i> (Naive Bayes, K-Means with every step, evaluation and comparison, duplicates) &rarr; "
  "<i>Recommendations</i> &rarr; <i>Performance analytics</i> (re-runs the pipeline under alternative settings). "
  "Results are cached per configuration, so changing a setting re-runs only what is needed. Screenshots are in the Appendix.")

# =========================================================== 4 PREPROCESSING
P("4. Preprocessing and feature choices", H1)
P("<b>Chosen configuration (E):</b> strip markup and decode entities &rarr; regex tokenisation &rarr; lowercase &rarr; drop "
  "numbers &rarr; remove English stop-words plus newswire boiler-plate (<i>said, mln, dlrs, pct, reuter, inc, corp</i>) "
  "&rarr; WordNet lemmatization (noun then verb) &rarr; tokens of length &ge; 2 &rarr; unigrams. The inverted index keeps "
  "terms with df &ge; 2 and df &le; 90% of documents. Section 9.1 justifies this choice experimentally.")
raw, pr = M["raw"], M["processed"]
table([
    ["Statistic", "Before (raw tokens)", "After preprocessing", "Change"],
    ["Vocabulary size", f"{raw['vocab_size']:,}", f"{pr['vocab_size']:,}", f"{(pr['vocab_size'] / raw['vocab_size'] - 1) * 100:+.1f}%"],
    ["Total tokens", f"{raw['total_tokens']:,}", f"{pr['total_tokens']:,}", f"{(pr['total_tokens'] / raw['total_tokens'] - 1) * 100:+.1f}%"],
    ["Hapax legomena (freq = 1)", f"{raw['hapax']:,}", f"{pr['hapax']:,}", f"{(pr['hapax'] / raw['hapax'] - 1) * 100:+.1f}%"],
    ["Mean / median tokens per doc", f"{raw['mean_doc_len']:.0f} / {raw['median_doc_len']:.0f}", f"{pr['mean_doc_len']:.0f} / {pr['median_doc_len']:.0f}", ""],
    ["Indexed terms (df >= 2, df <= 90%)", "", f"{M['indexed_vocab']:,}", f"{M['pruned']:,} pruned"],
], widths=[5.2 * cm, 3.6 * cm, 3.6 * cm, 3 * cm])
image(FIG("vocab"), caption="Figure 2 - Most frequent terms before and after preprocessing, and the rank-frequency (Zipf) curves. "
      "Before, the head is dominated by function words and 'said', 'mln', 'dlrs'; after, by topical terms (oil, rate, trade, bank, share).")
P("<b>TF-IDF (SMART ltc), computed from the inverted index without libraries.</b> For term t in document d:", B)
P("w(t,d) = (1 + log<sub>10</sub> tf<sub>t,d</sub>) &times; log<sub>10</sub>(N / df<sub>t</sub>),&nbsp;&nbsp; "
  "w'(t,d) = w(t,d) / sqrt( &Sigma;<sub>t'</sub> w(t',d)<super>2</super> )", EQ)
P("The postings lists provide tf and df; idf is computed once per term; each document vector is then divided by its "
  "Euclidean length so every story has unit length regardless of how long it is. Document similarity is the cosine, "
  "cos(q,d) = q&middot;d / (|q||d|), which reduces to a dot product for unit vectors. Free-text queries are vectorised "
  "with the corpus idf and scored term-at-a-time by walking only the postings of the query terms.")

# =========================================================== 5 CLASSIFICATION
P("5. Classification results - Multinomial Naive Bayes", H1)
P(f"The model estimates log P(c) and log P(t|c) = log[(count(t,c) + &alpha;) / (&Sigma;<sub>t'</sub> count(t',c) + &alpha;|V|)] "
  f"with &alpha; = 1 and predicts argmax<sub>c</sub> [log P(c) + &Sigma;<sub>t</sub> tf(t,d) log P(t|c)]. The split is "
  f"stratified 70/30 and keeps every duplicate group on one side, giving {M['train']} training and {M['test']} test "
  f"documents. Results below are for the default seed (42); averaged over 5 seeds the accuracy is "
  f"<b>{ms(E['nb_accuracy'])}</b> and macro-F1 <b>{ms(E['nb_macro_f1'])}</b>, so seed 42 sits at the lower end.")
rows = [["Class", "Precision", "Recall", "F1", "Support"]]
for c in classes:
    v = M["per_class"][c]; rows.append([c, f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), v["support"]])
rows.append(["<b>Macro avg</b>", f3(M["macro"]["precision"]), f3(M["macro"]["recall"]), f3(M["macro"]["f1"]), M["test"]])
rows.append(["<b>Weighted avg</b>", f3(M["weighted"]["precision"]), f3(M["weighted"]["recall"]), f3(M["weighted"]["f1"]), M["test"]])
rows.append(["<b>Accuracy</b>", "", "", f"<b>{f3(M['accuracy'])}</b>", M["test"]])
table(rows, widths=[3.5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm])
image(FIG("classification"), caption="Figure 3 - Actual vs predicted class distribution on the test set, and the confusion matrix (rows = actual).")
cm_ = M["confusion"]
P(f"<b>Reading the errors.</b> <i>acq</i> is the easiest class (F1 {f3(M['per_class']['acq']['f1'])}): takeover stories share a "
  f"distinctive vocabulary (stake, acquisition, shareholder). The hardest pair is <i>interest</i> vs <i>money-fx</i>: "
  f"{cm_[2][3]} interest stories were predicted as money-fx and {cm_[3][2]} money-fx stories as interest, because central-bank "
  f"stories discuss both rates and currencies. <i>ship</i> has high precision but lower recall; some shipping stories "
  f"about Gulf tankers and port strikes are pulled towards <i>crude</i>.")
P("<b>Sample predictions</b> (test set; all predictions are listed in the app and in results.json):", B)
preds = M["predictions"]
sample = [p for p in preds if p["actual"] != p["predicted"]][:7] + [p for p in preds if p["actual"] == p["predicted"]][:7]
rows = [["doc_id", "Title", "Actual", "Predicted", ""]]
for p in sample:
    rows.append([p["doc_id"], p["title"][:62], p["actual"], p["predicted"], "correct" if p["actual"] == p["predicted"] else "wrong"])
table(rows, widths=[2 * cm, 8.3 * cm, 1.9 * cm, 1.9 * cm, 1.4 * cm])
tf = M["top_features"]
P("<b>Most discriminative terms per class</b> (highest log P(t|c)/P(t|not c)): " + "; ".join(
    f"<i>{c}</i>: {', '.join(t for t, _ in tf[c][:5])}" for c in classes) + ".")

# =========================================================== 6 CLUSTERING
P("6. Clustering results - K-Means", H1)
P("K-Means runs on the unit-length TF-IDF vectors with squared Euclidean distance (for unit vectors "
  "||x - c||<super>2</super> = 2 - 2cos, so it groups by cosine similarity). K is set to the number of topics (6). "
  "The four steps of the algorithm are logged and shown in the app:")
bullets([
    "<b>Initialisation (k-means++):</b> the first seed is a random document; each next seed is sampled with probability "
    "proportional to its squared distance from the nearest seed already chosen. Seeds of the best run: " + ", ".join(M["kmeans_seeds"]) + ".",
    "<b>Assignment:</b> every document goes to the centroid with the smallest squared distance.",
    "<b>Update:</b> each centroid becomes the mean of its member vectors (an empty cluster would be re-seeded with the farthest document).",
    f"<b>Stopping:</b> stop when no document changes cluster, or the largest centroid shift is below 10<super>-6</super>, or after 100 "
    f"iterations. The best run stopped after {len(M['kmeans_log'])} iterations because {M['kmeans_stop']}. "
    f"Five restarts were made and the lowest-SSE run kept (SSEs: {', '.join(f'{s:.1f}' for s in M['kmeans_restart_sse'])}).",
])
rows = [["Iteration", "Docs reassigned", "Max centroid shift", "SSE", "Cluster sizes"]]
for l in M["kmeans_log"]:
    rows.append([l["iteration"], l["reassigned"], f"{l['max_centroid_shift']:.5f}", f"{l['sse']:.2f}", str(l["sizes"])])
table(rows, widths=[2 * cm, 2.8 * cm, 3.2 * cm, 2.4 * cm, 5 * cm])
image(FIG("cluster_scatter"), caption=f"Figure 4 - Documents projected to 2-D with PCA (visualisation only, PCA computed by numpy SVD; "
      f"the first two components explain {sum(M['pca_var']) * 100:.1f}% of variance). Left: K-Means clusters; right: true topics.")
image(FIG("cluster_stats"), caption="Figure 5 - Cluster sizes, cluster x topic contingency table, and convergence of SSE / reassignments.")
cmap = M["cluster_map"]
rows = [["Cluster", "Size", "Majority topic (share)", "Top centroid terms"]]
for k in sorted(cmap, key=int):
    rows.append([f"C{k}", sum(cont[int(k)]), f"{cmap[k][0]} ({cmap[k][1]:.0%})", ", ".join(M["centroid_terms"][int(k)][:8])])
table(rows, widths=[1.6 * cm, 1.4 * cm, 3.6 * cm, W - 6.6 * cm])
P(f"<b>Evaluation.</b> Purity = (1/N) &Sigma;<sub>k</sub> max<sub>j</sub> |&omega;<sub>k</sub> &cap; c<sub>j</sub>| = "
  f"<b>{f3(M['purity'])}</b>. Rand Index = (TP + TN)/(TP + FP + FN + TN) over all {sum(M['ri_counts'].values()):,} pairs "
  f"(TP = {M['ri_counts']['TP']:,}, FP = {M['ri_counts']['FP']:,}, FN = {M['ri_counts']['FN']:,}, TN = {M['ri_counts']['TN']:,}) "
  f"= <b>{f3(M['rand_index'])}</b>. Both are computed by hand-written functions. Over 5 seeds: Purity {ms(E['km_purity'])}, "
  f"RI {ms(E['km_rand_index'])}.")
P(f"<b>What the clusters found.</b> C3 is an almost pure M&amp;A cluster. C0 mixes <i>crude</i> ({cont[0][1]}) and <i>ship</i> "
  f"({cont[0][4]}): in 1987 much shipping news was about tankers attacked in the Gulf and oil-port strikes, so the "
  f"unsupervised view sees one 'oil and tankers' theme where the taxonomy has two. C1 joins <i>interest</i> and "
  f"<i>money-fx</i> (central banks), and C4 joins <i>trade</i> with <i>money-fx</i> (the US-Japan trade dispute and the yen). "
  f"C2 is a 6-document cluster of short-term discount-note rate tables (five FHLBB, one Sallie Mae): these templated reports are near-identical after number removal and "
  f"capture a centroid on their own (see Section 9.2). C5 collects the remaining shipping and offshore stories.")
P("<b>Cluster membership</b> (first 14 documents; the full table is in the app and results.json):", B)
rows = [["doc_id", "Title", "True label", "Cluster", "Cluster majority"]]
for m in M["membership"][:14]:
    rows.append([m["doc_id"], m["title"][:58], m["label"], f"C{m['cluster']}", cmap[str(m["cluster"])][0]])
table(rows, widths=[2 * cm, 8 * cm, 2 * cm, 1.6 * cm, 2.3 * cm])

# =========================================================== 7 COMPARISON
P("7. Comparison of classification and clustering", H1)
image(FIG("comparison"), width=W * 0.75, caption="Figure 6 - Naive Bayes metrics and K-Means Purity / Rand Index (seed 42).")
table([
    ["", "Naive Bayes (supervised)", "K-Means (unsupervised)"],
    ["Uses labels?", f"Yes - {M['train']} labelled training stories", "No - labels only used afterwards for evaluation"],
    ["Metrics", "Accuracy, precision, recall, F1, confusion matrix", "Purity, Rand Index, contingency table"],
    ["Seed 42", f"Accuracy {f3(M['accuracy'])}, macro-F1 {f3(M['macro']['f1'])}", f"Purity {f3(M['purity'])}, RI {f3(M['rand_index'])}"],
    ["Mean ± std, 5 seeds", f"Accuracy {ms(E['nb_accuracy'])}, macro-F1 {ms(E['nb_macro_f1'])}", f"Purity {ms(E['km_purity'])}, RI {ms(E['km_rand_index'])}"],
    ["Reference point", "Majority-class guess = 17%", f"One cluster: purity 17%; random 6-way split: RI about {random_ri:.2f}"],
    ["Main confusion", "interest vs money-fx", "crude + ship merged; interest + money-fx merged"],
    ["Stability", f"std {E['nb_accuracy'][1]:.3f} (split only)", f"std {E['km_purity'][1]:.3f} (seeding + split of local minima)"],
], widths=[3.3 * cm, (W - 3.3 * cm) / 2, (W - 3.3 * cm) / 2])
P("The supervised classifier is clearly better at reproducing the editorial taxonomy, as expected: it is told what the "
  "classes mean. K-Means has to infer structure from word co-occurrence alone; it recovers the M&amp;A topic almost "
  "perfectly, but it groups by <i>theme</i> rather than by desk, merging topics whose stories share vocabulary. Purity "
  "and accuracy are not directly comparable (purity lets several clusters share a majority label and ignores small "
  "classes), and the Rand Index looks high partly because most pairs are 'different class, different cluster' "
  f"(TN = {M['ri_counts']['TN']:,} of {sum(M['ri_counts'].values()):,}); against a random baseline of about {random_ri:.2f} the "
  "gain is real but moderate. The two methods are complementary: Naive Bayes for routing stories into a fixed taxonomy, "
  "K-Means for spotting themes that cut across it.")

# =========================================================== 8 RECOMMENDATION
P("8. Recommendation results", H1)
P("<b>Approach: content-based Top-K.</b> A query story (or a free-text description) is represented by its "
  "unit-length TF-IDF vector; every other story is scored by cosine similarity and the K highest are returned with "
  "their score, topic, source and the terms that contribute most to the dot product. The query's own duplicates "
  "(same duplicate group, or cosine &ge; 0.90) are excluded, because they would otherwise occupy the top slots. "
  "Free-text queries are scored through the inverted index, touching only documents that share a query term.")
for qid, v in R["recommendations"].items():
    P(f"<b>Query {qid}</b> - <i>{v['title']}</i> (topic: {v['category']})", B)
    rows = [["#", "doc_id", "Title", "Topic", "Cosine", "Top shared terms"]]
    for i, r_ in enumerate(v["recs"], 1):
        rows.append([i, r_["doc_id"], r_["title"][:46], r_["category"], f"{r_['score']:.3f}", ", ".join(r_["terms"][:4])])
    table(rows, widths=[0.7 * cm, 1.9 * cm, 6 * cm, 1.7 * cm, 1.4 * cm, W - 11.7 * cm])
u = R["recommendations"]["RTR-06301"]["unfiltered_top1"]
screenshot("08_recommendation", "Figure 7 - Recommendation interface: query selection, Top-K slider, duplicate filter, "
           "result table with scores and metadata, and score chart", crop=(0, 1040), max_h=13 * cm)
P("<b>Findings.</b>")
bullets([
    f"Across the whole corpus, {E['rec_precision_at_5'][0]:.1%} of Top-5 recommendations share the query's topic "
    "(precision@5, used as an offline proxy since no relevance judgements exist). For the three queries above, "
    "14 of 15 recommendations are on-topic, and the one 'miss' (RTR-03449, labelled <i>interest</i>, for a Bank of Japan "
    "intervention story) is in fact highly relevant - same speaker, same policy - which shows the label-based proxy "
    "understates usefulness.",
    f"Without the duplicate filter, the top hit for RTR-06301 would be {u['doc_id']} (cosine {u['score']:.3f}): the "
    "previous week's API bulletin, which uses identical wording with different figures. Every one of the "
    f"{lk['n_docs_with_dup']} documents that has a duplicate would otherwise get that duplicate as its top recommendation.",
    "Scores for genuinely related but distinct stories are low in absolute terms (0.15-0.30) because short wire stories "
    "share few terms; ranking, not the absolute value, is what matters.",
])
P("<b>Limitations on this dataset.</b>")
bullets([
    "Purely lexical: synonyms and paraphrases (\"intervene\" vs \"buy dollars\") are missed; there is no semantic model.",
    "Templated reports look identical once numbers are removed; the recommender cannot tell this week's figures from last week's.",
    "No recency, source authority or personalisation; every analyst receives the same list.",
    "Short stories have sparse vectors, so a few shared names (Sumita, Paris) can dominate the score.",
])

# =========================================================== 9 INFERENCES
P("9. Inference and discussion", H1)
P("All experiments re-run the complete pipeline and report the mean (and standard deviation) over 5 random seeds, which "
  "change both the train/test split and K-Means seeding. This matters: on a 108-document test set a single split can "
  "move accuracy by &plusmn;4 points.", B)

P("9.1 How did preprocessing affect vocabulary, classification, clustering and recommendation?", H2)
rows = [["", "Steps", "Vocab", "Index terms", "NB accuracy", "NB macro-F1", "Purity", "Rand Index", "Rec P@5"]]
for k_, r_ in pre.items():
    rows.append([k_, r_["config"][3:], f"{r_['vocab_before_pruning']:,}", f"{r_['indexed_vocab']:,}", ms(r_["nb_accuracy"]),
                 ms(r_["nb_macro_f1"]), ms(r_["km_purity"]), ms(r_["km_rand_index"]), f"{r_['rec_precision_at_5'][0]:.3f}"])
table(rows, widths=[0.6 * cm, 3.4 * cm, 1.3 * cm, 1.4 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 1.3 * cm])
image(FIG("preprocessing"), caption="Figure 8 - Effect of each preprocessing configuration (error bars: ±1 std over 5 seeds).")
A, Bc, C, D, F = pre["A"], pre["B"], pre["C"], pre["D"], pre["F"]
bullets([
    f"<b>Vocabulary.</b> Lowercasing and removing markup/numbers cut the vocabulary from {A['vocab_before_pruning']:,} to "
    f"{Bc['vocab_before_pruning']:,} types (-{(1 - Bc['vocab_before_pruning'] / A['vocab_before_pruning']) * 100:.0f}%), mostly by merging "
    f"'OIL'/'Oil'/'oil' and dropping figures. Stop-word removal barely changes the number of types but removes "
    f"{(1 - C['tokens'] / Bc['tokens']) * 100:.0f}% of all tokens. Stemming shrinks the vocabulary further to {D['vocab_before_pruning']:,} "
    f"(lemmatization: {E['vocab_before_pruning']:,}, gentler); adding bigrams multiplies it to {F['vocab_before_pruning']:,}.",
    f"<b>Classification.</b> The largest gain comes from stop-word removal ({ms(Bc['nb_accuracy'])} &rarr; {ms(C['nb_accuracy'])}): in a "
    "multinomial model, frequent function words contribute large, uninformative likelihood terms. Stemming or lemmatization "
    f"adds about one more point ({D['nb_accuracy'][0]:.3f} / {E['nb_accuracy'][0]:.3f}) by pooling evidence across word forms, within one "
    f"standard deviation. Bigrams did not help ({F['nb_accuracy'][0]:.3f}): with ~250 training stories most bigrams occur once and only add noise.",
    f"<b>Clustering</b> is the most sensitive task. Normalising case and markup lifted purity from {A['km_purity'][0]:.3f} to "
    f"{Bc['km_purity'][0]:.3f} (raw all-caps headlines and HTML copies otherwise form their own groups). Lemmatization gave the best "
    f"purity and RI ({ms(E['km_purity'])}, {ms(E['km_rand_index'])}). Bigrams hurt badly ({F['km_purity'][0]:.3f}): more dimensions, sparser "
    "vectors and smaller distances between documents make the centroids less distinct. Differences among C-E are inside the "
    "seed-to-seed variation, so they should be read as trends, not firm rankings.",
    f"<b>Recommendation</b> precision@5 is stable (0.71-0.73) across all settings: TF-IDF already down-weights common words, so "
    "the top neighbours are driven by rare, topical terms in every configuration. Preprocessing mainly improves the "
    "<i>explanations</i> (shared terms such as 'refinery, bpd' instead of 'the, said').",
    "<b>Choice:</b> configuration E (lemmatization) gives the best balance: top or near-top on every task, readable terms "
    "in the inverted index and explanations, and a vocabulary half the size of the raw one.",
])

P("9.2 How do duplicates affect the results, and how were they detected and handled?", H2)
P(f"<b>Detection.</b> Exact duplicates: MD5 of lowercased, markup-free, punctuation-free text. Near duplicates: TF-IDF "
  f"cosine &ge; 0.90 or word 5-shingle Jaccard &ge; 0.60. Pairs with high cosine but Jaccard &lt; 0.35 are labelled "
  f"<i>template</i>. Union-find merges pairs into groups. Result: {len(R['dup_pairs'])} pairs - {kinds['exact']} exact (two "
  f"genuine Reuters duplicates), {kinds['near']} near (all 12 injected copies, i.e. recall 12/12, plus one real re-issue: "
  "'PARTNERS CALL GENCORP RESPONSE UNPRODUCTIVE' sent twice with re-worded sentences), and "
  f"{kinds['template']} template pairs (five weekly FHLBB discount-note rate tables and two weekly API inventory bulletins).")
rows = [["Effect when duplicates are kept", "Evidence"]]
rows += [
    ["Classification: train/test leakage", f"With a naive split, test stories whose twin was in training were classified "
     f"{lk['leaked_accuracy']:.0%} correctly ({lk['leaked_test_docs']} cases over 10 splits) versus {lk['clean_accuracy']:.1%} for "
     "the others - the model memorises rather than generalises, inflating accuracy."],
    ["Recommendation: wasted slots", f"For {lk['top1_is_duplicate']:.0%} of the {lk['n_docs_with_dup']} documents that have a "
     "duplicate, the Top-1 recommendation was that duplicate (cosine 0.87-1.00), which tells the analyst nothing new. "
     "Precision@5 even looks better with duplicates kept (0.728 vs 0.721) because a copy always has the same label - a misleading gain."],
    ["Clustering: distorted centroids", f"{lk['dup_pairs_same_cluster']:.0%} of duplicate pairs land in the same cluster. "
     "Copies add weight to one story's vocabulary and templated groups can capture a whole centroid (cluster C2 in Section 6)."],
]
table(rows, widths=[4.5 * cm, W - 4.5 * cm])
rows = [["Mode", "Docs", "NB accuracy", "NB macro-F1", "Purity", "Rand Index", "P@5"]]
for r_ in dd:
    rows.append([r_["mode"], r_["docs"], ms(r_["nb_accuracy"]), ms(r_["nb_macro_f1"]), ms(r_["km_purity"]),
                 ms(r_["km_rand_index"]), f"{r_['rec_precision_at_5'][0]:.3f}"])
table(rows, widths=[4.4 * cm, 1.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.4 * cm])
P("<b>Handling.</b> (1) One representative per exact/near group is kept (15 documents removed). (2) The train/test split "
  "is group-aware, so a story and its copy can never sit on opposite sides. (3) Template duplicates are kept for "
  "classification and clustering because they are distinct events with different figures, but they are flagged and "
  "excluded from a query's recommendation list. Removing them as well gives the highest Rand Index "
  f"({dd[2]['km_rand_index'][0]:.3f}) and lowest variance, so a deployment could also collapse them into one 'series' item. "
  "The averaged accuracy differences between modes are small on this corpus because duplicates are only ~4-8% of it; the "
  "leakage measurement above shows the effect clearly where it occurs.")

P("9.3 Recommendation approach: content-based vs collaborative", H2)
P("The application uses <b>content-based</b> recommendation because the corpus contains text and labels but no record "
  "of which analyst read, saved or forwarded which story. Each story's profile is its TF-IDF vector and similarity is cosine.")
table([
    ["", "Content-based (used)", "Collaborative filtering"],
    ["Data required", "Only the documents (text, optional metadata)", "A user x item interaction matrix: reads, clicks, ratings, dwell time"],
    ["Cold start", "New stories can be recommended immediately - essential for news", "New stories have no interactions, so cannot be recommended; new users have no history"],
    ["What it captures", "Topical / lexical similarity; explainable by shared terms", "Behavioural relatedness: 'analysts who read X also read Y', even with no shared words"],
    ["Weaknesses", "Over-specialisation (more of the same), no synonyms, no notion of quality or popularity", "Sparsity, popularity bias, needs many active users, harder to explain"],
    ["Preferable when", "Items are new and short-lived, text is rich, few users (this desk)", "Large stable user base and catalogue with long histories (e.g. research-report portal)"],
], widths=[2.8 * cm, (W - 2.8 * cm) / 2, (W - 2.8 * cm) / 2])
P("A <b>hybrid</b> is the natural next step for a real desk: content-based scores for fresh stories, blended with "
  "collaborative signals from the team's reading logs once they exist, plus a recency boost.")

P("9.4 How do these techniques together help domain experts decide?", H2)
bullets([
    f"<b>Routing and triage (classification):</b> Naive Bayes tags incoming stories with about {E['nb_accuracy'][0]:.0%} accuracy, "
    "so most stories reach the right desk automatically; low-confidence predictions (shown in the app) can be sent for a human check.",
    "<b>Theme discovery (clustering):</b> K-Means surfaced cross-desk themes the taxonomy hides - 'oil + tanker security' and "
    "'trade friction + currency' - which is exactly where an FX or energy analyst should look for spill-over risk.",
    "<b>Noise reduction (duplicates):</b> collapsing re-issues and corrected copies reduces reading load and prevents a single "
    "story from being counted as several signals.",
    "<b>Context (recommendation):</b> opening a story shows related coverage with the shared terms that justify the link "
    "(e.g. the Bank of Japan intervention story links to Paris Accord statements by Miyazawa and Baker), giving the analyst "
    "background in one click.",
    "<b>Corpus analytics:</b> vocabulary, term and length statistics show what the feed is talking about and help tune "
    "domain stop-words (mln, dlrs, pct) and thresholds.",
])

P("9.5 Naive Bayes vs K-Means, key findings, limitations and improvements", H2)
P(f"Using the appropriate metrics, Naive Bayes reaches accuracy {ms(E['nb_accuracy'])} and macro-F1 {ms(E['nb_macro_f1'])}; "
  f"K-Means reaches Purity {ms(E['km_purity'])} and Rand Index {ms(E['km_rand_index'])}. The supervised model reproduces the "
  "editorial taxonomy far better; the unsupervised model recovers it partially and reveals where the taxonomy's classes "
  "overlap in language.")
P("<b>Key experimental findings</b>")
bullets([
    "Stop-word removal is the most valuable step for Naive Bayes; case/markup normalisation and lemmatization matter most for clustering; bigrams hurt both on a corpus this small.",
    "The confusions of both models coincide (interest/money-fx, crude/ship), so they reflect genuine topic overlap, not a modelling bug.",
    f"K-Means is unstable: purity ranged roughly from 0.50 to 0.78 across seeds while the SSE varied by less than 2%. A lower SSE did not guarantee a partition closer to the taxonomy.",
    f"K = 6 gave the best average purity ({max(ksw, key=lambda r: r['purity'])['purity']:.3f}) and RI in the K sweep (Figure 9), even though SSE keeps falling with K - the elbow is weak in sparse text.",
    "Duplicates cause train/test leakage (100% vs 85% accuracy) and fill recommendation lists with copies; group-aware splitting and filtering remove both problems.",
])
image(FIG("ksweep"), width=W * 0.8, caption="Figure 9 - Purity, Rand Index and SSE as a function of K (mean of 3 seeds).")
P("<b>Limitations</b>")
bullets([
    "Modest data: 357 stories (about 60 per topic) from 1987; vocabulary and topics differ from today's newswire.",
    "Single-label subset only: real Reuters stories often carry several topics (e.g. money-fx and interest together), which the single-label evaluation penalises.",
    "Bag-of-words features ignore word order, negation and meaning; TF-IDF cannot link synonyms.",
    "Recommendation is evaluated with a label-based proxy (precision@5), not with analyst relevance judgements.",
    "K-Means assumes roughly spherical, similar-sized clusters and is sensitive to seeding; Purity and RI favour some partitions artificially (e.g. Purity rises with K).",
])
P("<b>Suggested improvements for real-world deployment</b>")
bullets([
    "Train on the full Reuters/current feed with multi-label classification (one-vs-rest NB) and calibrated confidence thresholds for human review.",
    "Add dense sentence embeddings (or LSA) for clustering and recommendation to capture synonyms; keep TF-IDF for explanations.",
    "Use more K-Means restarts or spherical K-Means/bisecting K-Means, and report Adjusted Rand Index and NMI alongside Purity and RI.",
    "Run de-duplication incrementally (MinHash/LSH on shingles) at ingestion, and collapse templated series into one timeline item.",
    "Log analyst clicks and saves to enable a hybrid recommender, add recency and source weighting, and evaluate with real relevance feedback.",
    "Incremental index updates and scheduled retraining as the feed grows; monitor topic drift.",
])

# =========================================================== APPENDIX
story.append(PageBreak())
P("Appendix A - How to run", H1)
P("<font face='Courier'>pip install -r requirements.txt</font><br/>"
  "<font face='Courier'>python -m nltk.downloader stopwords wordnet omw-1.4</font><br/>"
  "<font face='Courier'>streamlit run app.py</font><br/>"
  "Optional: <font face='Courier'>python run_experiments.py</font> reproduces every number and figure in this report, and "
  "<font face='Courier'>python report/build_report.py</font> rebuilds this PDF.")
P("Appendix B - Application screenshots", H1)
for name, cap in [("01_dashboard", "Dashboard and corpus overview"),
                  ("02_ingestion", "Data ingestion and corpus statistics"),
                  ("03_preprocessing", "Preprocessing configuration and feature engineering"),
                  ("04_classification", "Text mining hub - Naive Bayes classification"),
                  ("05_clustering", "Text mining hub - K-Means clustering steps and visualisations"),
                  ("06_evaluation", "Text mining hub - evaluation and comparison"),
                  ("07_duplicates", "Text mining hub - duplicate detection"),
                  ("09_recommendation_query", "Recommendation from a free-text query"),
                  ("10_analytics_prep", "Performance analytics - preprocessing comparison")]:
    screenshot(name, "Screenshot - " + cap)

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.8 * cm, title="FinNews Intelligence - IR Text Mining Report",
                        author=", ".join(n for n, _ in AUTHORS))
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
