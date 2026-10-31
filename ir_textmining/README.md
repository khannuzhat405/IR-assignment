# FinNews Intelligence — Text Mining & Document Intelligence System

AIMLZG537 Information Retrieval — Text Mining assignment.

A Streamlit application for **financial newswire analysis** (investment-research desk): it preprocesses
Reuters-21578 news stories, builds an inverted index and TF-IDF vectors by hand, classifies stories with a
hand-written Multinomial Naive Bayes, clusters them with a hand-written K-Means, evaluates with hand-written
Purity / Rand Index, detects duplicates, and recommends related stories (content-based Top-K).

> TF-IDF, the inverted index, cosine similarity, Naive Bayes, K-Means, Purity, Rand Index, precision/recall/F1
> and the confusion matrix are implemented from scratch (`textmining/`). No scikit-learn is used.
> NLTK is used only for the stop-word list, Porter stemmer and WordNet lemmatizer; numpy for array arithmetic.

## 1. Install

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

(Python 3.10+.)

## 2. Run the application

```bash
streamlit run app.py
```

Open http://localhost:8501. Use the sidebar to follow the workflow:

| Page | What it shows |
| --- | --- |
| 🏠 Dashboard | Corpus overview, category distribution, headline metrics |
| 📥 Data ingestion | Bundled dataset or upload CSV / JSON / ZIP of .txt; column mapping; corpus statistics; metadata vs text |
| ⚙️ Preprocessing & features | Configurable pipeline, before/after vocabulary and Zipf plots, inverted-index explorer, TF-IDF vectors, cosine similarity |
| 🔬 Text mining hub | Naive Bayes (metrics, confusion matrix, predictions), K-Means (initialisation, iterations, stopping, 2-D scatter, cluster sizes, membership), Purity / Rand Index and comparison, duplicate detection |
| 🔗 Recommendations | Top-K related stories for a selected document or free-text query, scores, metadata and shared terms |
| 📈 Performance analytics | Re-runs the pipeline under different preprocessing, de-duplication and K settings |

## 3. Reproduce the report numbers (optional)

```bash
python run_experiments.py           # -> results/results.json and results/fig_*.png  (~2 min)
python report/build_report.py       # -> report/IR_Text_Mining_Report.pdf
```

Screenshots for the report: start the app, then `playwright install chromium` and
`python tools/screenshots.py report/screenshots` (set `URL` in the script to your port).

## 4. Dataset

`data/reuters_finance_news.csv` — 372 documents built by `data/prepare_dataset.py` from the
Reuters-21578 (ApteMod) corpus distributed with NLTK:

* 6 single-label topics × ~60 stories: `acq`, `crude`, `trade`, `money-fx`, `interest`, `ship`
  (stories with ≥ 80 words, random seed 42)
* columns: `doc_id`, `title`, `text`, `category`, `category_name`, `source`, `split`, `word_count`,
  `injected_dup_of`
* 12 synthetic near-duplicates (`DUP-001`…`DUP-012`, flagged in `injected_dup_of`) imitate re-issued,
  HTML-wrapped and corrected wire copies, so duplicate detection can be measured. The corpus also contains
  real exact duplicates and templated recurring reports.

To rebuild: `python -c "import nltk; nltk.download('reuters')"` then `python data/prepare_dataset.py`.

## 5. Project layout

```
app.py                     Streamlit front end
textmining/
  preprocess.py            tokenisation, cleaning, stop-words, stemming/lemmatization, n-grams
  features.py              inverted index, TF-IDF (ltc), cosine similarity, index-based query scoring
  classify.py              Multinomial Naive Bayes, stratified split, confusion matrix, P/R/F1
  cluster.py               K-Means (k-means++/random, step log), Purity, Rand Index, PCA (display only)
  dedup.py                 exact / near / template duplicate detection
  recommend.py             content-based Top-K, explanations, precision@K
  pipeline.py, experiments.py
run_experiments.py         all report experiments (5 random seeds)
data/                      dataset + builder script
report/                    PDF report, its builder and screenshots
results/                   experiment output (json + figures)
```
