"""
Builds data/reuters_finance_news.csv from the Reuters-21578 newswire corpus
(ApteMod split as distributed with NLTK).

Business context: an investment-research desk that receives a stream of
financial newswire stories and needs them tagged by topic, grouped, de-duplicated
and linked to related stories.

The output keeps text and metadata in separate columns:
    doc_id, title, text, category, source, split, word_count, injected_dup_of

Run once:  python data/prepare_dataset.py
(requires: pip install nltk  and  nltk.download('reuters'))
"""
import os
import random
import re

import pandas as pd
from nltk.corpus import reuters

CATEGORIES = ["acq", "crude", "trade", "money-fx", "interest", "ship"]
PER_CATEGORY = 60
MIN_WORDS = 80
N_INJECTED_DUPS = 12
SEED = 42

CATEGORY_NAMES = {
    "acq": "Mergers & Acquisitions",
    "crude": "Crude Oil",
    "trade": "International Trade",
    "money-fx": "Foreign Exchange",
    "interest": "Interest Rates",
    "ship": "Shipping",
}


def split_title(raw):
    lines = raw.strip("\n").split("\n")
    title = lines[0].strip()
    body = "\n".join(lines[1:])
    return title, body


def make_near_duplicate(rng, title, body, variant):
    """Create a realistic near-duplicate: a re-issued / re-formatted wire story."""
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    if variant == 0:
        # re-issued as an UPDATE with the last sentence dropped
        new_title = "UPDATE 1-" + title
        new_body = " ".join(sentences[:-1]) if len(sentences) > 2 else body
    elif variant == 1:
        # syndicated copy wrapped in HTML markup with inconsistent casing
        new_title = title.title()
        new_body = "<p>" + body.upper().replace("\n", " ") + "</p> <br/>&lt;REUTER&gt;"
    else:
        # corrected copy: one sentence repeated, extra punctuation/whitespace noise
        new_title = "CORRECTED - " + title
        k = rng.randrange(len(sentences))
        new_body = "  ".join(sentences[: k + 1] + [sentences[k]] + sentences[k + 1:])
        new_body = new_body.replace(",", " ,,").replace(" said", " said...")
    return new_title, new_body


def main():
    rng = random.Random(SEED)
    rows = []
    for cat in CATEGORIES:
        ids = [f for f in reuters.fileids(cat) if reuters.categories(f) == [cat]]
        ids = [f for f in ids if len(reuters.words(f)) >= MIN_WORDS]
        rng.shuffle(ids)
        for fid in sorted(ids[:PER_CATEGORY], key=lambda x: int(x.split("/")[1])):
            title, body = split_title(reuters.raw(fid))
            split, num = fid.split("/")
            rows.append(
                dict(
                    doc_id=f"RTR-{int(num):05d}",
                    title=title,
                    text=body,
                    category=cat,
                    category_name=CATEGORY_NAMES[cat],
                    source="Reuters-21578 newswire (1987)",
                    split=split,
                    injected_dup_of="",
                )
            )

    # inject near-duplicates (clearly flagged so detection can be evaluated)
    originals = rng.sample(rows, N_INJECTED_DUPS)
    for i, orig in enumerate(originals):
        t, b = make_near_duplicate(rng, orig["title"], orig["text"], i % 3)
        rows.append(
            dict(
                orig,
                doc_id=f"DUP-{i + 1:03d}",
                title=t,
                text=b,
                source="Syndicated re-issue (synthetic near-duplicate)",
                injected_dup_of=orig["doc_id"],
            )
        )

    df = pd.DataFrame(rows)
    df["word_count"] = df["text"].str.split().str.len()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reuters_finance_news.csv")
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} documents -> {out}")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
