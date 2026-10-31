"""Configurable text-preprocessing pipeline.

Library use here is limited to linguistic resources (NLTK stop-word list,
Porter stemmer, WordNet lemmatizer). Tokenisation, cleaning and n-gram
generation are implemented directly.
"""
import html
import re
from dataclasses import asdict, dataclass

from nltk.corpus import stopwords as _sw
from nltk.stem import PorterStemmer, WordNetLemmatizer

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
try:
    _STOPWORDS = set(_sw.words("english"))
except LookupError:  # fallback if the NLTK list is missing
    _STOPWORDS = set(
        "a an the and or but if of at by for with about against between into through during "
        "before after above below to from up down in out on off over under again further then "
        "once here there when where why how all any both each few more most other some such no "
        "nor not only own same so than too very s t can will just don should now i me my we our "
        "you your he him his she her it its they them their what which who whom this that these "
        "those am is are was were be been being have has had having do does did doing would could".split()
    )
# Newswire boiler-plate that carries no topical meaning
DOMAIN_STOPWORDS = {"reuter", "reuters", "said", "says", "mln", "pct", "dlrs", "would", "also", "inc", "corp", "co", "ltd"}

_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*|\d+(?:\.\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    strip_markup: bool = True
    remove_numbers: bool = True
    remove_stopwords: bool = True
    domain_stopwords: bool = True
    normalizer: str = "lemma"  # "none" | "stem" | "lemma"
    ngram_max: int = 1  # 1 = unigrams, 2 = uni+bigrams
    min_token_len: int = 2

    def key(self):
        return tuple(sorted(asdict(self).items()))

    def describe(self):
        parts = []
        if self.lowercase:
            parts.append("lowercase")
        if self.strip_markup:
            parts.append("strip markup")
        if self.remove_numbers:
            parts.append("drop numbers")
        if self.remove_stopwords:
            parts.append("stop-words")
        if self.domain_stopwords:
            parts.append("domain stop-words")
        if self.normalizer != "none":
            parts.append({"stem": "Porter stemming", "lemma": "WordNet lemmatization"}[self.normalizer])
        parts.append("unigrams" if self.ngram_max == 1 else f"1-{self.ngram_max}-grams")
        return ", ".join(parts)


RAW_CONFIG = PreprocessConfig(
    lowercase=False, strip_markup=False, remove_numbers=False, remove_stopwords=False,
    domain_stopwords=False, normalizer="none", ngram_max=1, min_token_len=1,
)


def clean(text, cfg):
    text = html.unescape(str(text))
    if cfg.strip_markup:
        text = _TAG_RE.sub(" ", text)
    return text


def tokenize(text):
    """Regex tokenizer: words (with inner hyphen/apostrophe) and numbers."""
    return _TOKEN_RE.findall(text)


def ngrams(tokens, n_max):
    out = list(tokens)
    for n in range(2, n_max + 1):
        out.extend("_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out


def preprocess(text, cfg):
    text = clean(text, cfg)
    tokens = tokenize(text)
    if cfg.lowercase:
        tokens = [t.lower() for t in tokens]
    if cfg.remove_numbers:
        tokens = [t for t in tokens if not t[0].isdigit()]
    stop = set()
    if cfg.remove_stopwords:
        stop |= _STOPWORDS
    if cfg.domain_stopwords:
        stop |= DOMAIN_STOPWORDS
    if stop:
        tokens = [t for t in tokens if t.lower() not in stop]
    if cfg.normalizer == "stem":
        tokens = [_STEMMER.stem(t) for t in tokens]
    elif cfg.normalizer == "lemma":
        tokens = [_LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(t), pos="v") for t in tokens]
    tokens = [t for t in tokens if len(t) >= cfg.min_token_len]
    return ngrams(tokens, cfg.ngram_max)


def preprocess_corpus(texts, cfg):
    return [preprocess(t, cfg) for t in texts]
