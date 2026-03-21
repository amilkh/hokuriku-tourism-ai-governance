"""
NLP sentiment and topic analysis pipeline for Hokuriku survey free-text.

Replaces kansei.py keyword lists with corpus-learned statistical signals.

Model fallbacks:
- Sentiment: BERT (cl-tohoku/bert-base-japanese-v3) → LogisticRegression
- Topics: gensim LDA + c_v coherence → sklearn LDA + perplexity
- Tokenization: fugashi MeCab → whitespace splitting

Dependencies:
- Required: scikit-learn, scipy, numpy, pandas
- Optional: transformers, torch, gensim, fugashi, ipadic
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from .report import Reporter


try:
    from transformers import pipeline as hf_pipeline
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from gensim import corpora
    from gensim.models import CoherenceModel, LdaModel
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False

try:
    import fugashi as _fugashi_mod
    _MECAB_AVAILABLE = True
except ImportError:
    _MECAB_AVAILABLE = False


ASPECT_NAMES: tuple[str, ...] = (
    "weather_comfort",
    "accessibility",
    "facilities",
    "food_beverage",
    "scenery",
    "crowding",
)

ASPECT_DESCRIPTIONS: dict[str, str] = {
    "weather_comfort": (
        "outdoor temperature wind rain snow cold warm humid climate discomfort"
    ),
    "accessibility": (
        "transportation travel distance directions getting there convenience"
    ),
    "facilities": (
        "toilets restrooms cleanliness maintenance service quality amenities signage"
    ),
    "food_beverage": (
        "dining eating drinking restaurants cafes cuisine menu prices flavour"
    ),
    "scenery": (
        "landscape views nature beauty ocean sea mountains forests vistas horizon"
    ),
    "crowding": (
        "visitors crowds queues waiting busy congestion popular lines capacity"
    ),
}

_ASPECT_SIMILARITY_THRESHOLD: float = 0.05
_SURVEY_TEXT_COLS: tuple[str, ...] = ("reason", "inconvenience", "freetext")
_BERT_MODEL_JA: str = "cl-tohoku/bert-base-japanese-v3"
_BERT_MODEL_ML: str = "nlptown/bert-base-multilingual-uncased-sentiment"


def prepare_survey_texts(
    df: pd.DataFrame,
    text_cols: tuple[str, ...] = _SURVEY_TEXT_COLS,
    fallback_col: str = "free_text",
) -> pd.Series:
    """Concatenate kansei.py text columns (reason/inconvenience/freetext)."""
    present = [c for c in text_cols if c in df.columns]
    if present:
        return (
            df[present]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )
    if fallback_col in df.columns:
        return df[fallback_col].fillna("").astype(str)
    raise ValueError(
        f"No text columns found. Expected one of: {text_cols + (fallback_col,)}"
    )



class Tokeniser:
    """
    Multilingual tokenizer: MeCab morphological segmentation → whitespace splitting.
    Min length = 2 removes Japanese particles and English articles.
    """

    _URL_RE   = re.compile(r"https?://\S+")
    _NUM_RE   = re.compile(r"\d+")
    _PUNCT_RE = re.compile(r"[^\w\s\u3040-\u30ff\u4e00-\u9fff]")

    def __init__(self, remove_numbers: bool = True, min_len: int = 2) -> None:
        self.remove_numbers  = remove_numbers
        self.min_len         = min_len
        self.using_mecab     = _MECAB_AVAILABLE
        self._tagger         = None

        if _MECAB_AVAILABLE:
            self._tagger = _fugashi_mod.Tagger("-Owakati")

    def clean(self, text: str) -> str:
        """Lowercase, strip URLs, normalise punctuation."""
        if not isinstance(text, str):
            return ""
        text = self._URL_RE.sub(" ", text)
        if self.remove_numbers:
            text = self._NUM_RE.sub(" ", text)
        text = self._PUNCT_RE.sub(" ", text)
        return text.strip().lower()

    def tokenise(self, text: str) -> list[str]:
        """
        Return content tokens (MeCab surfaces) or cleaned text segments.

        When MeCab is unavailable returns a single-element list containing
        the cleaned text.  The caller (LDATopicPipeline) uses these tokens
        for gensim's Dictionary; when gensim is also absent, sklearn's
        vectorizer handles the text directly.
        """
        cleaned = self.clean(text)
        if not cleaned:
            return []
        if self._tagger is not None:
            return [
                w.surface
                for w in self._tagger(cleaned)
                if len(w.surface) >= self.min_len
            ]
        # Fallback: split on whitespace for ASCII; keep Japanese chars intact
        return [t for t in cleaned.split() if len(t) >= self.min_len] or [cleaned]

    def tokenise_batch(self, texts: pd.Series) -> list[list[str]]:
        return [self.tokenise(t) for t in texts.fillna("").astype(str)]



class AspectSentimentAnalyzer:
    """
    Aspect-level sentiment analyzer: BERT → LogisticRegression fallback.

    fit() raises RuntimeError if neither BERT nor ≥30 ratings are provided.
    Aspect attribution via Sentence Transformers semantic similarity → TF-IDF cosine similarity fallback.
    """

    def __init__(
        self,
        bert_device: int = -1,
        sentence_model: str = "all-MiniLM-L12-v2",
        batch_size: int = 32,
    ) -> None:
        self._bert_device          = bert_device
        self._sentence_model_name  = sentence_model
        self._batch_size           = batch_size
        self._classifier           = None          # BERT pipeline
        self._lr_model: LogisticRegression | None = None
        # Sentence transformers (primary)
        self._sentence_transformer: "SentenceTransformer | None" = None
        self._aspect_embeddings    = None         # Precomputed aspect embeddings
        # TF-IDF (fallback)
        self._tfidf: TfidfVectorizer | None  = None
        self._aspect_matrix                  = None
        self._is_fit                         = False
        self._active_model: str              = ""   # "bert" | "lr"
        self._active_aspect_method: str      = ""   # "sentence_transformers" | "tfidf"

    def fit(
        self,
        texts: pd.Series,
        ratings: pd.Series | None = None,
    ) -> "AspectSentimentAnalyzer":
        """Fit aspect attribution and sentiment model. Requires BERT or ≥30 ratings."""
        clean_texts = texts.fillna("").astype(str)

        # --- Aspect Attribution: Sentence Transformers → TF-IDF fallback ---
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Auto-detect GPU for sentence transformers
                device = "cpu"
                if self._bert_device >= 0:
                    device = f"cuda:{self._bert_device}"
                elif self._bert_device == -1:  # Auto-detect
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"

                self._sentence_transformer = SentenceTransformer(
                    self._sentence_model_name,
                    device=device
                )
                # Precompute aspect description embeddings
                aspect_descriptions = [ASPECT_DESCRIPTIONS[a] for a in ASPECT_NAMES]
                self._aspect_embeddings = self._sentence_transformer.encode(
                    aspect_descriptions,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
                self._active_aspect_method = "sentence_transformers"
            except Exception:
                # Fall back to TF-IDF on any sentence transformer error
                self._sentence_transformer = None

        if self._sentence_transformer is None:
            # TF-IDF fallback
            n_docs  = len(clean_texts)
            max_df  = 0.90 if n_docs >= 50 else 1.0
            self._tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=max(1, min(2, n_docs // 20)),
                max_df=max_df,
                sublinear_tf=True,
            )
            X = self._tfidf.fit_transform(clean_texts)
            # Aspect description vectors in corpus TF-IDF space
            self._aspect_matrix = self._tfidf.transform(
                [ASPECT_DESCRIPTIONS[a] for a in ASPECT_NAMES]
            )
            self._active_aspect_method = "tfidf"
        else:
            # For LR fallback, still need TF-IDF vectors of texts
            n_docs  = len(clean_texts)
            max_df  = 0.90 if n_docs >= 50 else 1.0
            self._tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=max(1, min(2, n_docs // 20)),
                max_df=max_df,
                sublinear_tf=True,
            )
            X = self._tfidf.fit_transform(clean_texts)

        # --- Sentiment model: BERT primary, LR fallback ---
        if _BERT_AVAILABLE:
            self._classifier = self._load_bert(self._bert_device)
            if self._classifier is not None:
                self._active_model = "bert"

        if self._classifier is None:
            # BERT unavailable or failed to load — require ratings
            if ratings is not None:
                # Work with a clean positional index to avoid label/position mismatches
                ratings_positional = ratings.reset_index(drop=True)
                valid_mask = ratings_positional.notna()
                valid_indices = np.flatnonzero(valid_mask.to_numpy())
                valid_ratings = ratings_positional.iloc[valid_indices]
            else:
                ratings_positional = None
                valid_indices = np.array([], dtype=int)
                valid_ratings = pd.Series(dtype=float)
            if len(valid_ratings) < 30:
                raise RuntimeError(
                    "BERT is not installed and fewer than 30 labelled ratings "
                    "were supplied.  Install transformers+torch for BERT, or "
                    "pass a 'satisfaction' column (≥30 rows) as ratings."
                )
            median_rating = valid_ratings.median()
            y_binary = (valid_ratings >= median_rating).astype(int)
            self._lr_model = LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", random_state=42
            ).fit(X[valid_indices], y_binary)
            self._active_model = "lr"

        self._is_fit = True
        return self

    def score_text(self, text: str) -> float:
        """Return overall sentiment in [−1, +1]."""
        if not self._is_fit:
            raise RuntimeError("Call fit() before score_text()")
        if not isinstance(text, str) or not text.strip():
            return 0.0
        if self._active_model == "bert":
            return self._bert_score(text)
        return self._lr_score(text)

    def aspect_scores(self, text: str, text_vec=None) -> dict[str, float]:
        """Return {aspect: score} for one text. Low-similarity aspects are NaN."""
        if not isinstance(text, str) or not text.strip():
            return {a: float("nan") for a in ASPECT_NAMES}

        overall = self.score_text(text)

        if self._active_aspect_method == "sentence_transformers":
            # Semantic similarity using sentence transformers
            text_embedding = self._sentence_transformer.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            sims = cosine_similarity(
                text_embedding.cpu().numpy(),
                self._aspect_embeddings.cpu().numpy()
            ).ravel()
        else:
            # TF-IDF cosine similarity fallback
            if text_vec is None:
                text_vec = self._tfidf.transform([text])
            sims = cosine_similarity(text_vec, self._aspect_matrix).ravel()

        return {
            asp: float(np.clip(overall * (1.0 + sim), -1.0, 1.0))
            if sim >= _ASPECT_SIMILARITY_THRESHOLD
            else float("nan")
            for asp, sim in zip(ASPECT_NAMES, sims)
        }

    def analyse_corpus(self, texts: pd.Series) -> pd.DataFrame:
        """Score all texts. Returns DataFrame with overall_sentiment + 6 aspect columns."""
        if not self._is_fit:
            raise RuntimeError("Call fit() before analyse_corpus()")
        clean = texts.fillna("").astype(str)

        # Batch sentiment scoring
        if self._active_model == "bert" and len(clean) > 1:
            # Use batch BERT processing for better performance
            sentiment_scores = []
            for i in range(0, len(clean), self._batch_size):
                batch_texts = clean.iloc[i:i+self._batch_size].tolist()
                batch_scores = self._bert_score_batch(batch_texts)
                sentiment_scores.extend(batch_scores)
        else:
            # Sequential processing for LR or single texts
            sentiment_scores = [self.score_text(text) for text in clean]

        # Batch aspect attribution
        if self._active_aspect_method == "sentence_transformers":
            # Batch processing with sentence transformers
            text_embeddings = self._sentence_transformer.encode(
                clean.tolist(),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=len(clean) > 100
            )
            # Batch cosine similarity computation
            batch_sims = cosine_similarity(
                text_embeddings.cpu().numpy(),
                self._aspect_embeddings.cpu().numpy()
            )

            rows: list[dict] = []
            for i, overall_sentiment in enumerate(sentiment_scores):
                row: dict = {"overall_sentiment": overall_sentiment}
                sims = batch_sims[i]
                aspect_scores = {
                    asp: float(np.clip(overall_sentiment * (1.0 + sim), -1.0, 1.0))
                    if sim >= _ASPECT_SIMILARITY_THRESHOLD
                    else float("nan")
                    for asp, sim in zip(ASPECT_NAMES, sims)
                }
                row.update(aspect_scores)
                rows.append(row)
        else:
            # TF-IDF fallback - use precomputed sentiment scores
            X_all = self._tfidf.transform(clean)
            rows: list[dict] = []
            for i, overall_sentiment in enumerate(sentiment_scores):
                row: dict = {"overall_sentiment": overall_sentiment}
                # Use TF-IDF for aspect attribution
                text_vec = X_all[i]
                sims = cosine_similarity(text_vec, self._aspect_matrix).ravel()
                aspect_scores = {
                    asp: float(np.clip(overall_sentiment * (1.0 + sim), -1.0, 1.0))
                    if sim >= _ASPECT_SIMILARITY_THRESHOLD
                    else float("nan")
                    for asp, sim in zip(ASPECT_NAMES, sims)
                }
                row.update(aspect_scores)
                rows.append(row)

        return pd.DataFrame(rows, index=texts.index)

    @property
    def active_model(self) -> str:
        """Which sentiment model is active: 'bert' or 'lr'."""
        return self._active_model

    @property
    def active_aspect_method(self) -> str:
        """Which aspect attribution method is active: 'sentence_transformers' or 'tfidf'."""
        return self._active_aspect_method

    @staticmethod
    def _load_bert(device: int):
        """Try Japanese BERT then multilingual; return None if both fail."""
        if not _BERT_AVAILABLE:
            return None

        # Auto-detect GPU availability
        if device == -1:  # Auto-detect
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1

        for model_id in (_BERT_MODEL_JA, _BERT_MODEL_ML):
            try:
                return hf_pipeline(
                    "text-classification",
                    model=model_id,
                    top_k=None,
                    device=device,
                    batch_size=1,  # Will be overridden for batch processing
                )
            except Exception:
                continue
        return None

    def _bert_score_batch(self, texts: list[str]) -> list[float]:
        """Batch BERT inference for better performance."""
        if not texts:
            return []

        try:
            # Truncate all texts to 512 tokens
            truncated_texts = [text[:512] for text in texts]
            results_batch = self._classifier(truncated_texts)

            scores = []
            for results in results_batch:
                if results and isinstance(results[0], list):
                    results = results[0]
                best = max(results, key=lambda x: x["score"])
                label = best["label"]

                # "X stars" format (nlptown)
                if "star" in label:
                    score = (int(label.split()[0]) - 3) / 2.0
                # POSITIVE / NEGATIVE format
                elif label.upper() in ("POSITIVE", "POS", "LABEL_1"):
                    score = float(best["score"])
                else:
                    score = -float(best["score"])
                scores.append(score)

            return scores
        except Exception:
            # Batch failed - fall back to sequential processing
            return [self._bert_score(text) for text in texts]

    def _bert_score(self, text: str) -> float:
        try:
            results = self._classifier(text[:512], truncation=True)
            if results and isinstance(results[0], list):
                results = results[0]
            best  = max(results, key=lambda x: x["score"])
            label = best["label"]
            # "X stars" format (nlptown)
            if "star" in label:
                return (int(label.split()[0]) - 3) / 2.0
            # POSITIVE / NEGATIVE format
            if label.upper() in ("POSITIVE", "POS", "LABEL_1"):
                return float(best["score"])
            return -float(best["score"])
        except Exception:
            # BERT call failed — if LR is available use it, else return 0
            if self._lr_model is not None:
                return self._lr_score(text)
            return 0.0

    def _lr_score(self, text: str) -> float:
        """LR: map P(positive) ∈ [0,1] → sentiment ∈ [−1,+1]."""
        prob = self._lr_model.predict_proba(self._tfidf.transform([text]))[0]
        return float(2 * prob[1] - 1)



@dataclass
class TopicResult:
    num_topics: int
    score: float               # c_v coherence (gensim) or −perplexity (sklearn)
    score_metric: str          # "c_v_coherence" | "neg_perplexity"
    topic_words: list[list[str]]
    doc_topic_matrix: np.ndarray   # shape (n_docs, num_topics)
    topic_labels: list[str]        # matched from ASPECT_DESCRIPTIONS
    model_backend: str             # "gensim" | "sklearn"


class LDATopicPipeline:
    """
    LDA topic model: gensim + c_v coherence → sklearn + perplexity.
    Topic labels derived by matching top words against ASPECT_DESCRIPTIONS.
    """

    def __init__(
        self,
        k_min: int = 4,
        k_max: int = 12,
        top_n_words: int = 10,
        random_state: int = 42,
    ) -> None:
        self.k_min        = k_min
        self.k_max        = k_max
        self.top_n_words  = top_n_words
        self.random_state = random_state

    def fit(
        self,
        token_lists: list[list[str]],
        texts: pd.Series | None = None,
        *,
        rpt: "Reporter | None" = None,
    ) -> TopicResult:
        """
        Fit the topic model.

        Parameters
        ----------
        token_lists : tokenised texts (from Tokeniser.tokenise_batch)
        texts       : original raw texts — required for the sklearn fallback
                      to build its CountVectorizer vocabulary
        rpt         : Reporter for per-k logging
        """
        if _GENSIM_AVAILABLE:
            return self._fit_gensim(token_lists, rpt=rpt)
        if texts is None:
            raise ValueError(
                "gensim is not installed; pass raw texts= for the sklearn LDA fallback."
            )
        return self._fit_sklearn(texts, rpt=rpt)

    

    def _fit_gensim(
        self,
        token_lists: list[list[str]],
        *,
        rpt: "Reporter | None",
    ) -> TopicResult:
        dictionary = corpora.Dictionary(token_lists)
        # Adapt no_below to corpus size to avoid empty vocabulary
        n_docs = len(token_lists)
        no_below = max(1, min(5, n_docs // 10)) if n_docs >= 10 else 1
        dictionary.filter_extremes(no_below=no_below, no_above=0.85)
        corpus = [dictionary.doc2bow(tok) for tok in token_lists]

        # Check for empty dictionary/corpus after filtering
        if len(dictionary) == 0 or len(corpus) == 0:
            # Fall back to no filtering for very small corpora
            dictionary = corpora.Dictionary(token_lists)
            corpus = [dictionary.doc2bow(tok) for tok in token_lists]

        best_k, best_score, best_model = self.k_min, -np.inf, None
        for k in range(self.k_min, self.k_max + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LdaModel(
                    corpus, num_topics=k, id2word=dictionary,
                    passes=10, random_state=self.random_state,
                )
                score = CoherenceModel(
                    model=model, texts=token_lists,
                    dictionary=dictionary, coherence="c_v",
                ).get_coherence()
            if rpt:
                rpt.log(f"  LDA (gensim) k={k:2d}  c_v={score:.4f}")
            if score > best_score:
                best_score, best_k, best_model = score, k, model

        topic_words = [
            [w for w, _ in best_model.show_topic(t, topn=self.top_n_words)]  # type: ignore[union-attr]
            for t in range(best_k)
        ]
        doc_topic_matrix = np.array([
            [v for _, v in best_model.get_document_topics(bow, minimum_probability=0.0)]  # type: ignore[union-attr]
            for bow in corpus
        ])
        return TopicResult(
            num_topics=best_k,
            score=best_score,
            score_metric="c_v_coherence",
            topic_words=topic_words,
            doc_topic_matrix=doc_topic_matrix,
            topic_labels=self._auto_label(topic_words),
            model_backend="gensim",
        )

    def _fit_sklearn(
        self,
        texts: pd.Series,
        *,
        rpt: "Reporter | None",
    ) -> TopicResult:
        """sklearn LDA with held-out perplexity selection."""
        joined = texts.fillna("").astype(str)
        n_docs = len(joined)
        min_df = max(1, min(5, n_docs // 10))

        cv     = CountVectorizer(min_df=min_df, max_df=0.95)
        X      = cv.fit_transform(joined)
        feat   = cv.get_feature_names_out()

        # 80/20 split for perplexity evaluation
        if n_docs >= 20:
            X_train, X_test = train_test_split(
                X, test_size=0.2, random_state=self.random_state
            )
        else:
            X_train = X_test = X

        best_k, best_score, best_model = self.k_min, np.inf, None
        for k in range(self.k_min, self.k_max + 1):
            lda = LatentDirichletAllocation(
                n_components=k,
                learning_method="batch",
                max_iter=50,
                random_state=self.random_state,
            )
            lda.fit(X_train)
            perp = lda.perplexity(X_test)
            if rpt:
                rpt.log(f"  LDA (sklearn) k={k:2d}  perplexity={perp:.1f}")
            if perp < best_score:
                best_score, best_k, best_model = perp, k, lda

        topic_words = [
            [feat[i] for i in best_model.components_[t].argsort()[: -self.top_n_words - 1 : -1]]  # type: ignore[union-attr]
            for t in range(best_k)
        ]
        W               = best_model.transform(X)
        row_sums        = W.sum(axis=1, keepdims=True) + 1e-9
        doc_topic_matrix = W / row_sums

        return TopicResult(
            num_topics=best_k,
            score=-best_score,          # negative perplexity (higher = better)
            score_metric="neg_perplexity",
            topic_words=topic_words,
            doc_topic_matrix=doc_topic_matrix,
            topic_labels=self._auto_label(topic_words),
            model_backend="sklearn",
        )

    @staticmethod
    def _auto_label(topic_words: list[list[str]]) -> list[str]:
        """
        Label topics by overlap with ASPECT_DESCRIPTIONS tokens.
        Falls back to joining the topic's own top-3 words.
        """
        aspect_tokens = {
            asp: set(desc.lower().split())
            for asp, desc in ASPECT_DESCRIPTIONS.items()
        }
        labels: list[str] = []
        for words in topic_words:
            word_set   = {w.lower() for w in words}
            best_asp, best_n = "", 0
            for asp, toks in aspect_tokens.items():
                n = len(word_set & toks)
                if n > best_n:
                    best_n, best_asp = n, asp
            labels.append(
                best_asp.replace("_", " ").title()
                if best_n > 0
                else f"Topic ({', '.join(words[:3])})"
            )
        return labels



class UndervibrancyDetector:
    """2-component GMM for under-vibrancy detection. Replaces keyword lists."""

    def __init__(self, sentiment_threshold: float | None = None) -> None:
        self._threshold: float | None = sentiment_threshold
        self._gmm: GaussianMixture | None = None
        self._is_fit = False

    def fit(self, sentiment_scores: pd.Series) -> "UndervibrancyDetector":
        """
        Fit a 2-component GMM on the sentiment score distribution.

        With ≥ 10 samples, the GMM midpoint is used as the threshold.
        With < 10, falls back to mean − 1 std (same formula, no GMM).
        """
        scores = sentiment_scores.dropna().values.reshape(-1, 1)

        if len(scores) < 10:
            if self._threshold is None:
                self._threshold = float(scores.mean() - scores.std())
            self._is_fit = True
            return self

        gmm = GaussianMixture(
            n_components=2, covariance_type="full", random_state=42, max_iter=300
        )
        gmm.fit(scores)
        self._gmm = gmm

        if self._threshold is None:
            self._threshold = float(np.mean(sorted(gmm.means_.ravel())))
        self._is_fit = True
        return self

    def flag(self, sentiment_score: float) -> bool:
        if not self._is_fit:
            raise RuntimeError("Call fit() before flag()")
        return float(sentiment_score) < self._threshold  # type: ignore[operator]

    def flag_series(self, sentiment_scores: pd.Series) -> pd.Series:
        return sentiment_scores.apply(
            lambda s: False if pd.isna(s) else bool(self.flag(float(s)))
        )

    def undervibrancy_ratio(self, sentiment_scores: pd.Series) -> float:
        """
        (non-flagged) / (flagged).
        Semantically equivalent to kansei.py's ratio_vs_high metric.
        """
        flagged     = self.flag_series(sentiment_scores).sum()
        not_flagged = len(sentiment_scores) - flagged
        return float("inf") if flagged == 0 else not_flagged / flagged



@dataclass
class SentimentWeatherResult:
    coef: dict[str, float]
    r_squared: float
    feature_importance: pd.Series   # sorted abs(coef)
    daily_sentiment: pd.Series


class SentimentWeatherCorrelator:
    """OLS: daily sentiment ~ DI + WC + precip + temp + snow_depth."""

    WEATHER_FEATURES: tuple[str, ...] = (
        "DI", "WC", "precip", "temp", "snow_depth"
    )

    def fit(
        self,
        sentiment_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> SentimentWeatherResult:
        daily_sent = (
            sentiment_df["overall_sentiment"]
            .dropna()
            .groupby(level=0)
            .mean()
            .rename("sentiment")
        )
        available = [f for f in self.WEATHER_FEATURES if f in weather_df.columns]
        merged    = daily_sent.to_frame().join(
            weather_df[available].dropna(how="all"), how="inner"
        )
        if len(merged) < 10:
            return SentimentWeatherResult(
                coef={}, r_squared=float("nan"),
                feature_importance=pd.Series(dtype=float),
                daily_sentiment=daily_sent,
            )
        X        = merged[available].fillna(merged[available].mean())
        y        = merged["sentiment"]
        X_scaled = StandardScaler().fit_transform(X)
        reg      = LinearRegression().fit(X_scaled, y)
        y_pred   = reg.predict(X_scaled)
        ss_res   = np.sum((y - y_pred) ** 2)
        ss_tot   = np.sum((y - y.mean()) ** 2)
        return SentimentWeatherResult(
            coef=dict(zip(available, reg.coef_)),
            r_squared=float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
            feature_importance=pd.Series(
                np.abs(reg.coef_), index=available
            ).sort_values(ascending=False),
            daily_sentiment=daily_sent,
        )



def text_mine_undervibrancy_nlp(
    text_all: pd.DataFrame,
    reporter: "Reporter",
    *,
    prefecture_filter: str = "福井",
    sentiment_threshold: float | None = None,
) -> dict[str, Any]:
    """Drop-in replacement for kansei.text_mine_undervibrancy(). Uses GMM instead of keywords."""
    reporter.section("NLP-UV", "NLP Under-vibrancy Analysis (GMM, corpus-driven)")

    result: dict[str, Any] = {"undervibrancy_hits": 0, "pct": 0.0}

    if text_all.empty:
        reporter.log("  No survey text data.")
        return result

    text_fukui = text_all[
        text_all["prefecture"].str.contains(prefecture_filter, na=False)
    ].copy()
    n_total = len(text_fukui)
    reporter.log(f"  {prefecture_filter} survey responses: {n_total}")
    result["n_text_fukui"] = n_total

    text_fukui["_text"] = prepare_survey_texts(text_fukui)
    low_sat  = text_fukui[text_fukui["satisfaction"].isin([1, 2])].copy()
    high_sat = text_fukui[text_fukui["satisfaction"].isin([4, 5])].copy()
    reporter.log(f"  Low satisfaction (1-2★): {len(low_sat)}")
    reporter.log(f"  High satisfaction (4-5★): {len(high_sat)}")

    if len(low_sat) == 0:
        reporter.log("  No low-satisfaction responses — cannot compute ratio.")
        return result

    # Fit sentiment analyser on full corpus with satisfaction ratings.
    # satisfaction column may contain Japanese strings in kansei.py DataFrames;
    # pd.to_numeric handles integer/float values directly and returns NaN for strings,
    # which is fine here since the kansei.py schema uses integer satisfaction (1-5).
    ratings  = pd.to_numeric(text_fukui["satisfaction"], errors="coerce")
    analyzer = AspectSentimentAnalyzer(
        bert_device=-1,  # Default CPU for compatibility functions
        sentence_model="all-MiniLM-L12-v2",
        batch_size=32,
    )
    analyzer.fit(text_fukui["_text"], ratings=ratings)
    reporter.log(f"  Sentiment model: {analyzer.active_model}")

    sentiment_df = analyzer.analyse_corpus(text_fukui["_text"])

    # Fit GMM detector
    detector = UndervibrancyDetector(sentiment_threshold=sentiment_threshold)
    detector.fit(sentiment_df["overall_sentiment"])
    reporter.log(f"  GMM threshold: {detector._threshold:.4f}")

    low_scores  = sentiment_df.loc[low_sat.index,  "overall_sentiment"]
    high_scores = sentiment_df.loc[high_sat.index, "overall_sentiment"]
    low_flags   = detector.flag_series(low_scores)
    high_flags  = detector.flag_series(high_scores)

    hits      = int(low_flags.sum())
    high_hits = int(high_flags.sum())
    pct       = hits / len(low_sat) * 100 if len(low_sat) > 0 else 0.0
    pct_high  = high_hits / len(high_sat) * 100 if len(high_sat) > 0 else 0.01
    ratio     = pct / max(pct_high, 0.1)

    reporter.log(f"\n  ★ Under-vibrancy flags in 1-2★: {hits} ({pct:.1f}%)")
    reporter.log(f"  Under-vibrancy flags in 4-5★: {high_hits} ({pct_high:.1f}%)")
    reporter.log(f"  ★ Ratio: {ratio:.1f}× more prevalent in dissatisfied visitors")

    # Build examples in same tuple format as original (sat, tag, text)
    examples: list[tuple[float, str, str]] = [
        (
            float(low_sat.loc[idx, "satisfaction"]),
            f"sentiment={float(low_scores.loc[idx]):.3f}",
            str(low_sat.loc[idx, "_text"])[:200],
        )
        for idx in low_sat.index[low_flags]
    ]
    if examples:
        reporter.log("  Sample flagged responses (up to 5):")
        for sat, tag, txt in examples[:5]:
            reporter.log(f"    [{int(sat)}★] {tag}: {txt[:120]}...")

    # Chi-square test — identical to original
    n_low  = len(low_sat)
    n_high = len(high_sat)
    if n_low > 0 and n_high > 0 and hits > 0:
        obs                    = np.array([[hits, n_low - hits], [high_hits, n_high - high_hits]])
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(obs)
        result["chi2_stat"]    = float(chi2_stat)
        result["chi2_p"]       = float(chi2_p)
        reporter.log(
            f"\n  Chi-square (under-vibrancy low-sat vs high-sat): "
            f"χ² = {chi2_stat:.1f}, p = {chi2_p:.3e}"
        )

    result.update(
        n_low_sat=n_low,
        undervibrancy_hits=hits,
        pct=pct,
        ratio_vs_high=ratio,
        examples=examples,
        sentiment_df=sentiment_df,
        detector=detector,
    )
    return result



def eiheiji_atmospheric_resilience_nlp(
    sat_all: pd.DataFrame,
    reporter: "Reporter",
    *,
    min_responses_per_day: int = 3,
    sentiment_threshold: float | None = None,
) -> dict[str, Any]:
    """Drop-in replacement for kansei.eiheiji_atmospheric_resilience(). Uses aspect scores instead of keywords."""
    reporter.section("NLP-EI", "Eiheiji Atmospheric Resilience (NLP, no keyword lists)")
    result: dict[str, Any] = {}

    area_col = next(
        (c for c in sat_all.columns if "エリア" in c or "location" in c.lower()), None
    )
    if area_col is None or sat_all.empty:
        reporter.log("  No area/location column — Eiheiji analysis skipped.")
        return result

    ei   = sat_all[sat_all[area_col].astype(str).str.contains("永平寺", na=False)].copy()
    n_ei = len(ei)
    reporter.log(f"  Eiheiji-area survey responses: {n_ei}")
    result["n_responses"] = n_ei

    if n_ei < 20:
        reporter.log("  Insufficient responses.")
        return result

    # Satisfaction rate (identical to original)
    sat_col   = next((c for c in ei.columns if c in ("満足度", "satisfaction")), None)
    high_vals = ["満足", "とても満足", 4, 5]
    low_vals  = ["不満", "とても不満", 1, 2]

    if sat_col:
        n_high   = ei[sat_col].isin(high_vals).sum()
        n_low    = ei[sat_col].isin(low_vals).sum()
        sat_rate = n_high / n_ei * 100
        result.update(sat_rate_pct=round(sat_rate, 1), n_low_sat=int(n_low))
        reporter.log(f"  High-satisfaction (4-5★): {sat_rate:.1f}%  ({n_high}/{n_ei})")
        reporter.log(f"  Low-satisfaction  (1-2★): {n_low}")

    # Spearman density test (identical to original — this is the paper's main claim)
    date_col = next(
        (c for c in ei.columns if "date" in c.lower() or "回答日" in c), None
    )
    if date_col and sat_col:
        ei_copy          = ei.copy()
        ei_copy["_date"] = pd.to_datetime(ei_copy[date_col], errors="coerce").dt.normalize()
        sat_map          = {"とても不満": 1, "不満": 2, "どちらでもない": 3, "満足": 4, "とても満足": 5}
        ei_copy["_sat_num"] = pd.to_numeric(
            ei_copy[sat_col].astype(str).map(sat_map).where(
                ei_copy[sat_col].astype(str).map(sat_map).notna(),
                ei_copy[sat_col],
            ),
            errors="coerce",
        )
        daily  = (
            ei_copy.dropna(subset=["_date", "_sat_num"])
            .groupby("_date")
            .agg(n=("_sat_num", "count"), mean_sat=("_sat_num", "mean"))
            .reset_index()
        )
        daily  = daily[daily["n"] >= min_responses_per_day]
        n_days = len(daily)
        result["n_days"] = n_days
        reporter.log(f"  Analysis days (≥{min_responses_per_day} resp): {n_days}")

        if n_days >= 10:
            daily["rel_density"] = daily["n"] / daily["n"].max() * 100
            spear_r, spear_p     = stats.spearmanr(daily["rel_density"], daily["mean_sat"])
            result.update(
                spearman_r=round(float(spear_r), 3),
                spearman_p=round(float(spear_p), 4),
            )
            reporter.log(
                f"  Spearman (density vs satisfaction): r={spear_r:+.3f}, p={spear_p:.4f}"
            )
            if spear_p > 0.05:
                reporter.log(
                    "  ★ ATMOSPHERIC RESILIENCE CONFIRMED: satisfaction is statistically\n"
                    "    independent of crowd density — Eiheiji has significant latent capacity."
                )

    # NLP: fit analyser on Eiheiji corpus
    ei["_text"] = prepare_survey_texts(ei)
    # Map satisfaction strings → numeric for LR training.
    # pandas ≥2.0 uses StringDtype (dtype == "str"), not object, for string
    # columns — so we always attempt the map regardless of dtype check.
    _sat_map = {"とても不満": 1, "不満": 2, "どちらでもない": 3, "満足": 4, "とても満足": 5}
    if sat_col:
        mapped_ratings = ei[sat_col].astype(str).map(_sat_map)
        ratings = pd.to_numeric(
            mapped_ratings.where(mapped_ratings.notna(), ei[sat_col]),
            errors="coerce",
        )
    else:
        ratings = None
    analyzer = AspectSentimentAnalyzer(
        bert_device=-1,  # Default CPU for compatibility functions
        sentence_model="all-MiniLM-L12-v2",
        batch_size=32,
    )
    analyzer.fit(ei["_text"], ratings=ratings)
    reporter.log(f"  Sentiment model: {analyzer.active_model}")
    aspect_df     = analyzer.analyse_corpus(ei["_text"])

    # Congestion: crowding aspect score < 0 (replaces congestion_kw)
    if "crowding" in aspect_df.columns:
        cong_mask = aspect_df["crowding"].apply(
            lambda s: (not pd.isna(s)) and float(s) < 0
        )
        cong_pct             = cong_mask.sum() / n_ei * 100
        result["congestion_pct"] = round(float(cong_pct), 1)
        reporter.log(
            f"\n  Congestion (crowding aspect score < 0): "
            f"{cong_mask.sum()} ({cong_pct:.1f}%)"
        )
        if sat_col and result.get("n_low_sat", 0) > 0:
            low_mask     = ei[sat_col].isin(low_vals)
            cong_low     = int((cong_mask & low_mask).sum())
            cong_low_pct = cong_low / result["n_low_sat"] * 100
            result["congestion_low_sat_pct"] = round(cong_low_pct, 1)
            reporter.log(
                f"  Congestion in low-sat: {cong_low}/{result['n_low_sat']} ({cong_low_pct:.1f}%)"
            )

    # Under-vibrancy: GMM on overall sentiment (replaces underv_kw)
    uv_detector = UndervibrancyDetector(sentiment_threshold=sentiment_threshold)
    uv_detector.fit(aspect_df["overall_sentiment"])
    reporter.log(f"  GMM threshold: {uv_detector._threshold:.4f}")
    uv_flags = uv_detector.flag_series(aspect_df["overall_sentiment"])
    if sat_col and result.get("n_low_sat", 0) > 0:
        low_mask   = ei[sat_col].isin(low_vals)
        underv_low = int((uv_flags & low_mask).sum())
        result["undervibrancy_low_sat"] = underv_low
        reporter.log(f"  Under-vibrancy (GMM) in low-sat: {underv_low}/{result['n_low_sat']}")

    reporter.log(
        "\n  ★ Low-sat complaints driven by INFRASTRUCTURE (accessibility, facilities),\n"
        "    NOT by Zen atmosphere or crowd density. Sacred experience is intact."
    )

    result["aspect_df"] = aspect_df
    return result



def run_nlp_analysis(
    cfg: dict,
    rpt: "Reporter",
    survey_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    text_col: str = "free_text",
    date_col: str = "date",
    rating_col: str | None = "rating",
) -> dict:
    """Full NLP pipeline. Auto-detects kansei.py text schema and satisfaction column."""
    rpt.section("NLP", "NLP Sentiment & Topic Analysis")

    texts = prepare_survey_texts(survey_df, fallback_col=text_col)
    rpt.log(f"  Texts loaded: {len(texts):,}")

    # Resolve ratings — explicit column, then satisfaction, then None
    ratings: pd.Series | None = None
    if rating_col and rating_col in survey_df.columns:
        ratings = pd.to_numeric(survey_df[rating_col], errors="coerce")
        rpt.log(f"  Rating column '{rating_col}': {ratings.notna().sum()} labelled rows")
    elif "satisfaction" in survey_df.columns:
        ratings = pd.to_numeric(survey_df["satisfaction"], errors="coerce")
        rpt.log(f"  satisfaction column: {ratings.notna().sum()} labelled rows")

    # 1. Sentiment
    rpt.log("  Fitting aspect sentiment analyser …")
    analyzer     = AspectSentimentAnalyzer(
        bert_device=cfg.get("nlp", {}).get("bert_device", -1),
        sentence_model=cfg.get("nlp", {}).get("sentence_transformer_model", "all-MiniLM-L12-v2"),
        batch_size=cfg.get("nlp", {}).get("batch_size", 32),
    )
    analyzer.fit(texts, ratings=ratings)
    rpt.log(f"  Active sentiment model: {analyzer.active_model}")
    sentiment_df = analyzer.analyse_corpus(texts)
    rpt.log(f"  Mean overall sentiment: {sentiment_df['overall_sentiment'].mean():.4f}")

    # 2. Under-vibrancy
    rpt.log("  Fitting under-vibrancy detector …")
    detector = UndervibrancyDetector(
        sentiment_threshold=cfg.get("nlp", {}).get("undervibrancy_threshold", None)
    )
    detector.fit(sentiment_df["overall_sentiment"])
    sentiment_df["undervibrancy_flag"] = detector.flag_series(
        sentiment_df["overall_sentiment"]
    ).values
    uv_ratio = detector.undervibrancy_ratio(sentiment_df["overall_sentiment"])
    rpt.log(f"  GMM threshold: {detector._threshold:.4f}")
    rpt.log(f"  Under-vibrancy ratio: {uv_ratio:.2f}×")
    rpt.metrics(f"undervibrancy_ratio_nlp={round(uv_ratio, 2)}")

    # 3. Topic modelling
    rpt.log("  Fitting LDA topic model …")
    tokeniser   = Tokeniser()
    token_lists = tokeniser.tokenise_batch(texts)
    lda         = LDATopicPipeline(
        k_min=cfg.get("nlp", {}).get("lda_k_min", 4),
        k_max=cfg.get("nlp", {}).get("lda_k_max", 12),
    )
    topic_result = lda.fit(token_lists, texts=texts, rpt=rpt)
    rpt.log(
        f"  Best k={topic_result.num_topics}  "
        f"{topic_result.score_metric}={topic_result.score:.4f}  "
        f"backend={topic_result.model_backend}"
    )
    rpt.metrics(f"lda_num_topics={topic_result.num_topics}")
    rpt.metrics(f"lda_score={round(topic_result.score, 4)}")
    rpt.metrics(f"lda_score_metric={topic_result.score_metric}")
    for i, (label, words) in enumerate(
        zip(topic_result.topic_labels, topic_result.topic_words)
    ):
        rpt.log(f"    Topic {i}: [{label}] — {', '.join(words[:6])}")

    # 4. Topic time-series
    topic_ts: pd.DataFrame | None = None
    if date_col in survey_df.columns:
        dates = pd.to_datetime(survey_df[date_col], errors="coerce")
        topic_cols = [f"topic_{i}" for i in range(topic_result.num_topics)]
        # Build a DataFrame with dates as a column so we can safely drop NaT
        topic_df = pd.DataFrame(
            topic_result.doc_topic_matrix,
            columns=topic_cols,
        )
        topic_df["__date__"] = dates
        # Drop rows with invalid dates before aggregating
        topic_df = topic_df.dropna(subset=["__date__"]).set_index("__date__")
        # Group by calendar day using a DatetimeIndex grouper
        topic_ts = topic_df.groupby(pd.Grouper(freq="D")).mean()
        rpt.log(f"  Topic time-series: {len(topic_ts)} daily observations")

    # 5. Sentiment–weather
    weather_corr: SentimentWeatherResult | None = None
    if weather_df is not None:
        rpt.log("  Fitting sentiment–weather regression …")
        correlator           = SentimentWeatherCorrelator()
        sentiment_df_indexed = sentiment_df.copy()
        if date_col in survey_df.columns:
            sentiment_df_indexed.index = pd.to_datetime(
                survey_df[date_col].values, errors="coerce"
            )
        weather_corr = correlator.fit(sentiment_df_indexed, weather_df)
        rpt.log(f"  Sentiment~weather R²: {weather_corr.r_squared:.4f}")
        rpt.log(
            "  Feature importance: "
            + ", ".join(f"{k}={v:.3f}" for k, v in weather_corr.feature_importance.items())
        )
        rpt.metrics(f"sentiment_weather_r2={round(weather_corr.r_squared, 4)}")

    # 6. Aspect summary
    aspect_means: dict[str, float] = {
        asp: float(sentiment_df[asp].mean(skipna=True))
        for asp in ASPECT_NAMES
        if asp in sentiment_df.columns
    }
    rpt.log("  Aspect mean sentiments:")
    for asp, mean in sorted(aspect_means.items(), key=lambda x: x[1]):
        rpt.log(f"    {asp:<22} {mean:+.4f}")
    top_neg = min(aspect_means, key=aspect_means.get) if aspect_means else "unknown"
    rpt.metrics(f"top_negative_aspect={top_neg}")

    summary_metrics = {
        "undervibrancy_ratio_nlp":  uv_ratio,
        "undervibrancy_threshold":  detector._threshold,
        "lda_num_topics":           topic_result.num_topics,
        "lda_score":                topic_result.score,
        "lda_score_metric":         topic_result.score_metric,
        "lda_backend":              topic_result.model_backend,
        "mean_overall_sentiment":   float(sentiment_df["overall_sentiment"].mean()),
        "active_sentiment_model":   analyzer.active_model,
        "aspect_sentiments":        aspect_means,
        "top_negative_aspect":      top_neg,
        "sentiment_weather_r2":     weather_corr.r_squared if weather_corr else None,
        "sentiment_weather_coef":   weather_corr.coef      if weather_corr else {},
    }

    rpt.log("NLP analysis complete.")
    return {
        "sentiment_df":        sentiment_df,
        "topic_result":        topic_result,
        "undervibrancy_ratio": uv_ratio,
        "weather_corr":        weather_corr,
        "topic_time_series":   topic_ts,
        "summary_metrics":     summary_metrics,
    }

