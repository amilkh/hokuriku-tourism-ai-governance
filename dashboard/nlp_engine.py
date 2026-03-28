"""
NLP Engine — Hokuriku Tourism AI Governance Dashboard
----------------------------------------------------------------------------
Provides Natural Language Processing capabilities for analysing
Japanese and English tourism survey free-text responses.

Techniques implemented:
  • Tokenisation    — Janome (pure-Python Japanese morphological analyser)
  • Keyword Extract — TF-IDF with domain-aware stop-words
  • Topic Modelling — Latent Dirichlet Allocation (sklearn)
  • Sentiment       — Bilingual lexicon-based polarity scoring
  • N-gram Analysis — Bigram / trigram frequency distributions
  • Word-Cloud Data — Weighted term maps for visualisation

Author  : Dawood Imtiaz
Date    : 2026-03
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)


# -----------------------------------------------------------
#  Japanese Tokeniser Wrapper (Janome with regex fallback)
# -----------------------------------------------------------

class _TokeniserBackend:
    """Thin wrapper: prefer Janome → fall back to regex."""

    def __init__(self):
        self._janome = None
        try:
            from janome.tokenizer import Tokenizer as JanomeTokenizer
            self._janome = JanomeTokenizer()
            logger.info("Japanese tokeniser: Janome loaded")
        except ImportError:
            logger.warning(
                "Janome not installed — falling back to regex tokeniser. "
                "Install with:  pip install janome"
            )

    def tokenise(self, text: str) -> List[str]:
        if self._janome is not None:
            return self._janome_tokenise(text)
        return self._regex_tokenise(text)

    def _janome_tokenise(self, text: str) -> List[str]:
        """Extracting nouns, verbs, adjectives via Janome."""
        tokens = []
        for tok in self._janome.tokenize(text):
            pos = tok.part_of_speech.split(",")[0]
            surface = tok.surface
            if pos in ("名詞", "動詞", "形容詞") and len(surface) > 1:
                tokens.append(surface)
            elif re.match(r"^[a-zA-Z]{2,}$", surface):
                tokens.append(surface.lower())
        return tokens

    @staticmethod
    def _regex_tokenise(text: str) -> List[str]:
        """Baseline regex: CJK runs + Latin words ≥ 2 chars."""
        ja_tokens = re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,}", text)
        en_tokens = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)]
        return ja_tokens + en_tokens


# --------------------------------------
#  Stop-words (domain-aware, bilingual)
# ---------------------------------------

STOP_WORDS_JA = {
    "する", "いる", "なる", "ある", "こと", "もの", "ため", "それ",
    "これ", "あの", "その", "どの", "よう", "ところ", "ほう",
    "思う", "行く", "来る", "見る", "言う", "できる", "ない",
    "いい", "よい", "多い", "少ない", "大きい", "小さい",
}

STOP_WORDS_EN = {
    "the", "be", "to", "of", "and", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "her",
    "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about",
    "who", "get", "which", "go", "me", "when", "make", "can",
    "like", "very", "was", "were", "been", "being", "had", "has",
    "did", "does", "are", "is", "am",
}

ALL_STOP = STOP_WORDS_JA | STOP_WORDS_EN


# -------------------------------------------------
#  Sentiment Lexicon (tourism-focused, bilingual)
# -------------------------------------------------

_POSITIVE_JA = {
    "素晴らしい": 1.0, "最高": 1.0, "美味しい": 0.8, "綺麗": 0.8,
    "楽しい": 0.9, "良い": 0.6, "嬉しい": 0.8, "満足": 0.7,
    "快適": 0.7, "親切": 0.8, "便利": 0.6, "絶景": 1.0,
    "感動": 0.9, "温かい": 0.6, "美しい": 0.8, "豊か": 0.6,
    "歓迎": 0.7, "落ち着く": 0.6, "魅力": 0.7, "おすすめ": 0.7,
}
_NEGATIVE_JA = {
    "悪い": -0.6, "辛い": -0.7, "不便": -0.7, "残念": -0.6,
    "寒い": -0.4, "暑い": -0.3, "混雑": -0.5, "高い": -0.3,
    "少ない": -0.4, "大変": -0.5, "問題": -0.5, "困る": -0.6,
    "不満": -0.7, "心配": -0.4, "危険": -0.6, "退屈": -0.5,
}
_POSITIVE_EN = {
    "wonderful": 1.0, "excellent": 1.0, "amazing": 1.0, "beautiful": 0.8,
    "great": 0.8, "fantastic": 0.9, "incredible": 0.9, "outstanding": 1.0,
    "stunning": 0.9, "delicious": 0.8, "enjoyable": 0.7, "recommended": 0.7,
    "breathtaking": 1.0, "world-class": 1.0, "spiritual": 0.6, "lovely": 0.7,
}
_NEGATIVE_EN = {
    "bad": -0.6, "poor": -0.6, "terrible": -0.9, "difficult": -0.5,
    "disappointing": -0.7, "cold": -0.3, "inconvenient": -0.6,
    "needs": -0.3, "improvement": -0.3, "lacking": -0.5, "limited": -0.4,
}

SENTIMENT_LEXICON = {**_POSITIVE_JA, **_NEGATIVE_JA, **_POSITIVE_EN, **_NEGATIVE_EN}


# -------------------------------------
#  Main NLP Engine Class
# -------------------------------------

class TourismNLPEngine:
    """
    End-to-end NLP engine for Hokuriku tourism survey analysis.

    Usage::

        engine = TourismNLPEngine()
        results = engine.full_analysis(df["free_text"])
    """

    def __init__(self):
        self._tok = _TokeniserBackend()

    # ─── Tokenisation ──────────────
    def tokenise_documents(
        self, texts: List[str], remove_stops: bool = True
    ) -> List[List[str]]:
        """Tokenise a list of documents into filtered token lists."""
        result = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                result.append([])
                continue
            tokens = self._tok.tokenise(text)
            if remove_stops:
                tokens = [t for t in tokens if t not in ALL_STOP]
            result.append(tokens)
        return result

    # ─── TF-IDF Keyword Extraction ──────
    def extract_keywords(
        self, texts: List[str], top_n: int = 25
    ) -> List[Tuple[str, float]]:
        """
        Extract top-N keywords using TF-IDF scoring.
        Returns list of (term, tfidf_score) sorted descending.
        """
        tokenised = self.tokenise_documents(texts)
        joined = [" ".join(toks) for toks in tokenised]
        joined = [t for t in joined if t.strip()]
        if not joined:
            return []

        tfidf = TfidfVectorizer(
            max_features=500,
            token_pattern=r"(?u)\b\w+\b",
        )
        matrix = tfidf.fit_transform(joined)
        scores = np.asarray(matrix.mean(axis=0)).flatten()
        terms = tfidf.get_feature_names_out()

        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # ─── Topic Modelling (LDA) ──────────────
    def topic_modelling(
        self, texts: List[str], n_topics: int = 5, top_words: int = 8
    ) -> Dict:
        """
        Perform LDA topic modelling.

        Returns
        -------
        dict with keys:
            topics    — list of {id, words, weight}
            doc_topic — array (n_docs, n_topics) with topic probabilities
        """
        tokenised = self.tokenise_documents(texts)
        joined = [" ".join(toks) for toks in tokenised]
        joined_clean = [t for t in joined if len(t.strip()) > 0]
        if len(joined_clean) < n_topics:
            return {"topics": [], "doc_topic": np.array([])}

        cv = CountVectorizer(
            max_features=500,
            token_pattern=r"(?u)\b\w+\b",
        )
        dtm = cv.fit_transform(joined_clean)

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method="online",
        )
        doc_topic = lda.fit_transform(dtm)

        terms = cv.get_feature_names_out()
        topics = []
        for idx, component in enumerate(lda.components_):
            top_idx = component.argsort()[-top_words:][::-1]
            words = [terms[i] for i in top_idx]
            weight = float(component[top_idx].sum())
            topics.append({
                "id": idx,
                "label": f"Topic {idx + 1}",
                "words": words,
                "weight": weight,
            })

        return {"topics": topics, "doc_topic": doc_topic}

    # ─── Sentiment Analysis ────────────────
    def sentiment_analysis(self, texts: List[str]) -> pd.DataFrame:
        """
        Compute lexicon-based sentiment polarity for each text.
        Returns DataFrame with columns:
            text, polarity, label, positive_terms, negative_terms
        """
        records = []
        for text in texts:
            if not isinstance(text, str):
                records.append({
                    "text": str(text), "polarity": 0.0,
                    "label": "Neutral",
                    "positive_terms": [], "negative_terms": [],
                })
                continue

            tokens = self._tok.tokenise(text)
            pos_terms, neg_terms = [], []
            score = 0.0

            for token in tokens:
                if token in SENTIMENT_LEXICON:
                    val = SENTIMENT_LEXICON[token]
                    score += val
                    (pos_terms if val > 0 else neg_terms).append(token)

            # Normalise by token count
            polarity = score / max(len(tokens), 1)
            label = (
                "Positive" if polarity > 0.1
                else "Negative" if polarity < -0.1
                else "Neutral"
            )
            records.append({
                "text": text,
                "polarity": round(polarity, 3),
                "label": label,
                "positive_terms": pos_terms,
                "negative_terms": neg_terms,
            })

        return pd.DataFrame(records)

    # ─── N-gram Analysis ───────────────
    def ngram_frequency(
        self, texts: List[str], n: int = 2, top_k: int = 20
    ) -> List[Tuple[str, int]]:
        """Return top-K n-grams by frequency."""
        tokenised = self.tokenise_documents(texts)
        counter: Counter = Counter()
        for toks in tokenised:
            for i in range(len(toks) - n + 1):
                gram = " ".join(toks[i : i + n])
                counter[gram] += 1
        return counter.most_common(top_k)

    # ─── Word Cloud Data ────────────────
    def wordcloud_frequencies(self, texts: List[str]) -> Dict[str, float]:
        """Return {word: frequency} dict suitable for WordCloud()."""
        tokenised = self.tokenise_documents(texts)
        counter: Counter = Counter()
        for toks in tokenised:
            counter.update(toks)
        # Normalise to [0, 1]
        if not counter:
            return {}
        max_count = counter.most_common(1)[0][1]
        return {word: count / max_count for word, count in counter.most_common(150)}

    # ─── Full Analysis Pipeline ──────────────────────
    def full_analysis(self, texts: pd.Series, n_topics: int = 5) -> Dict:
        """
        Run the complete NLP pipeline on a Series of texts.

        Returns dict containing all analysis results.
        """
        text_list = texts.dropna().astype(str).tolist()
        logger.info(f"Running NLP pipeline on {len(text_list)} documents")

        return {
            "keywords": self.extract_keywords(text_list),
            "topics": self.topic_modelling(text_list, n_topics=n_topics),
            "sentiment": self.sentiment_analysis(text_list),
            "bigrams": self.ngram_frequency(text_list, n=2),
            "trigrams": self.ngram_frequency(text_list, n=3),
            "wordcloud": self.wordcloud_frequencies(text_list),
            "doc_count": len(text_list),
        }