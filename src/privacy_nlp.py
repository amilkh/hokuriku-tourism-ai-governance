"""Privacy Layer for Data Sanitization.

Intercepts PII (Personally Identifiable Information) before downstream
processing.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)
_nlp = None


def get_nlp_model():
    """Lazy-load the Japanese NER model if already installed.

    This function intentionally avoids runtime downloads to keep pipeline runs
    deterministic and avoid side effects.
    """
    global _nlp
    if _nlp is None and spacy is not None:
        try:
            _nlp = spacy.load("ja_core_news_sm")
        except OSError:
            logger.warning(
                "spaCy model 'ja_core_news_sm' is not installed. "
                "PERSON redaction will be skipped. Install with: "
                "python -m spacy download ja_core_news_sm"
            )
    return _nlp


def sanitize_text(text: str) -> str:
    """Redact common PII from free text using regex plus optional NER."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return text

    clean_text = text

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    clean_text = re.sub(email_pattern, "[REDACTED_EMAIL]", clean_text)

    # Handles common variants: 090-1234-5678, 03(1234)5678, 090 1234 5678.
    phone_pattern = r"\b0\d{1,4}(?:[-()\s]?\d{1,4}){1,2}\b"
    clean_text = re.sub(phone_pattern, "[REDACTED_PHONE]", clean_text)

    nlp = get_nlp_model()
    if nlp is not None:
        doc = nlp(clean_text)
        for ent in reversed(doc.ents):
            if ent.label_ == "PERSON":
                clean_text = (
                    clean_text[:ent.start_char]
                    + "[REDACTED_PERSON]"
                    + clean_text[ent.end_char:]
                )

    return clean_text


def apply_privacy_layer(df: pd.DataFrame, text_columns: list[str]) -> pd.DataFrame:
    """Apply PII sanitization across selected DataFrame text columns."""
    df_clean = df.copy()
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(sanitize_text)
    return df_clean
