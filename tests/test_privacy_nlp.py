from __future__ import annotations

import pandas as pd

from src.privacy_nlp import apply_privacy_layer, sanitize_text


def test_sanitize_text_redacts_email_and_phone(monkeypatch):
    monkeypatch.setattr("src.privacy_nlp.get_nlp_model", lambda: None)

    text = "Reach me at user@example.com or 090-1234-5678 before arrival"
    cleaned = sanitize_text(text)

    assert "user@example.com" not in cleaned
    assert "090-1234-5678" not in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_PHONE]" in cleaned


def test_apply_privacy_layer_applies_only_target_columns(monkeypatch):
    monkeypatch.setattr("src.privacy_nlp.get_nlp_model", lambda: None)

    df = pd.DataFrame(
        {
            "reason": ["Email is a@example.com"],
            "freetext": ["Call me at 03-1234-5678"],
            "note": ["keep this raw"],
        }
    )

    out = apply_privacy_layer(df, ["reason", "freetext"])

    assert out.loc[0, "reason"] == "Email is [REDACTED_EMAIL]"
    assert out.loc[0, "freetext"] == "Call me at [REDACTED_PHONE]"
    assert out.loc[0, "note"] == "keep this raw"


def test_sanitize_text_redacts_phone_parentheses_and_spaces(monkeypatch):
    monkeypatch.setattr("src.privacy_nlp.get_nlp_model", lambda: None)

    text = "Numbers: 03(1234)5678 and 090 1234 5678"
    cleaned = sanitize_text(text)

    assert "03(1234)5678" not in cleaned
    assert "090 1234 5678" not in cleaned
    assert cleaned.count("[REDACTED_PHONE]") == 2
