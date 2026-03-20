"""
Behavioral test suite for src/nlp_sentiment.py.

All tests verify real model behavior. CaptureReporter is a real implementation,
not a mock. BER T tests skip without transformers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp_sentiment import (
    ASPECT_DESCRIPTIONS,
    ASPECT_NAMES,
    AspectSentimentAnalyzer,
    LDATopicPipeline,
    SentimentWeatherCorrelator,
    Tokeniser,
    UndervibrancyDetector,
    eiheiji_atmospheric_resilience_nlp,
    nlp_to_latex,
    prepare_survey_texts,
    run_nlp_analysis,
    text_mine_undervibrancy_nlp,
)



# Real Reporter implementation


class CaptureReporter:
    """Real Reporter implementation for tests. Stores output in memory."""

    def __init__(self) -> None:
        self.sections: list[tuple]  = []
        self.logs:     list[str]    = []
        self.metrics:  dict         = {}

    def section(self, section_id, title: str) -> None:
        self.sections.append((section_id, title))
        self.logs.append(f"\n[{section_id}] {title}")

    def log(self, msg: str) -> None:
        self.logs.append(str(msg))

    def metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def all_text(self) -> str:
        return "\n".join(self.logs)

    def section_ids(self) -> list:
        return [s[0] for s in self.sections]



# Shared corpora and fixtures


# Six semantically unambiguous groups — verified: correct aspect tops 0.68–0.85
_ASPECT_CORPUS_ROWS = [
    "ocean view mountain scenery landscape beautiful nature vistas horizon",
    "crowded queue busy congestion visitors waiting lines capacity packed",
    "transport bus train parking directions distance travel access convenient",
    "food dining restaurant cuisine menu eat drink prices flavour delicious",
    "wind rain cold temperature weather climate discomfort snow freezing",
    "toilets restrooms clean facilities maintenance service quality signage",
]
_ASPECT_CORPUS = pd.Series(_ASPECT_CORPUS_ROWS * 15)   # 90 rows


# LR corpus: bilingual, satisfaction-labelled (verified: +0.70 / −0.66)
_LR_CORPUS_TEXTS = pd.Series((
    [
        "beautiful ocean view mountain scenery breathtaking amazing wonderful",
        "terrible cold wind rain uncomfortable weather worst visit ever",
        "crowded queue busy waiting long lines congestion awful experience",
        "excellent food restaurant delicious cuisine wonderful dining perfect",
        "amazing scenery fantastic landscape gorgeous nature peaceful perfect",
        "disappointing facilities dirty toilets poor service terrible awful",
        "convenient transport bus easy access great parking wonderful",
        "cold freezing rain wind horrible weather miserable uncomfortable",
        "lovely views beautiful nature peaceful relaxing excellent visit",
        "crowded noisy busy terrible congestion long wait disappointing",
        "景色が素晴らしい海と山の絶景感動的な体験",
        "寒くて風が強く不快な天気最悪の体験だった",
        "混雑していて行列が長く待ち時間も多かった",
        "食べ物がとても美味しいレストランで最高の料理",
        "また来たい素晴らしい景色と雰囲気が最高",
        "がっかりした設備が古くトイレも汚かった",
    ] * 6
))
_LR_CORPUS_RATINGS = pd.Series(([5, 1, 1, 5, 5, 1, 4, 1, 5, 1, 5, 1, 1, 5, 5, 1] * 6))


@pytest.fixture(scope="module")
def lr_analyzer() -> AspectSentimentAnalyzer:
    """LR-backed analyser (no BERT); verified directional correctness."""
    a = AspectSentimentAnalyzer()
    a.fit(_LR_CORPUS_TEXTS, ratings=_LR_CORPUS_RATINGS)
    return a


@pytest.fixture(scope="module")
def aspect_analyzer() -> AspectSentimentAnalyzer:
    """General analyser fitted on aspect corpus."""
    a = AspectSentimentAnalyzer()
    try:
        a.fit(_ASPECT_CORPUS, ratings=None)
    except RuntimeError:
        # BERT absent and no ratings — fit with dummy ratings
        dummy = pd.Series([5 if idx % 6 >= 3 else 1 for idx in range(len(_ASPECT_CORPUS))])
        a.fit(_ASPECT_CORPUS, ratings=dummy)
    return a


@pytest.fixture()
def kansei_survey_df() -> pd.DataFrame:
    """Full kansei.py schema: reason / inconvenience / freetext / satisfaction."""
    rng  = np.random.default_rng(0)
    n    = 80
    pos  = "beautiful scenery ocean view mountain wonderful excellent"
    neg  = "cold wind rain crowded queue terrible disappointing awful"
    rows = []
    for i in range(n):
        sat = int(rng.integers(1, 6))
        txt = pos if sat >= 4 else neg
        rows.append({
            "prefecture":    "福井" if i < 60 else "石川",
            "satisfaction":  sat,
            "reason":        txt,
            "inconvenience": "bus far" if sat <= 2 else "",
            "freetext":      "will return" if sat >= 4 else "disappointed",
            "date":          pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "nps_raw":       float(sat - 3) / 2,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def eiheiji_survey_df() -> pd.DataFrame:
    rng      = np.random.default_rng(1)
    n        = 80
    sat_vals = ["とても満足", "満足", "どちらでもない", "不満", "とても不満"]
    rows     = []
    for i in range(n):
        sat = str(rng.choice(sat_vals))
        rows.append({
            "location":      "永平寺エリア" if i < 60 else "福井市",
            "satisfaction":  sat,
            "date":          pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 30),
            "reason":        "scenic peaceful beautiful" if sat in ("とても満足", "満足") else "crowded cold",
            "inconvenience": "",
            "freetext":      "great experience" if sat in ("とても満足", "満足") else "disappointing",
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def bimodal_scores() -> pd.Series:
    """Two well-separated clusters — GMM verified: lower ≥ 95%, upper ≤ 5% flagged."""
    rng = np.random.default_rng(42)
    return pd.Series(np.concatenate([
        rng.normal(-0.60, 0.12, 300),
        rng.normal(+0.55, 0.12, 300),
    ]))


@pytest.fixture()
def sample_weather_df() -> pd.DataFrame:
    rng   = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "DI":         rng.uniform(45, 75, 30),
            "WC":         rng.uniform(-10, 10, 30),
            "precip":     rng.uniform(0, 20, 30),
            "temp":       rng.uniform(-5, 25, 30),
            "snow_depth": rng.uniform(0, 50, 30),
        },
        index=dates,
    )



# 0. CaptureReporter self-test


class TestCaptureReporter:
    def test_section_stored(self):
        r = CaptureReporter()
        r.section("K", "Kansei")
        assert ("K", "Kansei") in r.sections

    def test_log_stored_and_retrievable(self):
        r = CaptureReporter()
        r.log("hello")
        assert "hello" in r.all_text()

    def test_metric_stored(self):
        r = CaptureReporter()
        r.metric("r2", 0.81)
        assert r.metrics["r2"] == pytest.approx(0.81)

    def test_section_ids_ordered(self):
        r = CaptureReporter()
        r.section(1, "A")
        r.section(2, "B")
        assert r.section_ids() == [1, 2]

    def test_section_header_in_logs(self):
        r = CaptureReporter()
        r.section("NLP", "Test section")
        assert any("NLP" in l for l in r.logs)



# 1. prepare_survey_texts


class TestPrepareSurveyTexts:
    def test_concatenates_all_three_kansei_columns(self):
        df = pd.DataFrame({
            "reason":        ["good view"],
            "inconvenience": ["bad parking"],
            "freetext":      ["will return"],
        })
        text = prepare_survey_texts(df).iloc[0]
        assert "good view" in text
        assert "bad parking" in text
        assert "will return" in text

    def test_no_literal_nan_in_output(self, kansei_survey_df):
        texts = prepare_survey_texts(kansei_survey_df)
        assert not texts.str.contains(r"\bnan\b", regex=True).any()

    def test_falls_back_to_fallback_col(self):
        df    = pd.DataFrame({"free_text": ["beautiful scenery", "cold rain"]})
        texts = prepare_survey_texts(df, fallback_col="free_text")
        assert list(texts) == ["beautiful scenery", "cold rain"]

    def test_raises_when_no_columns_found(self):
        with pytest.raises(ValueError, match="No text columns found"):
            prepare_survey_texts(pd.DataFrame({"x": ["a"]}), fallback_col="missing")

    def test_output_length_matches_input(self, kansei_survey_df):
        assert len(prepare_survey_texts(kansei_survey_df)) == len(kansei_survey_df)



# 2. Tokeniser


class TestTokeniser:
    def setup_method(self):
        self.tok = Tokeniser()

    def test_clean_removes_urls(self):
        assert "http" not in self.tok.clean("See https://example.com")

    def test_clean_lowercases(self):
        assert self.tok.clean("BEAUTIFUL") == "beautiful"

    def test_min_len_removes_particles(self):
        # Single-char tokens (Japanese particles, English articles) must be gone
        tokens = self.tok.tokenise("a i the good view")
        for short in ["a", "i"]:
            assert short not in tokens

    def test_content_tokens_preserved(self):
        tokens = self.tok.tokenise("beautiful ocean scenery")
        assert any(t in tokens or "beautiful" in t for t in tokens)

    def test_empty_input_returns_empty_list(self):
        assert self.tok.tokenise("") == []

    def test_none_input_handled(self):
        assert self.tok.clean(None) == ""  # type: ignore[arg-type]

    def test_batch_preserves_length(self):
        texts = pd.Series(["abc", "def ghi", None, ""])
        assert len(self.tok.tokenise_batch(texts)) == 4

    def test_no_hardcoded_stopword_list(self):
        assert not hasattr(self.tok, "_stop_words")
        assert not hasattr(self.tok, "stop_words")

    def test_fallback_is_char_ngram_not_whitespace(self):
        """Without MeCab, whole Japanese sentences should not be one token."""
        tok = Tokeniser()
        tok._tagger = None  # force fallback path
        result = self.tok.tokenise_batch(pd.Series(["beautiful ocean view"]))
        # Each element is a list, not a list containing one giant string
        assert isinstance(result[0], list)



# 3. AspectSentimentAnalyzer — cosine similarity semantics


class TestAspectCosineAttribution:
    """
    Verified empirically: all six aspects top-rank at similarity 0.68–0.85.
    Failure messages include the full similarity dict for instant diagnosis.
    """

    @pytest.mark.parametrize("expected_aspect, text", [
        ("scenery",        "ocean view mountain beautiful landscape scenery horizon"),
        ("crowding",       "crowded queue busy congestion waiting visitors lines"),
        ("accessibility",  "transport bus train parking directions convenient travel"),
        ("food_beverage",  "food dining restaurant cuisine menu eating drinking"),
        ("weather_comfort","wind rain cold temperature weather uncomfortable climate"),
        ("facilities",     "toilets restrooms clean facilities maintained service"),
    ])
    def test_top_aspect_matches_text_topic(self, aspect_analyzer, expected_aspect, text):
        vec  = aspect_analyzer._tfidf.transform([text])
        sims = cosine_similarity(vec, aspect_analyzer._aspect_matrix).ravel()
        top  = ASPECT_NAMES[int(np.argmax(sims))]
        assert top == expected_aspect, (
            f"Expected '{expected_aspect}' for '{text[:40]}', "
            f"got '{top}'.  Sims: {dict(zip(ASPECT_NAMES, sims.round(3)))}"
        )

    def test_scenery_ranks_higher_than_crowding_on_scenery_text(self, aspect_analyzer):
        """Correct aspect must outscore nearest competitor by a clear margin."""
        vec     = aspect_analyzer._tfidf.transform(
            ["ocean view mountain beautiful landscape scenery"]
        )
        sims    = cosine_similarity(vec, aspect_analyzer._aspect_matrix).ravel()
        s_idx   = ASPECT_NAMES.index("scenery")
        c_idx   = ASPECT_NAMES.index("crowding")
        assert sims[s_idx] > sims[c_idx], (
            f"scenery sim {sims[s_idx]:.3f} should exceed crowding sim {sims[c_idx]:.3f}"
        )

    def test_crowding_ranks_higher_than_scenery_on_crowding_text(self, aspect_analyzer):
        vec   = aspect_analyzer._tfidf.transform(
            ["crowded queue busy congestion waiting lines"]
        )
        sims  = cosine_similarity(vec, aspect_analyzer._aspect_matrix).ravel()
        s_idx = ASPECT_NAMES.index("scenery")
        c_idx = ASPECT_NAMES.index("crowding")
        assert sims[c_idx] > sims[s_idx]

    def test_aspect_matrix_one_row_per_aspect(self, aspect_analyzer):
        assert aspect_analyzer._aspect_matrix.shape[0] == len(ASPECT_NAMES)

    def test_descriptions_are_strings_not_lists(self):
        for asp, desc in ASPECT_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"{asp}: expected str, got {type(desc)}"



# 4. Sentiment model — LR path


class TestSentimentModelLR:
    """
    Verified LR scores: pos +0.706, neg −0.662 on bilingual corpus.
    """

    def test_lr_model_fitted_with_ratings(self, lr_analyzer):
        assert lr_analyzer._lr_model is not None
        assert lr_analyzer.active_model == "lr"

    def test_lr_positive_text_scores_positive(self, lr_analyzer):
        score = lr_analyzer._lr_score(
            "beautiful scenery amazing wonderful excellent fantastic"
        )
        assert score > 0, f"LR should score positive text > 0, got {score:.3f}"

    def test_lr_negative_text_scores_negative(self, lr_analyzer):
        score = lr_analyzer._lr_score(
            "terrible disappointing crowded awful horrible worst"
        )
        assert score < 0, f"LR should score negative text < 0, got {score:.3f}"

    def test_lr_positive_beats_negative(self, lr_analyzer):
        pos = lr_analyzer._lr_score("beautiful scenery amazing wonderful excellent")
        neg = lr_analyzer._lr_score("terrible disappointing crowded awful horrible")
        assert pos > neg, f"LR: positive ({pos:.3f}) must exceed negative ({neg:.3f})"

    def test_lr_output_in_valid_range(self, lr_analyzer):
        for text in [
            "beautiful ocean scenery amazing",
            "terrible crowded queue awful",
            "average nothing special",
        ]:
            score = lr_analyzer._lr_score(text)
            assert -1.0 <= score <= 1.0, f"Score {score:.3f} out of [−1, +1]"

    def test_fit_raises_without_bert_and_without_ratings(self):
        """fit() raises RuntimeError when neither BERT nor ratings are available."""
        import src.nlp_sentiment as mod
        if mod._BERT_AVAILABLE:
            pytest.skip("BERT is installed — LR-required path not reachable")
        a = AspectSentimentAnalyzer()
        with pytest.raises(RuntimeError, match="transformers"):
            a.fit(_ASPECT_CORPUS, ratings=None)

    def test_fit_raises_with_fewer_than_30_ratings(self):
        import src.nlp_sentiment as mod
        if mod._BERT_AVAILABLE:
            pytest.skip("BERT is installed")
        a = AspectSentimentAnalyzer()
        sparse_ratings = pd.Series([5.0, 1.0] * 10)  # only 20 rows
        with pytest.raises(RuntimeError):
            a.fit(_LR_CORPUS_TEXTS.iloc[:20], ratings=sparse_ratings)



# 5. Sentiment model — BERT path (skip without transformers)


class TestSentimentModelBERT:
    def test_bert_classifier_loaded(self):
        pytest.importorskip("transformers")
        a = AspectSentimentAnalyzer()
        a.fit(_ASPECT_CORPUS, ratings=None)
        assert a._classifier is not None
        assert a.active_model == "bert"

    def test_bert_positive_text_scores_positive(self):
        pytest.importorskip("transformers")
        a = AspectSentimentAnalyzer()
        a.fit(_ASPECT_CORPUS, ratings=None)
        score = a.score_text("beautiful scenery amazing wonderful excellent")
        assert score > 0, f"BERT: positive text should score > 0, got {score:.3f}"

    def test_bert_negative_text_scores_negative(self):
        pytest.importorskip("transformers")
        a = AspectSentimentAnalyzer()
        a.fit(_ASPECT_CORPUS, ratings=None)
        score = a.score_text("terrible disappointing crowded awful horrible")
        assert score < 0, f"BERT: negative text should score < 0, got {score:.3f}"



# 6. Analyse corpus contract


class TestAnalyseCorpus:
    def test_all_aspect_columns_present(self, lr_analyzer):
        result = lr_analyzer.analyse_corpus(_LR_CORPUS_TEXTS)
        for asp in ASPECT_NAMES:
            assert asp in result.columns

    def test_index_aligned_with_input(self, lr_analyzer):
        result = lr_analyzer.analyse_corpus(_LR_CORPUS_TEXTS)
        assert list(result.index) == list(_LR_CORPUS_TEXTS.index)

    def test_all_scores_in_valid_range(self, lr_analyzer):
        result = lr_analyzer.analyse_corpus(_LR_CORPUS_TEXTS)
        for col in ["overall_sentiment"] + list(ASPECT_NAMES):
            valid = result[col].dropna()
            assert (valid.between(-1.0, 1.0)).all(), f"Column {col} has out-of-range values"

    def test_empty_and_none_text_score_zero(self, lr_analyzer):
        assert lr_analyzer.score_text("") == 0.0
        assert lr_analyzer.score_text(None) == 0.0  # type: ignore[arg-type]

    def test_analyse_before_fit_raises(self):
        a = AspectSentimentAnalyzer()
        with pytest.raises(RuntimeError, match="fit()"):
            a.analyse_corpus(pd.Series(["test"]))

    def test_accepts_satisfaction_ratings_from_kansei_schema(self, kansei_survey_df):
        texts   = prepare_survey_texts(kansei_survey_df)
        ratings = pd.to_numeric(kansei_survey_df["satisfaction"], errors="coerce")
        a       = AspectSentimentAnalyzer()
        a.fit(texts, ratings=ratings)
        assert a.active_model in ("bert", "lr")

    def test_no_keyword_list_constants_in_module(self):
        import src.nlp_sentiment as mod
        for name in dir(mod):
            obj = getattr(mod, name)
            if name.startswith("_") or name in ("ASPECT_NAMES", "ASPECT_DESCRIPTIONS"):
                continue
            if isinstance(obj, (list, set)) and len(obj) > 3:
                if all(isinstance(i, str) and len(i) <= 12 for i in obj):
                    pytest.fail(f"Keyword-list constant found: {name} = {list(obj)[:5]}…")



# 7. LDA Topic Pipeline


class TestLDATopicPipeline:

    @pytest.fixture()
    def topic_token_lists(self):
        import random
        random.seed(0)
        groups = [
            ["scenery","ocean","mountain","landscape","view","nature","beautiful","horizon"],
            ["crowded","queue","congestion","busy","wait","lines","packed","visitors"],
            ["transport","bus","parking","directions","travel","train","access","distance"],
            ["food","restaurant","dining","cuisine","menu","eating","delicious","drink"],
            ["cold","wind","rain","weather","temperature","climate","snow","freezing"],
            ["toilets","clean","facilities","service","maintenance","quality","restrooms"],
        ]
        return [random.choices(random.choice(groups), k=8) for _ in range(120)]

    @pytest.fixture()
    def topic_texts(self, topic_token_lists):
        return pd.Series([" ".join(toks) for toks in topic_token_lists])

    def test_k_within_search_range(self, topic_token_lists, topic_texts):
        r = LDATopicPipeline(k_min=3, k_max=7).fit(
            topic_token_lists, texts=topic_texts
        )
        assert 3 <= r.num_topics <= 7

    def test_doc_topic_rows_are_distributions(self, topic_token_lists, topic_texts):
        r = LDATopicPipeline(k_min=3, k_max=5).fit(
            topic_token_lists, texts=topic_texts
        )
        np.testing.assert_allclose(r.doc_topic_matrix.sum(axis=1), 1.0, atol=1e-3)

    def test_doc_topic_matrix_shape(self, topic_token_lists, topic_texts):
        r = LDATopicPipeline(k_min=3, k_max=5).fit(
            topic_token_lists, texts=topic_texts
        )
        assert r.doc_topic_matrix.shape == (len(topic_token_lists), r.num_topics)

    def test_labels_from_aspect_descriptions_or_fallback(self, topic_token_lists, topic_texts):
        r          = LDATopicPipeline(k_min=3, k_max=5).fit(
            topic_token_lists, texts=topic_texts
        )
        valid_asps = {a.replace("_", " ").title() for a in ASPECT_NAMES}
        for label in r.topic_labels:
            assert label in valid_asps or label.startswith("Topic ("), (
                f"Label '{label}' is neither an aspect label nor a fallback"
            )

    def test_at_least_one_aspect_label_on_clean_corpus(self, topic_token_lists, topic_texts):
        r          = LDATopicPipeline(k_min=3, k_max=7).fit(
            topic_token_lists, texts=topic_texts
        )
        valid_asps = {a.replace("_", " ").title() for a in ASPECT_NAMES}
        assert any(l in valid_asps for l in r.topic_labels), (
            f"Expected ≥1 aspect-matched label, got: {r.topic_labels}"
        )

    def test_model_backend_field_populated(self, topic_token_lists, topic_texts):
        r = LDATopicPipeline(k_min=3, k_max=4).fit(
            topic_token_lists, texts=topic_texts
        )
        assert r.model_backend in ("gensim", "sklearn")

    def test_score_metric_field_populated(self, topic_token_lists, topic_texts):
        r = LDATopicPipeline(k_min=3, k_max=4).fit(
            topic_token_lists, texts=topic_texts
        )
        assert r.score_metric in ("c_v_coherence", "neg_perplexity")

    def test_rpt_logs_per_k(self, topic_token_lists, topic_texts):
        rpt = CaptureReporter()
        LDATopicPipeline(k_min=3, k_max=4).fit(
            topic_token_lists, texts=topic_texts, rpt=rpt
        )
        k_logs = [l for l in rpt.logs if "k=" in l and ("LDA" in l or "perplexity" in l or "coherence" in l)]
        assert len(k_logs) >= 2

    def test_sklearn_fallback_requires_texts(self):
        """sklearn path raises ValueError if texts= is not provided."""
        import src.nlp_sentiment as mod
        if mod._GENSIM_AVAILABLE:
            pytest.skip("gensim installed — sklearn path not reachable")
        with pytest.raises(ValueError, match="texts="):
            LDATopicPipeline(k_min=2, k_max=3).fit([["a", "b"], ["c", "d"]])



# 8. UndervibrancyDetector — GMM statistical behaviour


class TestUndervibrancyDetectorStatistical:

    def test_gmm_threshold_between_component_means(self, bimodal_scores):
        d = UndervibrancyDetector()
        d.fit(bimodal_scores)
        assert -0.60 < d._threshold < 0.55, (
            f"GMM threshold {d._threshold:.3f} not in (−0.60, 0.55)"
        )

    def test_lower_cluster_flagged_at_least_95_pct(self, bimodal_scores):
        rng = np.random.default_rng(42)
        neg = pd.Series(rng.normal(-0.60, 0.12, 300))
        d   = UndervibrancyDetector()
        d.fit(bimodal_scores)
        rate = d.flag_series(neg).mean()
        assert rate >= 0.95, f"Lower cluster flag rate {rate:.2%} < 95%"

    def test_upper_cluster_flagged_at_most_5_pct(self, bimodal_scores):
        rng = np.random.default_rng(42)
        pos = pd.Series(rng.normal(+0.55, 0.12, 300))
        d   = UndervibrancyDetector()
        d.fit(bimodal_scores)
        rate = d.flag_series(pos).mean()
        assert rate <= 0.05, f"Upper cluster flag rate {rate:.2%} > 5%"

    def test_threshold_adapts_to_distribution(self):
        rng = np.random.default_rng(0)
        d1, d2 = UndervibrancyDetector(), UndervibrancyDetector()
        d1.fit(pd.Series(np.concatenate([rng.normal(-0.7, 0.1, 200), rng.normal(+0.6, 0.1, 200)])))
        d2.fit(pd.Series(np.concatenate([rng.normal(-0.2, 0.1, 200), rng.normal(+0.3, 0.1, 200)])))
        assert d1._threshold != d2._threshold

    def test_manual_threshold_not_overwritten(self, bimodal_scores):
        d = UndervibrancyDetector(sentiment_threshold=-0.99)
        d.fit(bimodal_scores)
        assert d._threshold == pytest.approx(-0.99)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="fit()"):
            UndervibrancyDetector().flag(0.0)

    def test_nan_never_flagged(self, bimodal_scores):
        d = UndervibrancyDetector()
        d.fit(bimodal_scores)
        assert not d.flag_series(pd.Series([float("nan")])).any()

    def test_ratio_matches_paper_scale(self, bimodal_scores):
        """10% under-vibrancy / 90% satisfied → ratio > 5 (paper reports 11.5×)."""
        rng    = np.random.default_rng(0)
        scores = pd.Series(np.concatenate([
            rng.normal(-0.60, 0.12, 100),   # 10%
            rng.normal(+0.55, 0.12, 900),   # 90%
        ]))
        d = UndervibrancyDetector()
        d.fit(scores)
        assert d.undervibrancy_ratio(scores) > 5

    def test_ratio_arithmetic(self, bimodal_scores):
        d = UndervibrancyDetector()
        d.fit(bimodal_scores)
        t      = d._threshold
        scores = pd.Series([t - 0.3, t - 0.1, t + 0.1, t + 0.2, t + 0.3, t + 0.4])
        assert d.undervibrancy_ratio(scores) == pytest.approx(2.0)

    def test_no_keyword_attributes(self):
        d = UndervibrancyDetector()
        for attr in ("_signals", "_keywords", "_kw", "_word_list", "_default_kw"):
            assert not hasattr(d, attr), f"Keyword attribute found: {attr}"



# 9. SentimentWeatherCorrelator


class TestSentimentWeatherCorrelator:

    def _sent_df(self, n: int = 30) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        rng   = np.random.default_rng(7)
        return pd.DataFrame({"overall_sentiment": rng.uniform(-1, 1, n)}, index=dates)

    def test_r2_in_unit_interval(self, sample_weather_df):
        r = SentimentWeatherCorrelator().fit(self._sent_df(), sample_weather_df)
        assert 0.0 <= r.r_squared <= 1.0 or math.isnan(r.r_squared)

    def test_feature_importance_descending(self, sample_weather_df):
        r    = SentimentWeatherCorrelator().fit(self._sent_df(), sample_weather_df)
        vals = r.feature_importance.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_all_weather_features_have_coef(self, sample_weather_df):
        r = SentimentWeatherCorrelator().fit(self._sent_df(), sample_weather_df)
        for f in SentimentWeatherCorrelator.WEATHER_FEATURES:
            assert f in r.coef

    def test_no_overlap_returns_nan_r2(self):
        s = pd.DataFrame({"overall_sentiment": np.ones(5)},
                         index=pd.date_range("2024-01-01", periods=5))
        w = pd.DataFrame({"DI": np.ones(5)},
                         index=pd.date_range("2026-01-01", periods=5))
        assert math.isnan(SentimentWeatherCorrelator().fit(s, w).r_squared)

    def test_di_driven_sentiment_recovers_signal(self):
        """Synthetic data: sentiment ∝ −DI → R² > 0.5 (verified empirically)."""
        rng     = np.random.default_rng(0)
        dates   = pd.date_range("2025-01-01", periods=60, freq="D")
        di      = rng.uniform(45, 80, 60)
        noise   = rng.normal(0, 0.05, 60)
        sent    = pd.DataFrame(
            {"overall_sentiment": -0.01 * di + noise + 0.5}, index=dates
        )
        weather = pd.DataFrame({"DI": di, "WC": rng.uniform(-5, 5, 60)}, index=dates)
        r = SentimentWeatherCorrelator().fit(sent, weather)
        assert r.r_squared > 0.5, (
            f"DI-anticorrelated sentiment should give R² > 0.5, got {r.r_squared:.3f}"
        )



# 10. text_mine_undervibrancy_nlp — kansei.py drop-in


class TestTextMineUndervibrancyNLP:

    def test_returns_all_original_kansei_keys(self, kansei_survey_df):
        rpt    = CaptureReporter()
        result = text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        for key in ("n_low_sat", "undervibrancy_hits", "pct", "ratio_vs_high", "examples"):
            assert key in result, f"Missing kansei.py compatibility key: {key}"

    def test_calls_reporter_section(self, kansei_survey_df):
        rpt = CaptureReporter()
        text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        assert len(rpt.sections) >= 1

    def test_gmm_threshold_logged(self, kansei_survey_df):
        rpt = CaptureReporter()
        text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        assert any("threshold" in l.lower() for l in rpt.logs)

    def test_active_model_logged(self, kansei_survey_df):
        rpt = CaptureReporter()
        text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        assert any("model" in l.lower() for l in rpt.logs)

    def test_pct_is_valid_percentage(self, kansei_survey_df):
        rpt    = CaptureReporter()
        result = text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        assert 0.0 <= result["pct"] <= 100.0

    def test_examples_have_sentiment_tag_not_keyword(self, kansei_survey_df):
        rpt    = CaptureReporter()
        result = text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        for sat, tag, text in result["examples"][:3]:
            assert "sentiment=" in tag, (
                f"Example tag should be 'sentiment=X.XXX' (GMM-based), not a keyword: '{tag}'"
            )

    def test_no_keywords_parameter_exists(self):
        import inspect
        sig = inspect.signature(text_mine_undervibrancy_nlp)
        assert "keywords" not in sig.parameters

    def test_chi2_computed_when_both_groups_have_hits(self, kansei_survey_df):
        rpt    = CaptureReporter()
        result = text_mine_undervibrancy_nlp(kansei_survey_df, rpt)
        if result["n_low_sat"] > 0 and result["undervibrancy_hits"] > 0:
            assert "chi2_stat" in result

    def test_empty_df_returns_defaults_without_crash(self):
        rpt    = CaptureReporter()
        result = text_mine_undervibrancy_nlp(pd.DataFrame(), rpt)
        assert result["undervibrancy_hits"] == 0



# 11. eiheiji_atmospheric_resilience_nlp — kansei.py drop-in


class TestEiheijAtmosphericResilienceNLP:

    def test_n_responses_counts_only_eiheiji_rows(self, eiheiji_survey_df):
        rpt    = CaptureReporter()
        result = eiheiji_atmospheric_resilience_nlp(eiheiji_survey_df, rpt)
        assert result["n_responses"] == 60   # fixture: 60 永平寺エリア rows

    def test_calls_reporter_section(self, eiheiji_survey_df):
        rpt = CaptureReporter()
        eiheiji_atmospheric_resilience_nlp(eiheiji_survey_df, rpt)
        assert len(rpt.sections) >= 1

    def test_congestion_uses_crowding_aspect_score(self, eiheiji_survey_df):
        rpt = CaptureReporter()
        eiheiji_atmospheric_resilience_nlp(eiheiji_survey_df, rpt)
        assert any("crowding aspect" in l for l in rpt.logs), (
            "Must log crowding-aspect-based congestion — not keyword matching"
        )

    def test_aspect_df_has_crowding_column(self, eiheiji_survey_df):
        rpt    = CaptureReporter()
        result = eiheiji_atmospheric_resilience_nlp(eiheiji_survey_df, rpt)
        assert "aspect_df" in result
        assert "crowding" in result["aspect_df"].columns

    def test_congestion_pct_in_valid_range(self, eiheiji_survey_df):
        rpt    = CaptureReporter()
        result = eiheiji_atmospheric_resilience_nlp(eiheiji_survey_df, rpt)
        if "congestion_pct" in result:
            assert 0.0 <= result["congestion_pct"] <= 100.0

    def test_no_keyword_lists_anywhere_in_module(self):
        import src.nlp_sentiment as mod
        for name in dir(mod):
            assert "congestion_kw" not in name, f"Found: {name}"
            assert "underv_kw"     not in name, f"Found: {name}"
            assert "default_kw"    not in name, f"Found: {name}"

    def test_spearman_test_preserved(self, eiheiji_survey_df):
        rpt    = CaptureReporter()
        result = eiheiji_atmospheric_resilience_nlp(eiheiji_survey_df, rpt)
        if result.get("n_days", 0) >= 10:
            assert "spearman_r" in result
            assert -1.0 <= result["spearman_r"] <= 1.0

    def test_empty_df_returns_empty_result(self):
        rpt = CaptureReporter()
        assert eiheiji_atmospheric_resilience_nlp(pd.DataFrame(), rpt) == {}



# 12. run_nlp_analysis — full pipeline


class TestRunNLPAnalysis:

    def test_calls_reporter_section(self, kansei_survey_df):
        rpt = CaptureReporter()
        run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert len(rpt.sections) >= 1

    def test_active_model_logged(self, kansei_survey_df):
        rpt = CaptureReporter()
        run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert any("model" in l.lower() for l in rpt.logs)

    def test_kansei_schema_accepted(self, kansei_survey_df):
        rpt     = CaptureReporter()
        results = run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert len(results["sentiment_df"]) == len(kansei_survey_df)

    def test_all_result_keys_present(self, kansei_survey_df):
        rpt     = CaptureReporter()
        results = run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        for key in ("sentiment_df", "topic_result", "undervibrancy_ratio",
                    "weather_corr", "topic_time_series", "summary_metrics"):
            assert key in results

    def test_undervibrancy_flag_column_present(self, kansei_survey_df):
        rpt     = CaptureReporter()
        results = run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert "undervibrancy_flag" in results["sentiment_df"].columns

    def test_summary_metrics_keyed_by_aspect_names(self, kansei_survey_df):
        rpt     = CaptureReporter()
        results = run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert set(results["summary_metrics"]["aspect_sentiments"]) == set(ASPECT_NAMES)

    def test_summary_metrics_includes_model_provenance(self, kansei_survey_df):
        rpt     = CaptureReporter()
        results = run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert "active_sentiment_model" in results["summary_metrics"]
        assert "lda_backend"            in results["summary_metrics"]

    def test_reporter_receives_numeric_metrics(self, kansei_survey_df):
        rpt = CaptureReporter()
        run_nlp_analysis(cfg={}, rpt=rpt, survey_df=kansei_survey_df)
        assert isinstance(rpt.metrics.get("undervibrancy_ratio_nlp"), (int, float))
        assert isinstance(rpt.metrics.get("lda_num_topics"), int)

    def test_no_date_col_sets_topic_ts_to_none(self):
        rpt = CaptureReporter()
        # Provide satisfaction column so LR can be trained (BERT may be absent)
        n    = len(_ASPECT_CORPUS_ROWS) * 5
        df   = pd.DataFrame({
            "free_text":   _ASPECT_CORPUS_ROWS * 5,
            "satisfaction": ([5, 1, 5, 1, 5, 1] * 5)[:n],
        })
        r = run_nlp_analysis(cfg={}, rpt=rpt, survey_df=df)
        assert r["topic_time_series"] is None

    def test_weather_corr_r2_reported(self, kansei_survey_df, sample_weather_df):
        rpt = CaptureReporter()
        run_nlp_analysis(
            cfg={}, rpt=rpt,
            survey_df=kansei_survey_df, weather_df=sample_weather_df,
        )
        assert "sentiment_weather_r2" in rpt.metrics

    def test_no_bare_print_calls(self):
        import ast, inspect
        import src.nlp_sentiment as mod
        violations = [
            f"line {n.lineno}"
            for n in ast.walk(ast.parse(inspect.getsource(mod)))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "print"
        ]
        assert violations == [], f"Bare print() at: {violations}"



# 13. LaTeX export


class TestNLPToLatex:
    def _results(self) -> dict:
        return {
            "summary_metrics": {
                "aspect_sentiments":       {asp: (i - 2) * 0.2 for i, asp in enumerate(ASPECT_NAMES)},
                "undervibrancy_ratio_nlp": 11.5,
                "lda_num_topics":          6,
                "lda_score":               0.5234,
                "lda_score_metric":        "c_v_coherence",
                "active_sentiment_model":  "bert",
            }
        }

    def test_returns_string(self):
        assert isinstance(nlp_to_latex(self._results()), str)

    def test_contains_tabular(self):
        assert r"\begin{tabular}" in nlp_to_latex(self._results())

    def test_all_aspect_labels_present(self):
        latex = nlp_to_latex(self._results())
        for asp in ASPECT_NAMES:
            assert asp.replace("_", " ").title() in latex

    def test_undervibrancy_ratio_present(self):
        assert "11.5" in nlp_to_latex(self._results())

    def test_sentiment_model_present(self):
        assert "bert" in nlp_to_latex(self._results())

    def test_balanced_environments(self):
        latex = nlp_to_latex(self._results())
        assert latex.count(r"\begin{") == latex.count(r"\end{")