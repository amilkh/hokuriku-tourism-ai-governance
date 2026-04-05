"""Regression tests for benchmark/ablation figure i18n and JA font resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src import visualizer as viz


def _has_japanese_script(s: str) -> bool:
    """True if string contains hiragana, katakana, or a CJK ideograph."""
    for ch in s:
        o = ord(ch)
        if 0x3040 <= o <= 0x309F:  # hiragana
            return True
        if 0x30A0 <= o <= 0x30FF:  # katakana
            return True
        if 0x4E00 <= o <= 0x9FFF:  # CJK unified (common kanji range)
            return True
    return False


def test_bench_i18n_en_ja_distinct_suptitles() -> None:
    en = viz._BENCH_I18N["en"]["suptitle"]
    ja = viz._BENCH_I18N["ja"]["suptitle"]
    assert isinstance(en, str) and isinstance(ja, str)
    assert en != ja
    assert "Chronological" in en
    assert _has_japanese_script(ja)


def test_bench_i18n_ja_metric_ylabels_differ_from_en() -> None:
    en_ylabs = [m[2] for m in viz._BENCH_I18N["en"]["metrics"]]  # type: ignore[index]
    ja_ylabs = [m[2] for m in viz._BENCH_I18N["ja"]["metrics"]]  # type: ignore[index]
    assert en_ylabs != ja_ylabs
    assert any(_has_japanese_script(y) for y in ja_ylabs)


def test_ablation_i18n_en_ja_titles_distinct() -> None:
    en = viz._ABLATION_I18N["en"]["mae_title"]
    ja = viz._ABLATION_I18N["ja"]["mae_title"]
    assert en != ja
    assert _has_japanese_script(ja)


def test_resolve_japanese_benchmark_fontproperties_type() -> None:
    fp = viz._resolve_japanese_benchmark_fontproperties()
    assert fp is None or fp.get_name()


def test_plot_benchmark_comparison_ja_suptitle(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: list[str] = []

    def fake_save(fig, path, reporter, dpi=150, ja_copy=None) -> None:
        t = fig._suptitle
        captured.append(t.get_text() if t is not None else "")
        plt.close(fig)

    monkeypatch.setattr(viz, "_save", fake_save)
    reporter = MagicMock()
    df = pd.DataFrame(
        {"model": ["ols"], "MAE": [1.0], "RMSE": [2.0], "R2": [0.5]},
    )
    viz.plot_benchmark_comparison(
        df,
        str(tmp_path / "fig.png"),
        reporter,
        lang="ja",
        ja_copy=False,
    )
    assert len(captured) == 1
    assert _has_japanese_script(captured[0])


def test_plot_benchmark_comparison_en_suptitle(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: list[str] = []

    def fake_save(fig, path, reporter, dpi=150, ja_copy=None) -> None:
        t = fig._suptitle
        captured.append(t.get_text() if t is not None else "")
        plt.close(fig)

    monkeypatch.setattr(viz, "_save", fake_save)
    reporter = MagicMock()
    df = pd.DataFrame(
        {"model": ["ols"], "MAE": [1.0], "RMSE": [2.0], "R2": [0.5]},
    )
    viz.plot_benchmark_comparison(
        df,
        str(tmp_path / "fig.png"),
        reporter,
        lang="en",
        ja_copy=False,
    )
    assert captured[0] == viz._BENCH_I18N["en"]["suptitle"]


def test_plot_ablation_impact_ja_panel_title(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: list[str] = []

    def fake_save(fig, path, reporter, dpi=150, ja_copy=None) -> None:
        titles = [ax.get_title() for ax in fig.axes]
        captured.extend(titles)
        plt.close(fig)

    monkeypatch.setattr(viz, "_save", fake_save)
    reporter = MagicMock()
    ablation = pd.DataFrame(
        {
            "scenario": ["full", "no_weather", "no_google_intent"],
            "delta_MAE_vs_full": [0.0, 1.0, 2.0],
            "delta_R2_vs_full": [0.0, -0.01, -0.02],
        },
    )
    viz.plot_ablation_impact(
        ablation,
        str(tmp_path / "fig.png"),
        reporter,
        lang="ja",
        ja_copy=False,
    )
    assert any(viz._ABLATION_I18N["ja"]["mae_title"] in t for t in captured)
