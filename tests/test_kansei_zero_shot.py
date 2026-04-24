from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

from src.kansei import run_zero_shot_diagnostics
from src.visualizer import plot_opportunity_gap_drivers


class _DummyReporter:
    def log(self, msg: str) -> None:
        _ = msg

    def optimize_png(self, path: str | Path) -> None:
        _ = path

    def save_fig(self, fig, fname, *, dpi=None, ja_copy=False):
        path = Path(fname)
        fig.savefig(path, dpi=dpi)
        return path


def test_zero_shot_returns_percentage_distribution(monkeypatch):
    # Fake transformers pipeline so tests are deterministic and offline.
    fake_mod = types.ModuleType("transformers")

    def fake_pipeline(task, model):
        assert task == "zero-shot-classification"
        assert model == "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

        def _classifier(texts, candidate_labels, **kwargs):
            assert kwargs["multi_label"] is False
            out = []
            for t in texts:
                if "cold" in t.lower():
                    top = "weather conditions"
                elif "bus" in t.lower():
                    top = "poor transportation"
                else:
                    top = candidate_labels[0]
                remaining = [x for x in candidate_labels if x != top]
                out.append({"labels": [top] + remaining})
            return out

        return _classifier

    fake_mod.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)

    survey = pd.DataFrame(
        {
            "satisfaction": [1, 2, 1, 5],
            "reason": ["Too cold", "Bus was late", "Cold rain", "Great"],
            "inconvenience": ["", "", "", ""],
            "freetext": ["", "", "", ""],
        }
    )

    result = run_zero_shot_diagnostics(
        survey,
        max_samples=2,
        text_max_chars=50,
    )

    assert set(result.keys()) == {
        "weather conditions",
        "poor transportation",
        "language barrier",
        "lack of information",
        "pricing",
    }
    assert abs(sum(result.values()) - 100.0) < 1e-6
    assert result["weather conditions"] == 50.0
    assert result["poor transportation"] == 50.0


def test_plot_opportunity_gap_drivers_uses_input_distribution(tmp_path):
    out_path = tmp_path / "drivers.png"
    reporter = _DummyReporter()

    fig = plot_opportunity_gap_drivers(
        {
            "weather conditions": 62.0,
            "poor transportation": 23.0,
            "language barrier": 10.0,
            "lack of information": 5.0,
        },
        str(out_path),
        reporter,
        dpi=72,
    )

    assert fig is not None
    assert out_path.exists()
    assert out_path.with_name("drivers_ja.png").exists()
