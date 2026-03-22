"""Tests for Reporter.save_fig ja_copy semantics."""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.report import Reporter


def _cfg(tmp_path, ja_copy: bool) -> dict:
    return {
        "_resolved": {"repo_dir": tmp_path, "workspace_root": tmp_path},
        "paths": {"output": "out", "figures": "figs"},
        "visualization": {"dpi": 72, "ja_copy": ja_copy},
    }


def test_save_fig_false_never_duplicates_even_when_config_true(tmp_path) -> None:
    r = Reporter(_cfg(tmp_path, ja_copy=True))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    primary = r.fig_dir / "fig.png"
    r.save_fig(fig, "fig.png", ja_copy=False)
    ja = r.fig_dir / "fig_ja.png"
    assert primary.is_file()
    assert not ja.exists()


def test_save_fig_default_uses_config_for_duplicate(tmp_path) -> None:
    r = Reporter(_cfg(tmp_path, ja_copy=True))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    r.save_fig(fig, "fig.png")
    assert (r.fig_dir / "fig.png").is_file()
    assert (r.fig_dir / "fig_ja.png").is_file()


def test_save_fig_default_no_duplicate_when_config_false(tmp_path) -> None:
    r = Reporter(_cfg(tmp_path, ja_copy=False))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    r.save_fig(fig, "fig.png")
    assert (r.fig_dir / "fig.png").is_file()
    assert not (r.fig_dir / "fig_ja.png").exists()
