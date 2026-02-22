"""Configuration loader for the analysis pipeline.

Loads settings.yaml and resolves all relative paths against the repo root
and workspace root so the pipeline is reproducible on any machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and resolve the pipeline configuration.

    Args:
        config_path: Explicit path to settings.yaml.  When ``None``
            (the default), the loader checks ``HTAG_CONFIG`` env-var
            first, then falls back to ``config/settings.yaml`` relative
            to the repository root.

    Returns:
        Fully-resolved configuration dictionary.  Two extra keys are
        injected under ``_resolved``:

        - ``repo_dir`` – absolute ``Path`` to this repository
        - ``workspace_root`` – absolute ``Path`` one level up (sibling repos)
    """
    repo_dir = Path(__file__).resolve().parent.parent

    if config_path is None:
        config_path = os.environ.get("HTAG_CONFIG")
    if config_path is None:
        config_path = repo_dir / "config" / "settings.yaml"

    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    workspace_root = repo_dir.parent

    cfg["_resolved"] = {
        "repo_dir": repo_dir,
        "workspace_root": workspace_root,
    }

    return cfg


def resolve_repo_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Join ``parts`` relative to the **repo** directory.

    Args:
        cfg: Configuration dict (must contain ``_resolved``).
        *parts: Path segments to join.

    Returns:
        Absolute ``Path``.
    """
    return cfg["_resolved"]["repo_dir"].joinpath(*parts)


def resolve_ws_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Join ``parts`` relative to the **workspace** root.

    Args:
        cfg: Configuration dict (must contain ``_resolved``).
        *parts: Path segments to join.

    Returns:
        Absolute ``Path``.
    """
    return cfg["_resolved"]["workspace_root"].joinpath(*parts)
