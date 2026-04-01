"""Pytest hooks and shared fixtures."""

from __future__ import annotations

import matplotlib

# Non-interactive backend before any test imports src.visualizer (which loads pyplot).
matplotlib.use("Agg")
