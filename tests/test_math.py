"""Focused math-accuracy tests for core research formulas.

Covers:
- Discomfort Index (kansei)
- Quadratic regression optimum (Eiheiji-style satisfaction curve)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.kansei import compute_discomfort_index


def test_discomfort_index_reference_values() -> None:
    temp = pd.Series([25.0, 30.0, 0.0])
    humidity = pd.Series([60.0, 80.0, 50.0])
    di = compute_discomfort_index(temp, humidity)

    assert abs(di.iloc[0] - 72.82) < 0.01
    assert abs(di.iloc[1] - 82.92) < 0.01
    assert abs(di.iloc[2] - 39.15) < 0.01


def test_quadratic_regression_optimum_density() -> None:
    x = np.array([10, 20, 30, 40, 50, 60, 70], dtype=float)
    y = -0.001 * x**2 + 0.1 * x + 1.5

    a, b, c = np.polyfit(x, y, 2)

    assert a < 0
    peak = -b / (2 * a)
    assert abs(peak - 50.0) < 1e-6

    y_peak = a * peak**2 + b * peak + c
    assert y_peak > y.min()
