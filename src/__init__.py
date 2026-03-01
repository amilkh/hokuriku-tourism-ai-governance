"""Hokuriku Tourism AI Governance Framework.

Production-grade research pipeline for AI-driven demand forecasting
and spatial optimization of Fukui Prefecture tourism.

Modules:
    config:               Configuration loader (settings.yaml)
    report:               Report / metrics writer
    validator:            Data integrity auditing (schema, drift, outliers)
    data_loader:          Camera, weather, RSI, survey data loading
    feature_engineering:  Calendar, weather severity, lags, interactions
    models:               OLS, Random Forest, robustness checks
    kansei:               Discomfort Index, satisfaction, text mining
    spatial:              Cross-prefectural CCF, multi-node analysis
    economics:            Opportunity gap, lost population, ranking simulation
    visualizer:           All figure generation
    latex_export:         LaTeX table generator for paper submission
"""

__version__ = "2.1.0"
__author__ = "Amil Khanzada"
