<div align="center">

# Hokuriku Tourism AI Governance Framework

### AI主導の来訪需要予測と空間的「過少賑わい」分析

**Amil Khanzada** — *特命助教、地域創生推進本部、福井大学*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21271429.svg)](https://doi.org/10.5281/zenodo.21271429)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Data Validated](https://img.shields.io/badge/rows%20audited-1.4M-brightgreen.svg)](src/validator.py)

> **エグゼクティブレポート：**
> [English](EXECUTIVE_REPORT.en.md) ·
> [日本語](EXECUTIVE_REPORT.ja.md)
>
> **Read in:** [English](README.md)

</div>

---

## 概要

本リポジトリは **Distributed Human Data Engine (DHDE)** を実装します。AIカメラ人流データ、気象庁（JMA）観測値、経路検索インプレッション（RSI）の検索意図、および97,719件の北陸観光アンケート回答を統合した、観光分析・政策提言パイプラインです。

このシステムは福井県の構造的観光赤字を定量化します。全国47都道府県中**最下位（47位）**の来訪者数となる背景には、需要（検索意図）は存在するにもかかわらず悪天候により来訪が阻まれる「計画摩擦」があり、これが年間**約119億6,000万円**の機会損失を生んでいます。

**キーワード：** 観光需要予測 · 感性工学 · 不快指数 · 空間飽和度 · 過少賑わい · 北陸地域ガバナンス

---

## 1. 理論的枠組み：Distributed Human Data Engine（DHDE）

DHDEは4種類のデータストリームを単一の分析パイプラインに統合します：

```
┌─────────────────────────────────────────────────────────────────┐
│                  DISTRIBUTED HUMAN DATA ENGINE                  │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ AIカメラ   │  │ JMA気象    │  │ RSI (route)  │  │ 北陸観光 │ │
│  │ 人流データ │  │ 8変数/時間 │  │ 経路検索意図 │  │ アンケート│ │
│  │ (エッジAI) │  │ (hourly)   │  │  (daily)     │  │ (97,719) │ │
│  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘  └────┬─────┘ │
│        └───────────┬───┴────────────────┴───────┬───────┘       │
│                    │     特徴量エンジニアリング    │               │
│                    │   カレンダー · 気象強度指数  │               │
│                    │   ラグ特徴 · 交互作用項      │               │
│                    └──────────────┬───────────────┘               │
│              ┌───────────────────┴───────────────────┐           │
│              │  OLS回帰 + ランダムフォレスト（RF）    │           │
│              │  頑健性：DW, NW-HAC, FD, LDV, VIF    │           │
│              └───────────────────┬───────────────────┘           │
│         ┌────────────────────────┼────────────────────────┐      │
│  ┌──────▼──────┐  ┌─────────────▼───────────┐  ┌─────────▼───┐  │
│  │ 機会損失    │  │ 感性評価                 │  │ 空間飽和度  │  │
│  │ 推計        │  │ DI · 体感温度 · テキスト │  │ マルチノード│  │
│  └──────────────┘  └─────────────────────────┘  └─────────────┘  │
│                   ──► analysis_metrics.txt                       │
│                   ──► LaTeX tables for paper                     │
└─────────────────────────────────────────────────────────────────┘
```

**空間ネットワークのノード：**

| ノード | 場所 | カメラソース | 気象ステーション |
|--------|------|------------|----------------|
| A | 東尋坊／三国 | tojinbo-shotaro | 三国（JMA） |
| B | 福井駅東口 | fukui-station-east | 福井（JMA） |
| C | 勝山（恐竜博物館） | katsuyama | 勝山（JMA） |
| D | レインボーライン | rainbow-line-parking-lot-1-gate | 福井（代理） |

---

## 2. 主要結果

| 指標 | 値 | 解釈 |
|------|----|----|
| **OLS R²** | 0.810（調整済み R² = 0.802） | ベースライン説明力 |
| **RF 5分割CV R²** | 0.557 ± 0.131 | サンプル外予測精度 |
| **一階差分モデル R²** | 0.708 | 自己相関補正後 |
| **LDV R² / DW** | 0.848 / 1.899 | 動的モデル、残差クリーン |
| **最大予測因子** | `directions` (RSI) | 経路検索意図、β = +0.456 |
| **石川→東尋坊ラグ** | r = +0.549 | 県間需要スピルオーバー |
| **来訪者数と満足度** | r = +0.150（p = 0.002） | **オーバーツーリズムなし** |
| **機会損失来訪者数** | 85,522人 | 年間Opportunity Gap |
| **冬季気象感度** | 夏季の6.26倍 | 季節的非対称性 |
| **過少賑わい比率** | 11.5倍 | 低評価レビューの出現頻度 |
| **全国順位（冬季）** | 47位／47都道府県 | 福井の構造的赤字 |

---

## 3. 約119億円の機会損失（多拠点集計）

**機会損失**は、経路検索意図に基づく予測来訪者数と、悪天候日における実際の来訪者数との差として定義されます：

$$
\text{Lost Visitors}_d = \hat{y}_d^{\text{OLS}} - y_d^{\text{actual}} \quad \text{when} \quad y_d < \hat{y}_d
$$

$$
\text{Total Economic Loss} = \sum_{d \in \mathcal{G}} \text{Lost Visitors}_d \times \bar{S}
$$

ここで $\bar{S} = ¥13{,}811$ は来訪者1人あたり平均消費額（福井観光アンケート、n = 97,719）、$\mathcal{G}$ はギャップ発生日の集合です。

| 構成要素 | 値 |
|---------|-----|
| ギャップ発生日数 | 42日 |
| 機会損失来訪者数合計 | 85,522人 |
| 来訪者1人あたり平均消費額 | ¥13,811 |
| **年間収益損失合計** | **約¥11.96B** |

---

## 4. 感性環境評価

### 4.1 不快指数（Discomfort Index）

$$
DI = 0.81 \cdot T + 0.01 \cdot H \cdot (0.99 \cdot T - 14.3) + 46.3
$$

T は気温（℃）、H は相対湿度（%）。

### 4.2 体感温度（Wind Chill）

$$
WC = 13.12 + 0.6215T - 11.37V^{0.16} + 0.3965TV^{0.16}
$$

V は風速（km/h）。T ≤ 10℃、V > 4.8 km/h の条件下で有効。

### 4.3 オーバーツーリズム閾値

日次来訪者数と平均満足度のスピアマン相関：

$$
r_s(\text{来訪者数}, \text{満足度}) = +0.150 \quad (p = 0.002)
$$

**正の相関**は、福井の課題がオーバーツーリズムではなく「**過少賑わい**」であることを示しています。

---

## 5. 空間飽和マップ

4ノードによる分析により、福井県の**地理的飽和**を達成しています：

```
         ┌──── Node C: 勝山（山間部・東） ────┐
         │                                    │
Node A: 東尋坊 ─── Node B: 福井駅 ─── Node D: レインボーライン
（沿岸部・北）    （都市部・中央）      （景勝地・南）
```

各ノードは独立した局所気象データでモデル化され、以下を実現します：
- **天気シールドネットワーク**：三国（沿岸）が荒天時、勝山（内陸）は晴れの可能性
- **リアルタイム気象誘導**による需要再配分

---

## 6. モデル頑健性

| 診断 | 統計量 | 解釈 |
|------|--------|------|
| Durbin–Watson（OLS） | 1.005 | 軽度の自己相関 → 下記で補正 |
| Durbin–Watson（一階差分） | 2.525 | **残差クリーン** |
| Newey–West HAC | 8変数が有意 | 不均一分散に対して頑健 |
| 一階差分 R² | 0.708 | トレンドをコントロール |
| LDV R² | 0.848 | 動的モデル仕様 |
| VIF（最大） | 15.0（カレンダー交互作用項） | 多重共線性なし |
| 気象データの寄与 | +0.056 R² | JMAデータの貢献を定量化 |

---

## 7. リポジトリ構成

```
hokuriku-tourism-ai-governance/
├── pyproject.toml                # PEP 517/621 パッケージ定義 → pip install .
├── requirements.txt              # ランタイム依存関係
├── config/
│   └── settings.yaml             # パイプライン設定（パス・パラメータ一元管理）
├── src/
│   ├── __init__.py               # パッケージメタデータ
│   ├── config.py                 # YAML設定ローダー & パス解決
│   ├── data_loader.py            # カメラ・JMA・RSI・アンケートローダー
│   ├── feature_engineering.py    # カレンダー・気象強度・ラグ・交互作用項
│   ├── models.py                 # OLS + ランダムフォレスト + 頑健性スイート
│   ├── kansei.py                 # 不快指数・体感温度・テキストマイニング
│   ├── economics.py              # 機会損失・来訪者数・順位推計
│   ├── spatial.py                # 県間CCF・マルチノードガバナンス
│   ├── validator.py              # データ完全性監査（スキーマ・ドリフト・外れ値）
│   ├── visualizer.py             # 全図表生成（12種以上、EN & JA）
│   ├── latex_export.py           # 論文用LaTeXテーブル生成
│   ├── report.py                 # 集中ロギング & メトリクス Reporter
│   └── run_analysis.py           # パイプラインのメインエントリポイント
├── tests/
│   ├── test_models.py            # OLS・RF・頑健性テスト
│   ├── test_kansei.py            # DI・体感温度の手計算検証
│   ├── test_validator.py         # スキーマ・外れ値・ドリフト検知テスト
│   ├── test_features.py          # 特徴量エンジニアリングパイプラインテスト
│   └── test_math.py              # 統計関数の数値正確性チェック
├── jma/                          # 気象庁観測データ（コミット済み）
│   ├── fetch_jma_monthly.py      # JMA時間別CSVスクレイパー
│   ├── merge_clean_jma.py        # ステーション別CSVへのマージ
│   └── jma_*.csv                 # マージ済み8フィールドデータセット
├── output/                       # 生成物（参照用にコミット済み）
├── EXECUTIVE_REPORT.en.md        # 英語エグゼクティブレポート（pandoc → PDF ソース）
├── EXECUTIVE_REPORT.ja.md        # 日本語エグゼクティブレポート（pandoc → PDF ソース）
│   ├── analysis_metrics.txt      # 機械可読なキーメトリクス
│   ├── *.png                     # 12種以上の論文品質図表（EN & JA）
│   ├── *.tex                     # 論文投稿用LaTeXテーブル
│   └── pdf/                      # コンパイル済みPDFレポート（EN & JA）
├── README.md
└── README.ja.md
```

---

## 8. データソース

| ソース | 種別 | カバレッジ | 行数 |
|--------|------|-----------|------|
| **AIカメラ**（Tojinbo-Shotaro） | エッジAI人数カウント（5分間隔） | 2024-12 → 2026-02 | 〜170K |
| **JMA**（三国・福井・勝山） | 時間別：降水・気温・日照・風速・湿度・積雪 | 2024-01 → 2026-02 | 〜140K |
| **Route Search Impressions (RSI)** | 日次：経路検索・地図表示・レビュー（47地点） | 2024-01 → 2026-02 | 〜35K |
| **北陸観光アンケート** | 満足度・NPS・自由記述（福井／石川／富山） | 2023 → 2026 | **97,719** |
| **福井観光アンケート（詳細）** | 消費額・属性・旅行パターン | 2022 → 2025 | 〜1M |

**`validator.py` による監査行数合計：** 約140万行

---

## 9. 再現手順

### セットアップ

```bash
# 作業ディレクトリとデータリポジトリを準備
mkdir hokuriku-workspace && cd hokuriku-workspace
git clone https://github.com/code4fukui/fukui-kanko-people-flow-data.git
git clone https://github.com/code4fukui/fukui-kanko-trend-report.git
git clone https://github.com/code4fukui/opendata.git
git clone https://github.com/code4fukui/fukui-kanko-survey.git

# 本リポジトリをクローンしてインストール
git clone https://github.com/amilkh/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance
pip install ".[dev]"
```

### コマンド

| コマンド | 内容 |
|---------|------|
| `python -m src.run_analysis` | フルパイプライン実行 → 図表・メトリクス・LaTeXテーブル生成 |
| `pandoc EXECUTIVE_REPORT.en.md --pdf-engine=xelatex -o output/pdf/executive_report_en.pdf` | 英語PDF生成 |
| `pandoc EXECUTIVE_REPORT.ja.md --pdf-engine=xelatex -o output/pdf/executive_report_ja.pdf` | 日本語PDF生成 |
| `pytest` | テスト実行 |
| `pytest --cov=src --cov-report=html` | カバレッジレポート付きテスト |
| `ruff check src/ tests/` | リントチェック |

> **PDF前提:** `sudo apt-get install -y pandoc texlive-xetex texlive-lang-japanese fonts-noto-cjk`
>
> **設定:** `HTAG_CONFIG=/path/to/settings.yaml` でカスタム設定を指定（デフォルト: `config/settings.yaml`）。

すべての生成物は `output/` に書き出されます：図表（EN & JA）、LaTeXテーブル、エグゼクティブレポート、コンパイル済みPDF。

---

## 10. モジュラーアーキテクチャ

パイプラインは厳格な**関心の分離**に従っています：

```python
# エントリポイント: src/run_analysis.py
cfg = load_config()                           # config.py
rpt = Reporter(cfg)                           # report.py
validation = validate_pipeline(cfg, rpt)      # validator.py
data = load_all_data(cfg, rpt)                # data_loader.py
daily, features = build_features(daily, ..)   # feature_engineering.py
ols = fit_ols(model_df, features, rpt)        # models.py
rf  = fit_random_forest(model_df, ..)         # models.py
robust = robustness_suite(model_df, ..)       # models.py
gap = compute_opportunity_gap(daily, ..)      # economics.py
kansei = discomfort_index_analysis(..)        # kansei.py
spatial = multi_node_analysis(cfg, ..)        # spatial.py
export_all_tables(results, ..)                # latex_export.py
```

すべてのモジュールは `Reporter` インスタンスを受け取り、決定論的なログ出力を行います。どのモジュールも `print()` を直接使用せず、すべての出力は集中型レポーターを経由します。

---

## 11. テスト & 検証

```
tests/
├── test_models.py     # OLS R²・RF特徴重要度・DW・エッジケース
├── test_kansei.py     # DI手計算・体感温度・ゴールデン値検証
├── test_validator.py  # スキーマ・外れ値・日付ギャップ・ドリフト検知
├── test_features.py   # カレンダー・気象強度・ラグ・エンコーディング
└── test_math.py       # 統計関数の数値正確性
```

`src/validator.py` は全データソースを自動監査します：
- **スキーマ不整合** — バージョン間での列の追加・削除
- **データドリフト** — 3ヶ月スライディングウィンドウでのKolmogorov–Smirnovテスト
- **外れ値** — 列ごとのIQR・Zスコア検出
- **日付ギャップ** — 時系列連続性の欠損日検出
- **ドメイン違反** — 負の降水量・極端な気温

---

## ライセンス

本プロジェクトは [MITライセンス](LICENSE) の下で公開されています。Copyright © 2026 Amil Khanzada.

タグ付きリリース [v1.0.0](https://github.com/amilkh/hokuriku-tourism-ai-governance/releases/tag/v1.0.0) は Measurement 誌論文（DOI: [10.1016/j.measurement.2026.122500](https://doi.org/10.1016/j.measurement.2026.122500)）の再現性アーカイブであり、Zenodo（DOI: [10.5281/zenodo.21271429](https://doi.org/10.5281/zenodo.21271429)）に恒久保存されています。

> 注: 本公開履歴は公開用に整理されています（内部原稿・会議資料・生成物を除外し、データソースの名称を出版論文と統一）。

## コントリビューター

カンザダ・アミルが開発し、以下のオープンソース貢献をいただきました。

- **Ibrahim Mustafa** — 時系列ベンチマークとRFアブレーション（`src/benchmark.py`）
- **Mohammed Abdullah** — プライバシー・サニタイズ層とゼロショット感性診断（`src/privacy_nlp.py`）
- **Dina Belay** — 再現性ドキュメントとクロスプラットフォーム修正
- **Dawood Imtiaz** — インタラクティブ・ダッシュボード（`dashboard/`）
- **Mitali (Hacxmr)** — Windows エンコーディング修正

現時点ではライセンスを付与していません。再利用・再配布・公開には著者による事前の書面許諾が必要です。
