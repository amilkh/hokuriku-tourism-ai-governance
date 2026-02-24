# Hokuriku Tourism AI Governance

> エグゼクティブレポート：
> - [English Executive Report](output/EXECUTIVE_REPORT.md)
> - [日本語エグゼクティブレポート](output/EXECUTIVE_REPORT.ja.md)
>
> 他の言語で読む：
> - [English](README.md)

## 福井県におけるAI主導の来訪需要予測と「過少な賑わい」分析

博士研究・北陸地域観光AIガバナンス助成プロジェクト

このリポジトリは、東尋坊（福井県）の日次来訪者数を予測するための再現可能な分析パイプラインを提供します。AIカメラデータ、気象庁（JMA）観測、経路検索インプレッション（RSI）のルート検索意図、北陸観光アンケートを統合しています。

### 主要結果

| 項目 | 値 |
|---|---|
| OLS R² | 0.810（調整済み R² = 0.802） |
| RF 5-fold CV R² | 0.557 ± 0.131 |
| 一階差分モデル R²（自己相関補正後） | 0.708 |
| LDV R² / DW | 0.848 / 1.893 |
| 最大予測因子 | `directions` (RSI)（ルート検索）, r = +0.781 |
| 石川→東尋坊の県間シグナル | r = +0.537 |
| 来訪者数と満足度 | r = +0.161（p = 0.001）— オーバーツーリズム兆候なし |
| 失われた来訪者（Opportunity Gap） | 85,512 |
| 冬季の気象感度 | 夏季の 6.29 倍 |
| 低満足レビューでの過少賑わい言及 | 11.4 倍（6.2% vs 0.5%） |
| 福井県の全国順位（冬季） | 47 位 / 47 都道府県 |

### リポジトリ構成

```
hokuriku-tourism-ai-governance/
├── run_analysis.py               # メイン分析パイプライン
├── requirements.txt              # Python依存関係
├── jma/                          # 気象庁データ（同梱）
├── output/                       # 生成物
│   ├── EXECUTIVE_REPORT.md
│   ├── EXECUTIVE_REPORT.ja.md
│   ├── *.png                     # 図表（英語版・日本語版）
│   └── analysis_metrics.txt      # 機械可読な指標サマリ
├── ../fukui-kanko-people-flow-data/  # AIカメラ日次データ（兄弟リポジトリ）
├── ../fukui-kanko-trend-report/      # 経路検索インプレッション（RSI）データ（兄弟リポジトリ）
├── ../opendata/                      # 北陸観光アンケートデータ（兄弟リポジトリ）
├── README.md
└── README.ja.md
```

### データソース

| ソース | 説明 | 期間 |
|---|---|---|
| AI Camera（Tojinbo-Shotaro） | エッジAIカメラによる日次人数カウント | 2024-12 → 2026-02 |
| JMA（気象庁） | 時間別の降水・気温・日照・風速 | 2024-01 → 2026-02 |
| Route Search Impressions (RSI) | 福井県観光47地点の経路検索・閲覧・レビュー | 2024-01 → 2026-02 |
| 北陸観光アンケート | 95,653件（満足度/NPS/自由記述） | 2023 → 2026 |

### 再現手順

解析を再現するには、本リポジトリと依存データリポジトリを同じ親ディレクトリ配下に配置してください。

```bash
# 1) 作業ディレクトリ作成
mkdir hokuriku-workspace && cd hokuriku-workspace

# 2) 依存データリポジトリをクローン
git clone https://github.com/YOUR_ORG/fukui-kanko-people-flow-data.git
git clone https://github.com/YOUR_ORG/fukui-kanko-trend-report.git
git clone https://github.com/YOUR_ORG/opendata.git

# 3) 本リポジトリをクローン
git clone https://github.com/YOUR_ORG/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance

# 4) 依存関係インストールと実行
pip install -r requirements.txt
python run_analysis.py
```

出力は `output/` ディレクトリに保存されます。

### 引用

この研究を利用する場合は、以下の形式で引用してください。

```
@misc{hokuriku-tourism-ai-governance-2026,
  author = {Amil Khanzada},
  title  = {AI-Driven Visitor Demand Forecasting and Under-vibrancy Analysis for Fukui Prefecture},
  year   = {2026},
  url    = {https://github.com/YOUR_ORG/hokuriku-tourism-ai-governance},
  note   = {福井大学 地域創生推進本部 特命助教}
}
```

### ライセンス

Copyright © Amil Khanzada. All rights reserved.

現時点ではライセンスは付与していません。再利用・再配布・公開には、
著者による事前の書面許諾が必要です。

