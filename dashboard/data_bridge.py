"""
Data Bridge Module — Hokuriku Tourism AI Governance Dashboard
-----------------------------------------------------------------
Bridges the dashboard to the DHDE pipeline's heterogeneous data sources.
Loads JMA weather, AI camera people-flow, Google Business Profile, and
Hokuriku survey data. Provides synthetic fallback data when source
repositories are unavailable, ensuring the dashboard is always demonstrable.

Author  : Dawood Imtiaz
Date    : 2026-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import traceback

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


# -------------------------------------------------------
#  Kansei Comfort Formulas (mirrored from src/kansei.py)
# -------------------------------------------------------

def discomfort_index(temp_c: float, humidity_pct: float) -> float:
    """DI = 0.81·T + 0.01·H·(0.99·T − 14.3) + 46.3"""
    return 0.81 * temp_c + 0.01 * humidity_pct * (0.99 * temp_c - 14.3) + 46.3


def wind_chill(temp_c: float, wind_speed_kmh: float) -> float:
    """WC = 13.12 + 0.6215T − 11.37V^0.16 + 0.3965TV^0.16"""
    if temp_c > 10 or wind_speed_kmh <= 4.8:
        return temp_c
    v = wind_speed_kmh
    return 13.12 + 0.6215 * temp_c - 11.37 * v**0.16 + 0.3965 * temp_c * v**0.16


# -----------------------------------------------------------
#  Known JMA Format — Exact column mapping from actual files
# -----------------------------------------------------------
#
#  Actual JMA columns (from debug):
#    timestamp, snow_depth_cm, snowfall_1h_cm, temp_c,
#    precip_1h_mm, sun_1h_h, wind_speed_ms, weather_type, humidity_pct
#
#  We rename to canonical names used throughout the dashboard.

JMA_COLUMN_RENAME = {
    "timestamp":       "datetime",
    "temp_c":          "temperature_c",
    "precip_1h_mm":    "precipitation_mm",
    "sun_1h_h":        "sunshine_hours",
    "wind_speed_ms":   "wind_speed_ms",      # already correct
    "humidity_pct":    "humidity_pct",         # already correct
    "snow_depth_cm":   "snow_depth_cm",        # already correct
    "snowfall_1h_cm":  "snowfall_1h_cm",       # keep separate (not same as depth)
    "weather_type":    "weather_type",         # keep as-is
}

# -------------------------------------------------------------------
#  Known Survey Format  (Exact Japanese column names from all.csv)
# -------------------------------------------------------------------

SURVEY_COLUMN_RENAME = {
    "都道府県":             "prefecture",
    "NPS":                  "nps_score",
    "年代":                 "age_group",
    "回答日時":             "date",
    "回答月":               "response_month",
    "回答エリア":           "area",
    "市町村":               "municipality",
    "同行者":               "travel_party",
    "エリア総消費額":       "spending_yen",
    "県内消費額":           "spending_in_prefecture",
    "宿泊数（全体）":       "nights_total",
    "宿泊数（県内）":       "nights_in_prefecture",
    "エリア訪問回数":       "visit_count",
    "今後の来訪意向":       "revisit_intention",
    "性別":                 "gender",
    # Text columns for NLP
    "満足度の理由":         "satisfaction_reason",
    "不便さの内容":         "inconvenience_detail",
    "推奨項目":             "recommendation_reason",
    "福井県に求めるもの":   "requests_for_fukui",
    "施設に求めるもの":     "requests_for_facility",
    "福井県内での交通手段の満足度の理由": "transport_satisfaction_reason",
}

# Columns to combine into a single free_text for NLP analysis
SURVEY_TEXT_COLUMNS = [
    "satisfaction_reason",        # 満足度の理由
    "inconvenience_detail",       # 不便さの内容
    "recommendation_reason",      # 推奨項目
    "requests_for_fukui",         # 福井県に求めるもの
    "requests_for_facility",      # 施設に求めるもの
]


# ---------------------------------------
#  DataBridge Class
# ---------------------------------------

class DataBridge:
    """Unified data access layer for the dashboard."""

    STATION_MAP = {
        "mikuni":     {"node": "A — Tojinbo",      "lat": 36.253, "lon": 136.148},
        "fukuicity":  {"node": "B — Fukui Station", "lat": 36.065, "lon": 136.221},
        "katsuyama":  {"node": "C — Katsuyama",     "lat": 36.061, "lon": 136.501},
        "mihama":     {"node": "D — Mihama",         "lat": 35.583, "lon": 135.883},
    }

    def __init__(self):
        self.data_status: Dict[str, str] = {}
        self._jma_cache: Optional[pd.DataFrame] = None
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._survey_cache: Optional[pd.DataFrame] = None

    # ───────────────────────────────────────────────────────────
    #  JMA Weather Loading
    # ───────────────────────────────────────────────────────────

    def load_jma_data(self) -> pd.DataFrame:
        """Load JMA weather observations from committed CSV files."""
        if self._jma_cache is not None:
            return self._jma_cache

        jma_dir = PROJECT_ROOT / "jma"
        if not jma_dir.exists():
            logger.warning(f"JMA directory not found: {jma_dir}")
            self.data_status["jma"] = "⚠️ Demo data (jma/ not found)"
            self._jma_cache = self._generate_demo_weather()
            return self._jma_cache

        # Find JMA CSV files
        jma_files = sorted(jma_dir.glob("jma_*.csv"))
        if not jma_files:
            self.data_status["jma"] = "⚠️ Demo data (no jma_*.csv files)"
            self._jma_cache = self._generate_demo_weather()
            return self._jma_cache

        logger.info(f"Found {len(jma_files)} JMA files: {[f.name for f in jma_files]}")

        frames = []
        for fp in jma_files:
            try:
                # Extract station name: jma_fukuicity_hourly_8.csv → fukuicity
                station = self._extract_station_name(fp.stem)

                # Read CSV (all files are UTF-8 based on debug)
                df = pd.read_csv(fp, encoding="utf-8")
                logger.info(f"Read {fp.name}: {len(df)} rows, cols={list(df.columns)}")

                # Rename columns using known mapping
                df = df.rename(columns=JMA_COLUMN_RENAME)

                # Parse datetime
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                elif "timestamp" in df.columns:
                    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
                else:
                    # Fallback: try first column
                    df["datetime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

                # Coerce numeric columns
                numeric_targets = [
                    "temperature_c", "precipitation_mm", "sunshine_hours",
                    "wind_speed_ms", "humidity_pct", "snow_depth_cm",
                    "snowfall_1h_cm",
                ]
                for col in numeric_targets:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                df["station"] = station

                valid_dates = df["datetime"].notna().sum()
                logger.info(
                    f"  ✅ Station '{station}': {len(df)} rows, "
                    f"{valid_dates} valid dates"
                )
                frames.append(df)

            except Exception as e:
                logger.error(f"Failed to load {fp.name}: {e}")
                logger.debug(traceback.format_exc())

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.dropna(subset=["datetime"], inplace=True)
            combined.sort_values(["station", "datetime"], inplace=True)

            n_stations = combined["station"].nunique()
            self.data_status["jma"] = (
                f"✅ {n_stations} stations, {len(combined):,} rows"
            )
            logger.info(
                f"JMA loaded: {n_stations} stations, {len(combined):,} total rows"
            )
            self._jma_cache = combined
        else:
            logger.warning("All JMA files failed to load — using demo data")
            self.data_status["jma"] = "⚠️ Demo data (all files failed)"
            self._jma_cache = self._generate_demo_weather()

        return self._jma_cache

    @staticmethod
    def _extract_station_name(stem: str) -> str:
        """
        Extract station name from filename stem.
        Examples:
            jma_fukuicity_hourly_8  → fukuicity
            jma_katsuyama_hourly_8  → katsuyama
            jma_mikuni_hourly_8     → mikuni
            jma_mihama_hourly_8     → mihama
        """
        name = stem.lower()
        # Remove prefix
        if name.startswith("jma_"):
            name = name[4:]
        # Remove _hourly_N or _daily_N suffix
        parts = name.split("_")
        # Keep only the station name part (before "hourly"/"daily")
        clean_parts = []
        for p in parts:
            if p in ("hourly", "daily", "monthly") or p.isdigit():
                break
            clean_parts.append(p)
        return "_".join(clean_parts) if clean_parts else name

    # ───────────────────────────────────────────────────────────
    #  Survey Data Loading
    # ───────────────────────────────────────────────────────────

    def load_survey_data(self) -> pd.DataFrame:
        """
        Load Hokuriku survey data. Prioritises all.csv (main survey).
        Ensures a free_text column exists for NLP analysis.
        """
        if self._survey_cache is not None:
            return self._survey_cache

        survey_dir = WORKSPACE_ROOT / "fukui-kanko-survey"

        if not survey_dir.exists():
            logger.info("Survey directory not found — using demo data")
            self.data_status["survey"] = "⚠️ Demo survey data"
            self._survey_cache = self._generate_demo_survey()
            return self._survey_cache

        # ── Strategy 1: Look for all.csv specifically (main survey) ──
        all_csv = survey_dir / "all.csv"
        if all_csv.exists():
            df = self._try_read_survey(all_csv)
            if df is not None and len(df) > 10:
                df = self._normalize_survey(df)
                self.data_status["survey"] = f"✅ {len(df):,} rows (all.csv)"
                self._survey_cache = df
                logger.info(f"Survey loaded from all.csv: {len(df)} rows")
                return df

        # ── Strategy 2: Find the largest CSV ──
        csv_files = list(survey_dir.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} survey CSV files")

        best_df = None
        best_size = 0
        best_name = ""

        for fp in csv_files:
            try:
                df = self._try_read_survey(fp)
                if df is not None and len(df) > best_size and len(df.columns) > 5:
                    best_df = df
                    best_size = len(df)
                    best_name = fp.name
            except Exception:
                continue

        if best_df is not None:
            best_df = self._normalize_survey(best_df)
            self.data_status["survey"] = f"✅ {len(best_df):,} rows ({best_name})"
            self._survey_cache = best_df
            logger.info(f"Survey loaded from {best_name}: {len(best_df)} rows")
            return best_df

        # ── Fallback: demo data ──
        logger.info("No usable survey files — using demo data")
        self.data_status["survey"] = "⚠️ Demo survey data"
        self._survey_cache = self._generate_demo_survey()
        return self._survey_cache

    @staticmethod
    def _try_read_survey(filepath: Path) -> Optional[pd.DataFrame]:
        """Try reading a survey CSV with multiple encodings."""
        for enc in ["utf-8", "utf-8-sig", "shift_jis", "cp932"]:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            except Exception:
                return None
        return None

    def _normalize_survey(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize survey columns using known Japanese→English mappings.
        Creates a combined free_text column and a numeric satisfaction column.
        """
        logger.info(f"Normalizing survey: {len(df)} rows, {len(df.columns)} columns")

        # ── Rename known Japanese columns ──
        rename_map = {}
        for ja_name, en_name in SURVEY_COLUMN_RENAME.items():
            if ja_name in df.columns:
                rename_map[ja_name] = en_name

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Renamed {len(rename_map)} columns")

        # ── Create numeric satisfaction from available sources ──

        # Priority 1: nps_score (already numeric, 1-10 scale)
        if "nps_score" in df.columns:
            df["nps_score"] = pd.to_numeric(df["nps_score"], errors="coerce")

        # Priority 2: 満足度(商品・サービス) — Japanese text satisfaction
        sat_text_col = None
        for col_name in ["満足度(商品・サービス)", "福井県内での交通手段の満足度"]:
            if col_name in df.columns:
                sat_text_col = col_name
                break

        if sat_text_col is not None:
            # Map Japanese satisfaction text to numeric scores
            text_to_num = {
                "とても満足": 5.0,
                "満足": 4.0,
                "どちらでもない": 3.0,
                "不満": 2.0,
                "とても不満": 1.0,
                "選択肢なし": np.nan,
            }
            df["satisfaction"] = df[sat_text_col].map(text_to_num)
            valid = df["satisfaction"].notna().sum()
            logger.info(f"Mapped '{sat_text_col}' to numeric satisfaction: "
                        f"{valid} valid values")
        elif "nps_score" in df.columns:
            # Convert NPS (1-10) to satisfaction (1-5)
            df["satisfaction"] = (df["nps_score"] / 2).clip(1, 5).round(1)
            logger.info("Created satisfaction from nps_score")
        else:
            df["satisfaction"] = np.nan
            logger.warning("No satisfaction source found")

        # ── Parse date if available ──
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # ── Create combined free_text from all text columns ──
        text_parts = []
        found_text_cols = []
        for col in SURVEY_TEXT_COLUMNS:
            if col in df.columns:
                text_parts.append(df[col].fillna("").astype(str))
                found_text_cols.append(col)

        if text_parts:
            df["free_text"] = pd.DataFrame(text_parts).T.apply(
                lambda row: "。".join(s for s in row if s.strip()), axis=1
            )
            df["free_text"] = df["free_text"].replace("", np.nan)
            valid_texts = df["free_text"].notna().sum()
            logger.info(
                f"Created free_text from {len(found_text_cols)} columns: "
                f"{valid_texts} non-empty entries"
            )
        else:
            text_col = self._find_longest_text_column(df)
            if text_col:
                df["free_text"] = df[text_col]
            else:
                df["free_text"] = self._generate_text_from_scores(df)

        return df

    @staticmethod
    def _find_longest_text_column(df: pd.DataFrame) -> Optional[str]:
        """Find the string column with longest average text."""
        best_col = None
        best_len = 0
        for col in df.select_dtypes(include=["object"]).columns:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if avg_len > 15 and unique_ratio > 0.05 and avg_len > best_len:
                best_col = col
                best_len = avg_len
        return best_col

    @staticmethod
    def _generate_text_from_scores(df: pd.DataFrame) -> pd.Series:
        """Generate synthetic text based on satisfaction scores."""
        np.random.seed(42)
        pos = [
            "東尋坊の景色が素晴らしかった。また来たい。",
            "福井の海鮮が最���でした。カニが特に美味しかった。",
            "恐竜博物館は子供も大人も楽しめる施設です。",
            "永平寺の雰囲気がとても良かった。",
            "温泉が最高でした。芦原温泉は泉質が良い。",
            "The scenery at Tojinbo was breathtaking!",
        ]
        neg = [
            "冬の天気が悪くて観光が辛かった。",
            "交通アクセスが不便。電車の本数が少ない。",
            "冬は寒すぎて外を歩けなかった。",
            "Winter weather made sightseeing difficult.",
        ]
        neu = [
            "普通の旅行でした。",
            "もう少し観光スポットが多いと良い。",
            "An average trip overall.",
        ]

        texts = []
        sat = df.get("satisfaction", pd.Series(dtype=float))
        for i in range(len(df)):
            s = sat.iloc[i] if i < len(sat) and pd.notna(sat.iloc[i]) else 3.5
            if s >= 4.0:
                texts.append(np.random.choice(pos))
            elif s <= 2.5:
                texts.append(np.random.choice(neg))
            else:
                texts.append(np.random.choice(neu))
        return pd.Series(texts, index=df.index)

    # ───────────────────────────────────────────────────────────
    #  Camera Data
    # ───────────────────────────────────────────────────────────

    def load_camera_data(self) -> Optional[pd.DataFrame]:
        """Attempt to load AI camera people-flow data."""
        camera_dir = WORKSPACE_ROOT / "fukui-kanko-people-flow-data"
        if not camera_dir.exists():
            self.data_status["camera"] = "⚠️ Sibling repo not cloned"
            return None

        csv_files = list(camera_dir.rglob("*.csv"))
        if not csv_files:
            self.data_status["camera"] = "⚠️ No CSV files found"
            return None

        frames = []
        for fp in csv_files[:50]:
            try:
                df = pd.read_csv(fp, encoding="utf-8")
                frames.append(df)
            except Exception:
                continue

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            self.data_status["camera"] = f"✅ {len(combined):,} rows"
            return combined

        self.data_status["camera"] = "⚠️ Could not parse camera data"
        return None

    # ───────────────────────────────────────────────────────────
    #  Daily Aggregated Weather
    # ───────────────────────────────────────────────────────────

    def get_daily_weather(self, station: str = "mikuni") -> pd.DataFrame:
        """Daily-aggregated weather enriched with Kansei comfort metrics."""
        if station in self._daily_cache:
            return self._daily_cache[station]

        jma = self.load_jma_data()
        station_df = jma[jma["station"] == station].copy()

        if station_df.empty:
            # Fall back to first available station
            available = sorted(jma["station"].unique())
            if available:
                station_df = jma[jma["station"] == available[0]].copy()
                logger.warning(f"Station '{station}' not found, using '{available[0]}'")
            else:
                return pd.DataFrame()

        station_df["date"] = station_df["datetime"].dt.date

        # Build aggregation rules
        numeric_cols = station_df.select_dtypes(include=[np.number]).columns
        agg = {}
        for c in numeric_cols:
            cl = c.lower()
            if "precip" in cl or "rain" in cl or "snowfall" in cl:
                agg[c] = "sum"
            elif "snow_depth" in cl:
                agg[c] = "max"
            else:
                agg[c] = "mean"

        if not agg:
            return pd.DataFrame()

        daily = station_df.groupby("date").agg(agg).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])

        # ── Kansei enrichment ──
        if "temperature_c" in daily.columns and "humidity_pct" in daily.columns:
            daily["discomfort_index"] = daily.apply(
                lambda r: discomfort_index(
                    float(r["temperature_c"]) if pd.notna(r["temperature_c"]) else 20,
                    float(r["humidity_pct"]) if pd.notna(r["humidity_pct"]) else 50,
                ),
                axis=1,
            )

        if "temperature_c" in daily.columns and "wind_speed_ms" in daily.columns:
            daily["wind_chill_c"] = daily.apply(
                lambda r: wind_chill(
                    float(r["temperature_c"]) if pd.notna(r["temperature_c"]) else 20,
                    float(r["wind_speed_ms"]) * 3.6
                    if pd.notna(r["wind_speed_ms"]) else 0,
                ),
                axis=1,
            )

        # ── Calendar features ──
        daily["month"] = daily["date"].dt.month
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
        daily["season"] = daily["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        })

        self._daily_cache[station] = daily
        return daily

    # -------------------------------------
    #  Demo Data Generators
    # -------------------------------------

    def _generate_demo_weather(self) -> pd.DataFrame:
        """Generate realistic synthetic JMA data for 4 stations."""
        np.random.seed(42)
        stations = ["mikuni", "fukuicity", "katsuyama", "mihama"]
        date_range = pd.date_range("2024-01-01", "2025-12-31", freq="h")

        frames = []
        for station in stations:
            n = len(date_range)
            doy = date_range.dayofyear.values
            base_temp = 15 + 12 * np.sin(2 * np.pi * (doy - 100) / 365)
            temp = base_temp + np.random.normal(0, 3, n)

            precip = np.maximum(0, np.random.exponential(1.5, n))
            precip[np.random.random(n) > 0.35] = 0.0

            df = pd.DataFrame({
                "datetime": date_range,
                "station": station,
                "precipitation_mm": precip,
                "temperature_c": temp,
                "sunshine_hours": np.clip(
                    np.random.normal(0.4, 0.25, n), 0, 1
                ),
                "wind_speed_ms": np.maximum(0.5, np.random.lognormal(1.0, 0.5, n)),
                "humidity_pct": np.clip(
                    65 + 15 * np.sin(2 * np.pi * (doy + 80) / 365)
                    + np.random.normal(0, 8, n), 20, 100,
                ),
                "snow_depth_cm": np.where(
                    temp < 2, np.maximum(0, np.random.exponential(5, n)), 0
                ),
            })
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _generate_demo_survey() -> pd.DataFrame:
        """Generate realistic demo survey with Japanese free text."""
        np.random.seed(123)
        n = 2000

        prefectures = np.random.choice(
            ["福井県", "石川県", "富山県"], n, p=[0.45, 0.35, 0.20]
        )
        satisfaction = np.clip(np.random.normal(3.8, 0.9, n), 1, 5).round(1)

        positive = [
            "東尋坊の景色が素晴らしかった。また来たい。",
            "福井の海鮮が最高でした。カニが特に美味しかった。",
            "恐竜博物館は子供も大人も楽しめる素晴らしい施設です。",
            "永平寺の雰囲気がとても良かった。心が落ち着きました。",
            "地元の方々がとても親切で、温かい歓迎を受けました。",
            "レインボーラインからの三方五湖の眺めは絶景でした。",
            "温泉が最高でした。芦原温泉は泉質が良い。",
            "越前そばが美味しかった。また食べに行きたい。",
            "丸岡城は歴史を感じられる素敵な場所でした。",
            "北陸新幹線で行きやすくなって嬉しい。",
            "The scenery at Tojinbo was breathtaking!",
            "Fukui's seafood was incredible. The crab was outstanding.",
            "The Dinosaur Museum is a world-class attraction.",
        ]
        negative = [
            "冬の天気が悪くて観光が辛かった。雪が多すぎる。",
            "交通アクセスが不便。電車の本数が少ない。",
            "観光案内が少なくて、どこに行けばいいかわからなかった。",
            "冬は寒すぎて外を歩けなかった。防寒対策が必要。",
            "観光地間の移動が大変。レンタカーが必須だと思う。",
            "英語の案内が少ない。外国人には不便。",
            "Winter weather made sightseeing very difficult.",
            "Public transportation needs significant improvement.",
        ]
        neutral = [
            "普通の旅行でした。特に良くも悪くもなかった。",
            "もう少し観光スポットが多いと良いと思う。",
            "食事は良かったが、宿泊施設の選択肢が少ない。",
            "静かな場所で落ち着けた。人混みがないのは良い。",
            "An average trip. Nothing particularly special.",
        ]

        texts = []
        for s in satisfaction:
            if s >= 4.0:
                texts.append(np.random.choice(positive))
            elif s <= 2.5:
                texts.append(np.random.choice(negative))
            else:
                texts.append(np.random.choice(neutral))

        months = np.random.choice(range(1, 13), n)
        nps = np.clip(
            (satisfaction * 2.2 - 2 + np.random.normal(0, 1.5, n)), 0, 10
        ).round(0)

        return pd.DataFrame({
            "date": pd.to_datetime(
                [f"2024-{m:02d}-{np.random.randint(1, 28):02d}" for m in months]
            ),
            "prefecture": prefectures,
            "satisfaction": satisfaction,
            "nps_score": nps,
            "free_text": texts,
            "age_group": np.random.choice(
                ["20代", "30代", "40代", "50代", "60代以上"], n
            ),
            "travel_party": np.random.choice(
                ["一人旅", "カップル", "家族", "友人", "団体"],
                n, p=[0.10, 0.25, 0.30, 0.20, 0.15],
            ),
            "spending_yen": np.clip(
                np.random.lognormal(9.5, 0.6, n), 3000, 100000
            ).round(-2),
        })

    # ─── Utility ────────────────────────────────────

    def get_data_status_report(self) -> Dict[str, str]:
        """Return data-source availability summary."""
        self.load_jma_data()
        self.load_survey_data()
        self.load_camera_data()
        return self.data_status