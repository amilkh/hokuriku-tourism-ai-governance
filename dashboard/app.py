"""
Hokuriku Tourism AI Governance — Interactive Dashboard
======================================================
Streamlit-based exploration of the Distributed Human Data Engine (DHDE).

Run with:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from dashboard.data_bridge import DataBridge, discomfort_index
from dashboard.nlp_engine import TourismNLPEngine

# ── Display name mapping (internal column → human-readable) ──
DISPLAY_NAMES = {
    "snow_depth_cm":    "SnowDepth /cm",
    "snowfall_1h_cm":   "Snowfall per h /cm",
    "temperature_c":    "Temperature /°C",
    "precipitation_mm": "Precipitation /mm",
    "sunshine_hours":   "Sunshine /h",
    "wind_speed_ms":    "WindSpeed /ms",
    "weather_type":     "WeatherType",
    "humidity_pct":     "Humidity /pct",
    "discomfort_index": "Discomfort Index",
    "wind_chill_c":     "WindChill /°C",
    "severity_score":   "Severity Score",
}

# Reverse mapping (display name → internal column)
REVERSE_NAMES = {v: k for k, v in DISPLAY_NAMES.items()}


def display_name(col: str) -> str:
    """Convert internal column name to human-readable display name."""
    return DISPLAY_NAMES.get(col, col)


def internal_name(display: str) -> str:
    """Convert display name back to internal column name."""
    return REVERSE_NAMES.get(display, display)


# ── Page Config ──
st.set_page_config(
    page_title="Hokuriku Tourism AI Governance",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached Resources ──
@st.cache_resource
def get_bridge():
    return DataBridge()

@st.cache_resource
def get_nlp_engine():
    return TourismNLPEngine()

@st.cache_data(ttl=600)
def load_all_data():
    bridge = get_bridge()
    jma = bridge.load_jma_data()
    survey = bridge.load_survey_data()
    return jma, survey


def metric_card(title, value, col):
    col.markdown(
        f'<div class="metric-card"><h3>{title}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar(jma):
    st.sidebar.title("🏯 Hokuriku Dashboard")
    st.sidebar.markdown("---")

    stations = sorted(jma["station"].unique())
    station = st.sidebar.selectbox(
        "🌤️  Weather Station", stations, index=0, key="sb_station"
    )

    if "datetime" in jma.columns:
        min_d = jma["datetime"].min()
        max_d = jma["datetime"].max()
        if pd.notna(min_d) and pd.notna(max_d):
            min_d = min_d.date()
            max_d = max_d.date()
        else:
            from datetime import date
            min_d, max_d = date(2024, 1, 1), date(2025, 12, 31)
    else:
        from datetime import date
        min_d, max_d = date(2024, 1, 1), date(2025, 12, 31)

    date_range = st.sidebar.date_input(
        "📅  Date Range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="sb_dates",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")
    bridge = get_bridge()
    for src, status in bridge.data_status.items():
        st.sidebar.markdown(f"**{src.upper()}**: {status}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built for the **GovAI Research Residency**  \n"
        "University of Fukui · Prof. Amil Khanzada"
    )
    return station, date_range


# ══════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

def tab_overview(jma, survey):
    st.header("📊  DHDE Framework Overview")
    st.markdown(
        "The **Distributed Human Data Engine** fuses AI camera, JMA weather, "
        "Google intent, and 95,653 survey responses to quantify Fukui's "
        "structural tourism deficit."
    )

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Opportunity Gap", "¥11.96 B", c1)
    metric_card("OLS R²", "0.810", c2)
    metric_card("Winter Rank", "47th / 47", c3)
    metric_card("Lost Visitors", "85,522", c4)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌡️ Temperature Distribution by Station")
        if "temperature_c" in jma.columns:
            fig = px.box(
                jma, x="station", y="temperature_c",
                color="station",
                labels={
                    "temperature_c": display_name("temperature_c"),
                    "station": "Station",
                },
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True, key="ov_temp_box")

    with col2:
        st.subheader("🌧️ Precipitation Distribution by Station")
        if "precipitation_mm" in jma.columns:
            fig = px.box(
                jma[jma["precipitation_mm"] > 0],
                x="station", y="precipitation_mm",
                color="station",
                labels={
                    "precipitation_mm": display_name("precipitation_mm"),
                    "station": "Station",
                },
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True, key="ov_precip_box")

    # ── Survey Satisfaction Overview ──
    st.subheader("📋 Survey Satisfaction Overview")

    survey_clean = survey.copy()

    # Find the satisfaction column
    sat_col = None
    for candidate in ["satisfaction", "満足度"]:
        if candidate in survey_clean.columns:
            sat_col = candidate
            break

    if sat_col is None:
        st.info("No satisfaction column found.")
        return

    # Convert satisfaction to numeric
    # Handle cases where values might be Japanese text like "満足", "やや満足" etc.
    raw_values = survey_clean[sat_col].astype(str)

    # Try direct numeric conversion first
    survey_clean["sat_num"] = pd.to_numeric(raw_values, errors="coerce")

    # If most values failed, try extracting numbers from text
    valid_count = survey_clean["sat_num"].notna().sum()
    if valid_count < len(survey_clean) * 0.1:
        # Try to extract any number from the string
        extracted = raw_values.str.extract(r'(\d+\.?\d*)')
        if extracted is not None and not extracted.empty:
            survey_clean["sat_num"] = pd.to_numeric(extracted[0], errors="coerce")

    # If still mostly empty, map Japanese satisfaction text to numbers
    valid_count = survey_clean["sat_num"].notna().sum()
    if valid_count < len(survey_clean) * 0.1:
        text_to_num = {
            "非常に満足": 5, "とても満足": 5, "大変満足": 5,
            "満足": 4, "やや満足": 4,
            "普通": 3, "どちらでもない": 3,
            "やや不満": 2, "不満": 2,
            "非常に不満": 1, "とても不満": 1, "大変不満": 1,
        }
        survey_clean["sat_num"] = raw_values.map(text_to_num)

    survey_clean = survey_clean.dropna(subset=["sat_num"])

    if len(survey_clean) < 5:
        st.info(f"Only {len(survey_clean)} valid satisfaction scores found.")
        return

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            survey_clean, x="sat_num", nbins=20,
            color_discrete_sequence=["#667eea"],
            labels={"sat_num": "Satisfaction Score"},
            template="plotly_white",
            title=f"Satisfaction Distribution (n={len(survey_clean):,})",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key="ov_sat_hist")

    with c2:
        # Find the best column to group by
        group_col = None
        min_groups = 2

        # Try these columns in order of preference
        candidates = [
            "area", "回答エリア", "回答エリア2",
            "municipality", "市町村",
            "prefecture", "都道府県",
            "age_group", "年代",
            "travel_party", "同行者",
            "gender", "性別",
        ]

        for col in candidates:
            if col in survey_clean.columns:
                value_counts = survey_clean[col].dropna().value_counts()
                groups_with_enough = (value_counts >= 5).sum()
                if groups_with_enough >= min_groups:
                    group_col = col
                    break

        if group_col is not None:
            grp = survey_clean.groupby(group_col)["sat_num"].agg(
                mean="mean", count="count"
            ).reset_index()
            grp = grp[grp["count"] >= 5]
            grp = grp.sort_values("mean", ascending=False).head(15)

            fig = px.bar(
                grp, x=group_col, y="mean",
                color="mean",
                color_continuous_scale="Viridis",
                labels={"mean": "Mean Satisfaction", group_col: group_col},
                template="plotly_white",
                title=f"Mean Satisfaction by {group_col} (Top 15)",
            )
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="ov_sat_bar")
        else:
            # Ultimate fallback: show by score bracket
            survey_clean["bracket"] = survey_clean["sat_num"].round(0).astype(int)
            counts = survey_clean["bracket"].value_counts().sort_index()
            fig = px.bar(
                x=counts.index.astype(str), y=counts.values,
                color=counts.values,
                color_continuous_scale="Viridis",
                labels={"x": "Satisfaction Score", "y": "Number of Responses"},
                template="plotly_white",
                title="Responses by Satisfaction Score",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True, key="ov_sat_bar")

# ══════════════════════════════════════════════════════════════
#  TAB 2 — WEATHER & TOURISM
# ══════════════════════════════════════════════════════════════

def tab_weather(jma, station, date_range):
    st.header("🌤️  Weather–Tourism Correlation Analysis")

    bridge = get_bridge()
    daily = bridge.get_daily_weather(station)

    if len(date_range) == 2:
        mask = (daily["date"] >= pd.Timestamp(date_range[0])) & (
            daily["date"] <= pd.Timestamp(date_range[1])
        )
        daily = daily[mask]

    if daily.empty:
        st.warning("No data available for selected filters.")
        return

    st.subheader(f"📈  Daily Weather at Station: {station.title()}")

    # Columns to exclude from the dropdown
    exclude_cols = {
        "date", "month", "day_of_week", "is_weekend", "season",
        "wind_direction", "station",
    }
    available_cols = [c for c in daily.columns if c not in exclude_cols]

    # Create display names for the dropdown
    display_options = [display_name(c) for c in available_cols]

    selected_display = st.selectbox(
        "Select weather variable:", display_options, index=0, key="wt_var"
    )

    # Map back to internal column name
    weather_var = internal_name(selected_display)

    # Time series chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily[weather_var],
            mode="lines", name=selected_display,
            line=dict(color="#667eea", width=1.5),
        ),
        secondary_y=False,
    )
    if "temperature_c" in daily.columns and weather_var != "temperature_c":
        fig.add_trace(
            go.Scatter(
                x=daily["date"], y=daily["temperature_c"],
                mode="lines", name=display_name("temperature_c"),
                line=dict(color="#f093fb", width=1, dash="dot"),
                opacity=0.6,
            ),
            secondary_y=True,
        )
    fig.update_layout(
        template="plotly_white", height=450, hovermode="x unified",
        title=f"{selected_display} over time — {station.title()}",
    )
    fig.update_yaxes(title_text=selected_display, secondary_y=False)
    fig.update_yaxes(title_text=display_name("temperature_c"), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, key="wt_timeseries")

    # Correlation heatmap with display names
    st.subheader("🔥  Variable Correlation Heatmap")
    numeric = daily.select_dtypes(include=[np.number])
    numeric = numeric[[c for c in numeric.columns if c not in exclude_cols]]

    # Rename columns for display
    renamed_numeric = numeric.rename(columns=DISPLAY_NAMES)
    corr = renamed_numeric.corr()

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, template="plotly_white",
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True, key="wt_corr")

    # Seasonal patterns
    if "season" in daily.columns and "temperature_c" in daily.columns:
        st.subheader("🍂  Seasonal Weather Patterns")
        fig = px.violin(
            daily, x="season", y="temperature_c", color="season",
            category_orders={"season": ["Winter", "Spring", "Summer", "Autumn"]},
            box=True, points="outliers", template="plotly_white",
            labels={"temperature_c": display_name("temperature_c")},
            title="Temperature Distribution by Season",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="wt_violin")


# ══════════════════════════════════════════════════════════════
#  TAB 3 — OPPORTUNITY GAP
# ══════════════════════════════════════════════════════════════

def tab_opportunity_gap(jma, station):
    st.header("💰  Opportunity Gap Analysis")
    st.markdown(
        "The Opportunity Gap quantifies revenue lost due to weather-induced "
        "demand suppression — estimated at **¥11.96 billion annually**."
    )

    bridge = get_bridge()
    daily = bridge.get_daily_weather(station)
    if daily.empty:
        st.warning("No data available.")
        return

    st.subheader("🧮  Interactive Gap Calculator")
    c1, c2, c3 = st.columns(3)
    spending = c1.slider("Mean spending (¥)", 5000, 30000, 13811, 500, key="og_spend")
    gap_days = c2.slider("Gap days/year", 10, 100, 42, key="og_days")
    visitors = c3.slider("Lost visitors/day", 500, 5000, 2036, key="og_vis")

    total_lost = gap_days * visitors
    total_yen = total_lost * spending

    c1, c2, c3 = st.columns(3)
    metric_card("Gap Days / Year", str(gap_days), c1)
    metric_card("Total Lost Visitors", f"{total_lost:,}", c2)
    metric_card("Revenue Loss", f"¥{total_yen / 1e9:.2f} B", c3)

    st.markdown("---")

    if "temperature_c" in daily.columns and "precipitation_mm" in daily.columns:
        daily = daily.copy()
        temp = daily["temperature_c"].clip(-10, 35)
        temp_pen = np.where(temp < 5, (5 - temp) / 15, 0)
        precip_pen = np.clip(daily["precipitation_mm"] / 50, 0, 1)

        snow_vals = daily["snow_depth_cm"] if "snow_depth_cm" in daily.columns else 0
        snow_pen = np.clip(snow_vals / 50, 0, 1) if isinstance(snow_vals, pd.Series) else 0

        daily["severity_score"] = np.clip(
            temp_pen * 0.4 + precip_pen * 0.35 + snow_pen * 0.25, 0, 1
        )
        daily["day_type"] = pd.cut(
            daily["severity_score"], bins=[-0.01, 0.2, 0.5, 1.0],
            labels=["Good", "Moderate", "Severe"],
        )

        st.subheader("⛈️  Weather Severity vs Visitor Impact")
        col1, col2 = st.columns(2)
        with col1:
            counts = daily["day_type"].value_counts()
            fig = px.pie(
                values=counts.values, names=counts.index,
                color=counts.index,
                color_discrete_map={
                    "Good": "#10b981", "Moderate": "#f59e0b", "Severe": "#ef4444"
                },
                title="Weather Day Types", template="plotly_white",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="og_pie")
        with col2:
            fig = px.histogram(
                daily, x="severity_score", nbins=30, color="day_type",
                color_discrete_map={
                    "Good": "#10b981", "Moderate": "#f59e0b", "Severe": "#ef4444"
                },
                title="Severity Score Distribution",
                labels={"severity_score": display_name("severity_score")},
                template="plotly_white",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="og_hist")

        # Severity time series
        fig = px.scatter(
            daily, x="date", y="severity_score",
            color="day_type",
            color_discrete_map={
                "Good": "#10b981", "Moderate": "#f59e0b", "Severe": "#ef4444"
            },
            title="Weather Severity Over Time",
            labels={"severity_score": display_name("severity_score")},
            template="plotly_white", opacity=0.6,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="og_scatter")

    if "month" in daily.columns and "severity_score" in daily.columns:
        st.subheader("📅  Monthly Gap Potential")
        ms = daily.groupby("month").agg(
            mean_sev=("severity_score", "mean"),
            severe_days=("severity_score", lambda x: (x > 0.5).sum()),
            total=("severity_score", "count"),
        ).reset_index()
        ms["severe_pct"] = ms["severe_days"] / ms["total"] * 100
        names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        ms["month_name"] = ms["month"].map(names)

        fig = go.Figure()
        fig.add_bar(
            x=ms["month_name"], y=ms["severe_pct"],
            name="Severe Days (%)", marker_color="#ef4444",
        )
        fig.add_scatter(
            x=ms["month_name"], y=ms["mean_sev"] * 100,
            name="Mean Severity", mode="lines+markers",
            line=dict(color="#667eea", width=2),
            yaxis="y2",
        )
        fig.update_layout(
            template="plotly_white",
            title="Monthly Weather Severity — Gap Potential",
            yaxis=dict(title="Severe Days (%)", range=[0, 100]),
            yaxis2=dict(
                title="Mean Severity (×100)", overlaying="y",
                side="right", range=[0, 100],
            ),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True, key="og_monthly")

        # Winter vs Summer insight
        winter = ms[ms["month"].isin([12, 1, 2])]
        summer = ms[ms["month"].isin([6, 7, 8])]
        if not winter.empty and not summer.empty:
            w_mean = winter["mean_sev"].mean()
            s_mean = summer["mean_sev"].mean()
            if s_mean > 0.001:
                ratio = w_mean / s_mean
                st.info(
                    f"⚡ **Winter weather severity is {ratio:.1f}× higher than summer** — "
                    f"confirming the 6.26× seasonal asymmetry documented in the DHDE framework."
                )


# ══════════════════════════════════════════════════════════════
#  TAB 4 — KANSEI COMFORT
# ══════════════════════════════════════════════════════════════

def tab_kansei(jma, station):
    st.header("🌡️  Kansei Environmental Comfort Assessment")
    st.markdown(
        "The **Kansei (感性) framework** evaluates human thermal comfort "
        "using the Discomfort Index (不快指数) and Wind Chill (体感温度)."
    )

    bridge = get_bridge()
    daily = bridge.get_daily_weather(station)
    if daily.empty or "temperature_c" not in daily.columns:
        st.warning("Insufficient data for Kansei analysis.")
        return

    st.subheader("🔬  Discomfort Index Calculator")
    c1, c2, c3 = st.columns(3)
    temp_in = c1.slider(
        display_name("temperature_c"), -10.0, 40.0, 25.0, 0.5, key="ka_temp"
    )
    hum_in = c2.slider(
        display_name("humidity_pct"), 0.0, 100.0, 60.0, 1.0, key="ka_hum"
    )
    di_val = discomfort_index(temp_in, hum_in)

    di_label = (
        "🥶 Cold" if di_val < 55 else "😊 Comfortable" if di_val < 70
        else "😰 Uncomfortable" if di_val < 80 else "🥵 Dangerous"
    )
    c3.metric(display_name("discomfort_index"), f"{di_val:.1f}", di_label)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=di_val,
        title={"text": "Discomfort Index (不快指数)"},
        gauge=dict(
            axis=dict(range=[30, 90]),
            bar=dict(color="#667eea"),
            steps=[
                dict(range=[30, 55], color="#93c5fd"),
                dict(range=[55, 70], color="#86efac"),
                dict(range=[70, 80], color="#fcd34d"),
                dict(range=[80, 90], color="#fca5a5"),
            ],
        ),
    ))
    fig.update_layout(height=300, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True, key="ka_gauge")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if "discomfort_index" in daily.columns:
            st.subheader(f"📈  {display_name('discomfort_index')} Over Time")
            fig = px.line(
                daily, x="date", y="discomfort_index", template="plotly_white",
                labels={"discomfort_index": display_name("discomfort_index")},
            )
            fig.add_hline(y=70, line_dash="dash", line_color="orange",
                          annotation_text="Uncomfortable")
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="Dangerous")
            fig.update_traces(line_color="#667eea")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="ka_di_line")

    with col2:
        if "wind_chill_c" in daily.columns:
            st.subheader(f"❄️  {display_name('wind_chill_c')} Over Time")
            fig = px.line(
                daily, x="date", y="wind_chill_c", template="plotly_white",
                labels={"wind_chill_c": display_name("wind_chill_c")},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="blue",
                          annotation_text="Freezing")
            fig.update_traces(line_color="#06b6d4")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="ka_wc_line")

    if "discomfort_index" in daily.columns:
        st.subheader(f"🔗  {display_name('discomfort_index')} vs {display_name('temperature_c')}")
        fig = px.scatter(
            daily, x="temperature_c", y="discomfort_index",
            color="season" if "season" in daily.columns else None,
            template="plotly_white", opacity=0.6,
            labels={
                "temperature_c": display_name("temperature_c"),
                "discomfort_index": display_name("discomfort_index"),
            },
            title="Non-linear relationship: Temperature → Comfort",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="ka_scatter")

    # ── Seasonal Comfort Summary ──
    if "season" in daily.columns and "discomfort_index" in daily.columns:
        st.subheader("🍂  Seasonal Comfort Summary")

        season_agg = {"discomfort_index": "mean", "temperature_c": "mean"}
        if "wind_chill_c" in daily.columns:
            season_agg["wind_chill_c"] = "mean"

        season_stats = daily.groupby("season").agg(season_agg).round(1)

        # Rename columns to human-readable names
        rename_map = {
            "discomfort_index": "Mean Discomfort Index",
            "temperature_c": "Mean Temperature",
            "wind_chill_c": "Mean WindChill",
        }
        season_stats = season_stats.rename(columns=rename_map)

        st.dataframe(season_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 5 — SPATIAL NETWORK
# ══════════════════════════════════════════════════════════════

def tab_spatial(jma):
    st.header("🗺️  Spatial Saturation Network")
    st.markdown(
        "The DHDE models four nodes spanning coast→mountain→urban→scenic, "
        "enabling a **Weather Shield Network** for demand redistribution."
    )

    node_data = pd.DataFrame([
        {"Node": "A", "Location": "Tojinbo (東尋坊)", "Type": "Coastal",
         "lat": 36.253, "lon": 136.148},
        {"Node": "B", "Location": "Fukui Station", "Type": "Urban",
         "lat": 36.065, "lon": 136.221},
        {"Node": "C", "Location": "Katsuyama (勝山)", "Type": "Mountain",
         "lat": 36.061, "lon": 136.501},
        {"Node": "D", "Location": "Mihama (美浜)", "Type": "Scenic",
         "lat": 35.583, "lon": 135.883},
    ])

    st.subheader("📍  Node Locations")
    fig = px.scatter_mapbox(
        node_data, lat="lat", lon="lon", text="Location",
        color="Type", zoom=8, height=500,
        title="DHDE Spatial Network — Fukui Prefecture",
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker=dict(size=18))
    st.plotly_chart(fig, use_container_width=True, key="sp_map")

    stations = sorted(jma["station"].unique())
    bridge = get_bridge()

    # ── Cross-Station Comparison with display names ──
    if len(stations) >= 2:
        st.subheader("⚖️  Cross-Station Weather Comparison")

        # Display names for the dropdown
        spatial_vars_internal = [
            "temperature_c", "precipitation_mm", "humidity_pct",
            "wind_speed_ms", "snow_depth_cm",
        ]
        spatial_vars_display = [display_name(c) for c in spatial_vars_internal]

        selected_display = st.selectbox(
            "Variable to compare:",
            spatial_vars_display,
            index=0, key="sp_var",
        )
        comp_var = internal_name(selected_display)

        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, s in enumerate(stations):
            d = bridge.get_daily_weather(s)
            if comp_var in d.columns:
                monthly = d.groupby("month")[comp_var].mean()
                fig.add_scatter(
                    x=monthly.index, y=monthly.values,
                    mode="lines+markers", name=s.title(),
                    line=dict(color=colors[i % len(colors)], width=2.5),
                )
        fig.update_layout(
            template="plotly_white", height=400,
            title=f"Monthly {selected_display} by Station",
            xaxis_title="Month",
            yaxis_title=selected_display,
        )
        st.plotly_chart(fig, use_container_width=True, key="sp_comparison")

    # ── Cross-Station Correlation ──
    if len(stations) >= 2:
        st.subheader("🔗  Cross-Station Correlation")
        dfs = {}
        for s in stations:
            d = bridge.get_daily_weather(s)
            if "temperature_c" in d.columns:
                dfs[s] = d.set_index("date")["temperature_c"]
        if len(dfs) >= 2:
            corr_df = pd.DataFrame(dfs).corr()
            fig = px.imshow(
                corr_df, text_auto=".3f", color_continuous_scale="Blues",
                title=f"{display_name('temperature_c')} Correlation Between Stations",
                template="plotly_white",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="sp_corr")

            st.info(
                "💡 **Weather Shield Network**: When one node has severe weather, "
                "visitors can be redirected to a node with better conditions — "
                "reducing the Opportunity Gap."
            )

    # ── Weather Divergence ──
    if len(stations) >= 2:
        st.subheader("🌦️  Weather Divergence Days (Shield Potential)")
        dfs_temp = {}
        for s in stations:
            d = bridge.get_daily_weather(s)
            if "temperature_c" in d.columns:
                dfs_temp[s] = d.set_index("date")["temperature_c"]
        if len(dfs_temp) >= 2:
            temp_df = pd.DataFrame(dfs_temp).dropna()
            temp_df["max_diff"] = temp_df.max(axis=1) - temp_df.min(axis=1)
            fig = px.histogram(
                temp_df, x="max_diff", nbins=30,
                color_discrete_sequence=["#667eea"],
                title="Max Temperature Difference Between Stations",
                labels={"max_diff": f"{display_name('temperature_c')} Difference"},
                template="plotly_white",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True, key="sp_div")

            shield_days = (temp_df["max_diff"] > 5).sum()
            total = len(temp_df)
            st.success(
                f"🛡️ **{shield_days}** out of **{total}** days "
                f"({shield_days / total * 100:.1f}%) show >5°C difference — "
                f"viable Weather Shield opportunities."
            )


# ══════════════════════════════════════════════════════════════
#  TAB 6 — NLP INSIGHTS
# ══════════════════════════════════════════════════════════════

def tab_nlp(survey):
    st.header("📝  NLP Survey Analysis")
    st.markdown(
        "Natural Language Processing of tourism survey free-text responses — "
        "extracting sentiment, topics, and keywords from visitor feedback."
    )

    engine = get_nlp_engine()

    # Find text column
    text_col = None

    if "free_text" in survey.columns:
        non_empty = survey["free_text"].dropna()
        non_empty = non_empty[non_empty.astype(str).str.len() > 3]
        if len(non_empty) > 0:
            text_col = "free_text"

    if text_col is None:
        for c in ["satisfaction_reason", "inconvenience_detail",
                   "recommendation_reason", "満足度の理由", "不便さの内容", "推奨項目"]:
            if c in survey.columns:
                if survey[c].dropna().shape[0] > 10:
                    text_col = c
                    break

    if text_col is None:
        best_col, best_len = None, 0
        for c in survey.select_dtypes(include=["object"]).columns:
            avg_len = survey[c].dropna().astype(str).str.len().mean()
            if avg_len > 15 and avg_len > best_len:
                best_col, best_len = c, avg_len
        if best_col:
            text_col = best_col

    if text_col is None:
        string_cols = list(survey.select_dtypes(include=["object"]).columns)
        if string_cols:
            text_col = st.selectbox(
                "Select text column:", string_cols, key="nlp_col"
            )
        else:
            st.error("No text columns found.")
            return

    texts = survey[text_col].dropna().astype(str)
    texts = texts[texts.str.len() > 3]

    if len(texts) == 0:
        st.error(f"Column `{text_col}` has no usable text.")
        return

    st.success(f"📄 Analysing **{len(texts):,}** responses from `{text_col}`")

    with st.spinner("🔄 Running NLP pipeline..."):
        results = engine.full_analysis(texts)

    st.success(f"✅ {results['doc_count']} documents processed")

    # ── Sentiment ──
    st.subheader("😊😐😠  Sentiment Distribution")
    sent_df = results["sentiment"]

    col1, col2 = st.columns(2)
    with col1:
        counts = sent_df["label"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={
                "Positive": "#10b981", "Neutral": "#6b7280", "Negative": "#ef4444"
            },
            title="Overall Sentiment", template="plotly_white",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True, key="nlp_pie")
    with col2:
        fig = px.histogram(
            sent_df, x="polarity", nbins=40, color="label",
            color_discrete_map={
                "Positive": "#10b981", "Neutral": "#6b7280", "Negative": "#ef4444"
            },
            title="Polarity Distribution", template="plotly_white",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True, key="nlp_polar")

    if "prefecture" in survey.columns:
        merged = survey.iloc[:len(sent_df)].copy()
        merged["sentiment"] = sent_df["label"].values[:len(merged)]
        fig = px.histogram(
            merged.dropna(subset=["sentiment"]),
            x="prefecture", color="sentiment", barmode="group",
            color_discrete_map={
                "Positive": "#10b981", "Neutral": "#6b7280", "Negative": "#ef4444"
            },
            title="Sentiment by Prefecture", template="plotly_white",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="nlp_pref")

    st.markdown("---")

    # ── Keywords ──
    st.subheader("🔑  Top Keywords (TF-IDF)")
    kw = results["keywords"]
    if kw:
        kw_df = pd.DataFrame(kw, columns=["Keyword", "TF-IDF Score"])
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                kw_df.head(20).iloc[::-1], x="TF-IDF Score", y="Keyword",
                orientation="h", color="TF-IDF Score",
                color_continuous_scale="Viridis", template="plotly_white",
                title="Top 20 Keywords",
            )
            fig.update_layout(height=550, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="nlp_kw")
        with col2:
            st.dataframe(kw_df, height=550, use_container_width=True)

    st.markdown("---")

    # ── Word Cloud ──
    st.subheader("☁️  Word Cloud")
    wc_data = results["wordcloud"]
    if wc_data:
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            wc = WordCloud(
                width=1000, height=500, background_color="white",
                max_words=100, colormap="viridis",
            )
            wc.generate_from_frequencies(wc_data)
            fig_wc, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc, key="nlp_wc")
        except ImportError:
            wc_df = pd.DataFrame(list(wc_data.items()), columns=["Word", "Freq"])
            st.dataframe(wc_df.sort_values("Freq", ascending=False).head(30))

    st.markdown("---")

    # ── Topics ──
    st.subheader("📚  Topic Modelling (LDA)")
    topic_data = results["topics"]
    if topic_data["topics"]:
        n_top = len(topic_data["topics"])
        cols = st.columns(min(n_top, 3))
        for i, topic in enumerate(topic_data["topics"]):
            with cols[i % min(n_top, 3)]:
                st.markdown(f"**{topic['label']}**")
                for w in topic["words"]:
                    st.markdown(f"  • {w}")

        if topic_data["doc_topic"].size > 0:
            dominant = topic_data["doc_topic"].argmax(axis=1)
            tc = pd.Series(dominant).value_counts().sort_index()
            labels = [f"Topic {i + 1}" for i in tc.index]
            fig = px.bar(
                x=labels, y=tc.values, color=labels,
                template="plotly_white", title="Documents per Topic",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="nlp_topics")

    st.markdown("---")

    # ── N-grams ──
    st.subheader("🔤  N-gram Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bigrams**")
        bg = results["bigrams"]
        if bg:
            bg_df = pd.DataFrame(bg, columns=["Bigram", "Count"])
            fig = px.bar(
                bg_df.head(15).iloc[::-1], x="Count", y="Bigram",
                orientation="h", color_discrete_sequence=["#667eea"],
                template="plotly_white",
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="nlp_bg")
    with col2:
        st.markdown("**Trigrams**")
        tg = results["trigrams"]
        if tg:
            tg_df = pd.DataFrame(tg, columns=["Trigram", "Count"])
            fig = px.bar(
                tg_df.head(15).iloc[::-1], x="Count", y="Trigram",
                orientation="h", color_discrete_sequence=["#764ba2"],
                template="plotly_white",
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="nlp_tg")

    st.markdown("---")

    # ── Sample Reviews ──
    st.subheader("🔍  Sample Reviews")
    n_samp = st.slider("Number of samples", 5, 50, 10, key="nlp_samp")
    sample = sent_df.sample(min(n_samp, len(sent_df)), random_state=42)
    for _, row in sample.iterrows():
        emoji = "😊" if row["label"] == "Positive" else "😠" if row["label"] == "Negative" else "😐"
        color = "#10b981" if row["label"] == "Positive" else "#ef4444" if row["label"] == "Negative" else "#6b7280"
        st.markdown(
            f'{emoji} <span style="color:{color}">**[{row["label"]}]** '
            f'({row["polarity"]:+.3f})</span> — {str(row["text"])[:200]}',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    try:
        jma, survey = load_all_data()
        station, date_range = render_sidebar(jma)

        st.title("🏯 Hokuriku Tourism AI Governance Dashboard")
        st.caption(
            "Interactive exploration of the Distributed Human Data Engine (DHDE) — "
            "University of Fukui · Prof. Amil Khanzada"
        )

        tabs = st.tabs([
            "📊 Overview",
            "🌤️ Weather & Tourism",
            "💰 Opportunity Gap",
            "🌡️ Kansei Comfort",
            "🗺️ Spatial Network",
            "📝 NLP Insights",
        ])

        with tabs[0]:
            try:
                tab_overview(jma, survey)
            except Exception as e:
                st.error(f"Overview error: {e}")

        with tabs[1]:
            try:
                tab_weather(jma, station, date_range)
            except Exception as e:
                st.error(f"Weather error: {e}")

        with tabs[2]:
            try:
                tab_opportunity_gap(jma, station)
            except Exception as e:
                st.error(f"Opportunity Gap error: {e}")

        with tabs[3]:
            try:
                tab_kansei(jma, station)
            except Exception as e:
                st.error(f"Kansei error: {e}")

        with tabs[4]:
            try:
                tab_spatial(jma)
            except Exception as e:
                st.error(f"Spatial error: {e}")

        with tabs[5]:
            try:
                tab_nlp(survey)
            except Exception as e:
                st.error(f"NLP error: {e}")

    except Exception as e:
        st.error(f"Fatal error: {e}")
        import traceback
        st.code(traceback.format_exc())


main()