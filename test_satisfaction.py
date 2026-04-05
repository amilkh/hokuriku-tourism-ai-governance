import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_bridge import DataBridge

st.title("Debug Satisfaction")

bridge = DataBridge()
survey = bridge.load_survey_data()

st.write(f"Total rows: {len(survey)}")
st.write(f"All columns: {list(survey.columns)}")

# Check every column that might be satisfaction
for col in survey.columns:
    if any(k in col.lower() for k in ["満足", "satisfaction", "score", "rating", "評価"]):
        st.markdown(f"### Column: `{col}`")
        st.write(f"Type: {survey[col].dtype}")
        st.write(f"Non-null: {survey[col].notna().sum()}")
        st.write(f"Unique values ({survey[col].nunique()}):")
        st.write(survey[col].value_counts().head(20))
        st.write(f"Sample values: {survey[col].dropna().head(10).tolist()}")
        st.markdown("---")