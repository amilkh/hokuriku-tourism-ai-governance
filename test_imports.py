import streamlit as st
import sys
from pathlib import Path

st.title("Import Test")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
st.write(f"Project root: {PROJECT_ROOT}")

# Test each import one by one
try:
    import pandas as pd
    st.success("✅ pandas loaded")
except Exception as e:
    st.error(f"❌ pandas: {e}")

try:
    import plotly.express as px
    st.success("✅ plotly loaded")
except Exception as e:
    st.error(f"❌ plotly: {e}")

try:
    from dashboard.data_bridge import DataBridge, discomfort_index
    st.success("✅ data_bridge loaded")
except Exception as e:
    st.error(f"❌ data_bridge: {e}")

try:
    from dashboard.nlp_engine import TourismNLPEngine
    st.success("✅ nlp_engine loaded")
except Exception as e:
    st.error(f"❌ nlp_engine: {e}")

# Test data loading
try:
    bridge = DataBridge()
    jma = bridge.load_jma_data()
    st.success(f"✅ JMA loaded: {len(jma)} rows")
    st.write(f"Stations: {jma['station'].unique().tolist()}")
    st.write(f"Columns: {list(jma.columns)}")
except Exception as e:
    st.error(f"❌ JMA loading: {e}")

try:
    survey = bridge.load_survey_data()
    st.success(f"✅ Survey loaded: {len(survey)} rows")
    st.write(f"Columns: {list(survey.columns)[:15]}")
    st.write(f"Has free_text: {'free_text' in survey.columns}")
except Exception as e:
    st.error(f"❌ Survey loading: {e}")

st.write("--- DEBUG COMPLETE ---")