"""
Optional Streamlit demo UI for NEO Sentinel.

This is a thin client over the deployed FastAPI service. It is not required
for the core deliverable; add it only after /health, /predict, and /docs
are confirmed working.

Run locally:
    streamlit run app/streamlit_app.py

Set the API_URL environment variable to point at your deployed API, e.g.:
    API_URL=https://neo-sentinel.onrender.com streamlit run app/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NEO Sentinel", page_icon="☄️")
st.title("NEO Sentinel: Asteroid Hazard Screening")
st.caption(f"Connected to API: {API_URL}")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        est_diameter_min = st.number_input("Estimated min diameter (km)", min_value=0.0, value=0.1, format="%.4f")
        relative_velocity = st.number_input("Relative velocity (km/h)", min_value=0.0, value=45000.0)
        absolute_magnitude = st.number_input("Absolute magnitude (H)", value=21.5)
    with col2:
        est_diameter_max = st.number_input("Estimated max diameter (km)", min_value=0.0, value=0.25, format="%.4f")
        miss_distance = st.number_input("Miss distance (km)", min_value=0.0, value=5000000.0)

    submitted = st.form_submit_button("Assess hazard risk")

if submitted:
    payload = {
        "est_diameter_min": est_diameter_min,
        "est_diameter_max": est_diameter_max,
        "relative_velocity": relative_velocity,
        "miss_distance": miss_distance,
        "absolute_magnitude": absolute_magnitude,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result["hazardous"]:
                st.error(f"Hazardous (probability: {result['hazard_probability']:.2%})")
            else:
                st.success(f"Not hazardous (probability: {result['hazard_probability']:.2%})")
            st.write(f"Risk tier: **{result['risk_tier']}**")
            st.write(f"Model: {result['model_name']}")
        else:
            st.warning(f"API returned {response.status_code}: {response.json().get('detail')}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")

st.divider()
st.caption("Source dataset: NASA Nearest Earth Objects (NeoWs), via data.gov and the NASA Open API.")
