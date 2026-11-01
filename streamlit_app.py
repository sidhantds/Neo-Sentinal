"""
NEO Sentinel: Streamlit app.

Runs the trained pipeline directly, so it deploys on its own to Streamlit
Community Cloud with no separate API server. Uses the same validation and
risk tiers as the FastAPI service in app/main.py.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.schemas import AsteroidFeatures  # noqa: E402
from preprocess import FEATURE_COLUMNS, risk_tier  # noqa: E402

MODELS_DIR = ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
SHAP_PATH = MODELS_DIR / "shap_summary.png"

FEATURE_LABELS = {
    "est_diameter_min": "Estimated min diameter (km)",
    "est_diameter_max": "Estimated max diameter (km)",
    "relative_velocity": "Relative velocity (km/h)",
    "miss_distance": "Miss distance (km)",
    "absolute_magnitude": "Absolute magnitude (H)",
}

PRESETS = {
    "Small, distant rock": dict(
        est_diameter_min=0.02, est_diameter_max=0.05, relative_velocity=30000.0,
        miss_distance=40_000_000.0, absolute_magnitude=25.5,
    ),
    "Typical flyby": dict(
        est_diameter_min=0.1, est_diameter_max=0.25, relative_velocity=45000.0,
        miss_distance=5_000_000.0, absolute_magnitude=21.5,
    ),
    "Large, close and fast": dict(
        est_diameter_min=0.6, est_diameter_max=1.4, relative_velocity=95000.0,
        miss_distance=900_000.0, absolute_magnitude=18.0,
    ),
}

SHORT_LABELS = {
    "est_diameter_min": "Min diameter",
    "est_diameter_max": "Max diameter",
    "relative_velocity": "Velocity",
    "miss_distance": "Miss distance",
    "absolute_magnitude": "Brightness (H)",
}

TIER_STYLE = {
    "low": ("🟢", "success"),
    "moderate": ("🟡", "info"),
    "elevated": ("🟠", "warning"),
    "high": ("🔴", "error"),
}

st.set_page_config(page_title="NEO Sentinel", page_icon="☄️", layout="wide")


@st.cache_resource
def load_model():
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    proba = bundle["pipeline"].predict_proba(frame[bundle["feature_columns"]])[:, 1]
    out = frame.copy()
    out["hazard_probability"] = proba.round(4)
    out["hazardous"] = proba >= 0.5
    out["risk_tier"] = [risk_tier(p) for p in proba]
    return out


bundle = load_model()
metrics = load_json(str(METRICS_PATH)) or {}
importance = load_json(str(IMPORTANCE_PATH))

# ---------- Sidebar ----------
with st.sidebar:
    st.header("☄️ NEO Sentinel")
    st.write("Asteroid hazard screening, trained on NASA's Near Earth Object catalog.")
    if bundle:
        st.success(f"Model loaded: **{bundle['model_name']}**")
        test = metrics.get("test_metrics", {})
        if "f1" in test:
            st.subheader("Held out test set")
            c1, c2 = st.columns(2)
            c1.metric("F1", f"{test['f1']:.3f}")
            c2.metric("Recall", f"{test['recall']:.3f}")
            c1.metric("Precision", f"{test['precision']:.3f}")
            c2.metric("ROC AUC", f"{test['roc_auc']:.3f}")
            st.caption("Measured once on data the model never saw during tuning.")
    else:
        st.error("No trained model found.")
    st.divider()
    st.markdown(
        "[Source code](https://github.com/sidhantds/Neo-Sentinal) · "
        "[Sidhant Pujara](https://linkedin.com/in/sidhant-pujara)"
    )

st.title("NEO Sentinel")
st.caption("Should this asteroid get a closer look? A fast first pass screen for near Earth objects.")

if bundle is None:
    st.warning(
        "The app has no trained model yet. Run `python src/train.py --data data/neo.csv`, "
        "then commit the files in `models/` (see the README) and redeploy."
    )
    st.stop()

tab_single, tab_batch, tab_model = st.tabs(["Check one object", "Screen a CSV", "How the model decides"])

# ---------- Single prediction ----------
with tab_single:
    preset = st.radio("Start from an example", list(PRESETS), index=1, horizontal=True)
    values = PRESETS[preset]

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        dmin = c1.number_input(FEATURE_LABELS["est_diameter_min"], min_value=0.0, value=values["est_diameter_min"], format="%.4f")
        dmax = c1.number_input(FEATURE_LABELS["est_diameter_max"], min_value=0.0, value=values["est_diameter_max"], format="%.4f")
        vel = c2.number_input(FEATURE_LABELS["relative_velocity"], min_value=0.0, value=values["relative_velocity"], step=1000.0)
        miss = c2.number_input(FEATURE_LABELS["miss_distance"], min_value=0.0, value=values["miss_distance"], step=100000.0)
        mag = c3.number_input(FEATURE_LABELS["absolute_magnitude"], value=values["absolute_magnitude"], step=0.1)
        submitted = st.form_submit_button("Assess hazard risk", type="primary")

    if submitted:
        try:
            features = AsteroidFeatures(
                est_diameter_min=dmin, est_diameter_max=dmax, relative_velocity=vel,
                miss_distance=miss, absolute_magnitude=mag,
            )
        except ValidationError as exc:
            for err in exc.errors():
                st.error(err["msg"].replace("Value error, ", ""))
        else:
            row = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
            result = predict_frame(bundle, row).iloc[0]
            tier = result["risk_tier"]
            icon, style = TIER_STYLE[tier]

            m1, m2, m3 = st.columns(3)
            m1.metric("Hazard probability", f"{result['hazard_probability']:.1%}")
            m2.metric("Risk tier", f"{icon} {tier.title()}")
            m3.metric("Classification", "Hazardous" if result["hazardous"] else "Not hazardous")
            st.progress(float(result["hazard_probability"]))
            getattr(st, style)(
                {
                    "low": "Low priority. No closer review suggested.",
                    "moderate": "Some risk signals. Worth a quick look if time allows.",
                    "elevated": "Strong risk signals. Recommend closer orbital review.",
                    "high": "Very likely hazardous. Prioritise for full orbital analysis.",
                }[tier]
            )

# ---------- Batch screening ----------
with tab_batch:
    st.write(
        "Upload a CSV with these columns to score many objects at once: "
        + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS)
        + ". Extra columns such as `name` are kept."
    )
    template = pd.DataFrame([PRESETS[p] for p in PRESETS], columns=FEATURE_COLUMNS)
    template.insert(0, "name", list(PRESETS))
    st.download_button("Download a template CSV", template.to_csv(index=False), "neo_template.csv", "text/csv")

    upload = st.file_uploader("Upload CSV", type="csv")
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            st.stop()

        missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            numeric = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
            bad = (
                numeric.isna().any(axis=1)
                | (numeric[["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance"]] <= 0).any(axis=1)
                | (numeric["est_diameter_min"] > numeric["est_diameter_max"])
            )
            if bad.any():
                st.warning(f"Skipped {int(bad.sum())} rows with missing, non positive or inconsistent values.")
            clean_rows = data.loc[~bad].copy()
            clean_rows[FEATURE_COLUMNS] = numeric.loc[~bad]

            if clean_rows.empty:
                st.error("No valid rows to score.")
            else:
                scored = predict_frame(bundle, clean_rows).sort_values("hazard_probability", ascending=False)
                counts = scored["risk_tier"].value_counts()
                cols = st.columns(4)
                for col, tier in zip(cols, ["high", "elevated", "moderate", "low"]):
                    col.metric(f"{TIER_STYLE[tier][0]} {tier.title()}", int(counts.get(tier, 0)))
                st.dataframe(scored, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results", scored.to_csv(index=False), "neo_sentinel_results.csv", "text/csv"
                )

# ---------- Model explanation ----------
with tab_model:
    left, right = st.columns(2)
    with left:
        st.subheader("Feature importance")
        if importance:
            imp = pd.DataFrame(
                {"feature": [SHORT_LABELS.get(k, k) for k in importance], "importance": list(importance.values())}
            ).set_index("feature")
            st.bar_chart(imp, horizontal=True, color="#FF6B35")
        else:
            st.info("Feature importance appears here after training.")
    with right:
        st.subheader("SHAP summary")
        if SHAP_PATH.exists():
            st.image(str(SHAP_PATH), caption="How each feature pushes a prediction toward hazardous or safe.")
        else:
            st.info("The SHAP plot appears here when it is generated during training.")

    if metrics.get("validation_metrics"):
        st.subheader("Model comparison (validation set)")
        rows = [
            {"model": name, **{k: v for k, v in m.items() if k != "confusion_matrix"}}
            for name, m in metrics["validation_metrics"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Selected: **{metrics.get('selected_model')}**, {metrics.get('selection_rule', 'highest validation F1')}.")

    st.info(
        "Scope: this is a screening aid, not a replacement for NASA's orbital hazard assessment. "
        "It sees five physical and orbital features and does not model trajectory uncertainty."
    )

st.divider()
st.caption("Data: NASA Nearest Earth Objects (NeoWs), via Kaggle. NASA data is public domain.")
