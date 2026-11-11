"""NEO Sentinel: interactive app.

Loads the model bundle directly (no API server needed), so it deploys on its
own to Streamlit Community Cloud. Scores through the same ModelBundle.score()
as the API.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from neo_sentinel.api.schemas import CloseApproach
from neo_sentinel.bundle import TIER_ADVICE, TIERS, ModelBundle
from neo_sentinel.config import FEATURE_LABELS, FEATURES, MODELS_DIR, model_path
from neo_sentinel.data import clean_features, parse_label
from neo_sentinel.monitoring import out_of_range

st.set_page_config(page_title="NEO Sentinel", page_icon="☄️", layout="wide")

TIER_ICON = {"low": "🟢", "moderate": "🟡", "elevated": "🟠", "high": "🔴"}
TIER_BOX = {"low": st.success, "moderate": st.info, "elevated": st.warning, "high": st.error}
PRESETS = {
    "Small, typical flyby": {"absolute_magnitude": 25.0, "relative_velocity": 45_000.0, "miss_distance": 30_000_000.0},
    "Big, distant pass": {"absolute_magnitude": 19.0, "relative_velocity": 60_000.0, "miss_distance": 65_000_000.0},
    "Big, middle distance": {"absolute_magnitude": 20.0, "relative_velocity": 60_000.0, "miss_distance": 18_000_000.0},
    "Big and close": {"absolute_magnitude": 19.0, "relative_velocity": 80_000.0, "miss_distance": 3_000_000.0},
}
MODEL_LABELS = {
    "brightness_rule": "H ≤ 22 rule (baseline)",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
}


@st.cache_resource
def load_bundle():
    path = model_path()
    return ModelBundle.load(path) if path.exists() else None


@st.cache_data(ttl=3600, show_spinner="Asking NASA for close approaches...")
def live_feed(start: date):
    from neo_sentinel.neows import fetch_feed

    key = os.environ.get("NASA_API_KEY")
    try:
        key = key or st.secrets.get("NASA_API_KEY")
    except Exception:
        pass
    return fetch_feed(start, key=key)


def pct(x: float) -> str:
    return f"{x:.1%}"


bundle = load_bundle()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("☄️ NEO Sentinel")
    st.write("Hazard screening for asteroid close approaches, trained on NASA's NeoWs catalog.")
    if bundle:
        test = bundle.card.get("test", {})
        rule = bundle.card.get("test_baseline_brightness_rule", {})
        st.success(f"Model **{bundle.version}**")
        st.caption("Held out test set: asteroids the model never saw.")
        c1, c2 = st.columns(2)
        c1.metric("Recall", pct(test.get("recall", 0)))
        c2.metric(
            "Flags",
            pct(test.get("flagged_share", 0)),
            delta=f"{(test.get('flagged_share', 0) - rule.get('flagged_share', 0)) * 100:+.1f} pts vs rule",
            delta_color="inverse",
        )
        c1.metric("Avg precision", f"{test.get('average_precision', 0):.3f}")
        c2.metric("Rule's AP", f"{rule.get('average_precision', 0):.3f}")
    st.divider()
    st.markdown("[Source code](https://github.com/sidhantds/Neo-Sentinal)")

st.title("NEO Sentinel")
st.caption("Which asteroid close approaches deserve a closer look? A fast first pass, with honest limits.")

if bundle is None:
    st.warning("No trained model found. Run `python -m neo_sentinel download` then `python -m neo_sentinel train`.")
    st.stop()

tab_one, tab_csv, tab_live, tab_model = st.tabs(
    ["Check one object", "Screen a CSV", "This week's approaches", "Model report"]
)

# ------------------------------------------------------------------ one object
with tab_one:
    preset = st.radio("Start from an example", list(PRESETS), index=1, horizontal=True)
    values = PRESETS[preset]
    with st.form("one"):
        c1, c2, c3 = st.columns(3)
        h = c1.number_input(
            FEATURE_LABELS["absolute_magnitude"],
            value=values["absolute_magnitude"],
            step=0.1,
            help="Brightness at a standard distance. Lower H means a bigger object.",
        )
        v = c2.number_input(FEATURE_LABELS["relative_velocity"], value=values["relative_velocity"], step=1000.0)
        d = c3.number_input(
            FEATURE_LABELS["miss_distance"],
            value=values["miss_distance"],
            step=100_000.0,
            help="For scale: the Moon is about 384,400 km away.",
        )
        submitted = st.form_submit_button("Assess", type="primary")

    if submitted:
        try:
            approach = CloseApproach(absolute_magnitude=h, relative_velocity=v, miss_distance=d)
        except ValidationError as exc:
            for err in exc.errors():
                st.error(f"{err['loc'][0]}: {err['msg']}")
        else:
            record = approach.model_dump()
            row = bundle.score(pd.DataFrame([record])).iloc[0]
            tier = row["risk_tier"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hazard probability", pct(row["hazard_probability"]))
            m2.metric("Risk tier", f"{TIER_ICON[tier]} {tier.title()}")
            m3.metric("Review list", "Yes" if row["flagged"] else "No")
            m4.metric(
                "Estimated size", f"{row['est_diameter_min_km'] * 1000:,.0f}–{row['est_diameter_max_km'] * 1000:,.0f} m"
            )
            TIER_BOX[tier](TIER_ADVICE[tier])
            for w in out_of_range(record, bundle.reference):
                st.warning(w)
            st.caption(
                f"Review threshold {pct(bundle.threshold)}. The H ≤ 22 rule alone would "
                f"{'flag' if row['brightness_rule_flag'] else 'not flag'} this object. "
                f"{d / 384_400:,.1f} lunar distances."
            )

            st.subheader("What if it were bigger or smaller?")
            grid = np.linspace(14, 30, 161)
            sweep = pd.DataFrame({"absolute_magnitude": grid, "relative_velocity": v, "miss_distance": d})
            sweep["hazard_probability"] = bundle.predict_proba(sweep)
            line = (
                alt.Chart(sweep)
                .mark_line(color="#FF6B35")
                .encode(
                    x=alt.X(
                        "absolute_magnitude",
                        title="Absolute magnitude H (left is bigger)",
                        scale=alt.Scale(reverse=True),
                    ),
                    y=alt.Y("hazard_probability", title="Hazard probability", axis=alt.Axis(format="%")),
                )
            )
            rule_line = alt.Chart(pd.DataFrame({"y": [bundle.threshold]})).mark_rule(strokeDash=[4, 4]).encode(y="y")
            here = (
                alt.Chart(pd.DataFrame({"absolute_magnitude": [h], "hazard_probability": [row["hazard_probability"]]}))
                .mark_point(size=120, filled=True, color="white")
                .encode(x="absolute_magnitude", y="hazard_probability")
            )
            st.altair_chart(line + rule_line + here, use_container_width=True)
            st.caption(
                "Same speed and miss distance, only H changes. Dashed line: review threshold. "
                "The jump near H = 22 is the model rediscovering NASA's brightness cut."
            )

# ------------------------------------------------------------------ CSV
with tab_csv:
    st.write(
        "Upload a CSV with "
        + ", ".join(f"`{c}`" for c in FEATURES)
        + ". Other columns are kept. The original NASA dataset file works as is, and if it has a "
        "`hazardous` column you also see how the model did against NASA's labels."
    )
    template = pd.DataFrame(PRESETS.values())
    template.insert(0, "name", list(PRESETS))
    st.download_button("Download a template", template.to_csv(index=False), "neo_template.csv", "text/csv")

    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            valid, bad = clean_features(raw)
        except Exception as exc:
            st.error(f"Could not use that file: {exc}")
        else:
            if bad.any():
                st.warning(f"Skipped {int(bad.sum())} rows with missing or non physical values.")
            if valid.empty:
                st.error("No valid rows to score.")
            else:
                scored = bundle.score(valid).sort_values("hazard_probability", ascending=False)
                counts = scored["risk_tier"].value_counts()
                cols = st.columns(5)
                cols[0].metric("Scored", len(scored))
                for col, tier in zip(cols[1:], reversed(TIERS), strict=True):
                    col.metric(f"{TIER_ICON[tier]} {tier.title()}", int(counts.get(tier, 0)))

                if "hazardous" in scored.columns:
                    try:
                        truth = parse_label(scored["hazardous"]).to_numpy()
                        flagged = scored["flagged"].to_numpy()
                        caught = flagged[truth == 1].mean() if truth.sum() else float("nan")
                        st.info(
                            f"Against NASA's labels: caught {pct(caught)} of hazardous objects "
                            f"while flagging {pct(flagged.mean())} of rows."
                        )
                    except Exception:
                        pass

                st.dataframe(scored, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results", scored.to_csv(index=False), "neo_sentinel_results.csv", "text/csv"
                )

# ------------------------------------------------------------------ live
with tab_live:
    st.write(
        "Real close approaches from NASA's NeoWs feed, scored by the model, next to NASA's own "
        "hazard label. Uses `NASA_API_KEY` if set (free at api.nasa.gov), otherwise the rate limited DEMO_KEY."
    )
    start = st.date_input("Week starting", value=date.today(), min_value=date(1900, 1, 1))
    if st.button("Fetch from NASA", type="primary"):
        try:
            feed = live_feed(start)
        except Exception as exc:
            st.error(f"NASA's API did not answer ({exc}). Try again shortly or set NASA_API_KEY.")
        else:
            valid, _ = clean_features(feed) if not feed.empty else (feed, None)
            if valid.empty:
                st.info("No close approaches in that week.")
            else:
                scored = bundle.score(valid).sort_values("hazard_probability", ascending=False)
                agree = (scored["flagged"] | ~scored["nasa_hazardous"]).mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Approaches", len(scored))
                c2.metric("On review list", int(scored["flagged"].sum()))
                c3.metric(
                    "NASA PHAs caught",
                    f"{int((scored['flagged'] & scored['nasa_hazardous']).sum())}"
                    f" / {int(scored['nasa_hazardous'].sum())}",
                )
                show = [
                    "close_approach_date",
                    "name",
                    "hazard_probability",
                    "risk_tier",
                    "nasa_hazardous",
                    "absolute_magnitude",
                    "miss_distance",
                    "relative_velocity",
                    "nasa_url",
                ]
                st.dataframe(
                    scored[show],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "nasa_url": st.column_config.LinkColumn("JPL page"),
                        "hazard_probability": st.column_config.ProgressColumn(
                            "Hazard probability", min_value=0.0, max_value=1.0, format="%.2f"
                        ),
                    },
                )
                st.caption(
                    f"Window {start} to {start + timedelta(days=6)}. Every NASA PHA should be on the review list; "
                    f"agreement on that: {pct(agree)}."
                )

# ------------------------------------------------------------------ model report
with tab_model:
    card = bundle.card
    meta = card.get("metadata", {})
    st.subheader("What is running")
    st.write(
        f"**{MODEL_LABELS.get(bundle.model_name, bundle.model_name)}**, version `{bundle.version}`, trained "
        f"{meta.get('trained_at', '?')} with scikit-learn {meta.get('sklearn_version', '?')}. "
        f"Data checksum `{meta.get('data_sha256', '?')[:16]}…`."
    )

    ds = card.get("dataset", {})
    st.subheader("Three facts about the data that shaped everything")
    st.markdown(
        f"1. **{ds.get('rows', 0):,} rows but only {ds.get('unique_asteroids', 0):,} asteroids.** Each asteroid "
        "appears once per recorded pass, with the same label every time. Splitting rows at random would put the same "
        "asteroid in training and test, so every split here is made by asteroid.\n"
        "2. **The two diameter columns are brightness in disguise.** They match the standard H to diameter formula "
        f"to within {ds.get('diameter_max_relative_error_vs_formula', 0):.0e}, so they are not used as inputs.\n"
        "3. **NASA's label uses orbit distance (MOID), which the data lacks.** A model can learn the brightness "
        "half of the rule and use miss distance as a rough stand in for the other half. That caps any score."
    )
    leak_path = MODELS_DIR / "experiment_split_leakage.json"
    if leak_path.exists():
        leak = json.loads(leak_path.read_text())["results"]
        st.caption(
            f"Why fact 1 matters: with a random row split, "
            f"{leak['random_rows']['test_asteroids_also_in_training']:.0%} of test asteroids were also in training, "
            f"and average precision looked like {leak['random_rows']['average_precision']:.2f}. Split by asteroid, "
            f"the same model scores {leak['by_asteroid']['average_precision']:.2f}."
        )

    st.subheader("Every candidate, on validation")
    rows = [
        {
            "model": MODEL_LABELS.get(n, n),
            "avg precision": m["average_precision"],
            "ROC AUC": m["roc_auc"],
            "recall": m["recall"],
            "precision": m["precision"],
            "share flagged": m["flagged_share"],
        }
        for n, m in card.get("validation", {}).items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Selection: {card.get('selection_rule')}. Threshold: {card.get('threshold_rule')}.")

    left, right = st.columns(2)
    with left:
        st.subheader("Calibration (test)")
        cal = pd.DataFrame(card.get("calibration_test", []))
        if not cal.empty:
            pts = (
                alt.Chart(cal)
                .mark_circle(size=90, color="#FF6B35")
                .encode(
                    x=alt.X("mean_predicted", title="Predicted probability", axis=alt.Axis(format="%")),
                    y=alt.Y("observed_rate", title="Actually hazardous", axis=alt.Axis(format="%")),
                    tooltip=["mean_predicted", "observed_rate", "count"],
                )
            )
            diag = (
                alt.Chart(pd.DataFrame({"x": [0, 0.5], "y": [0, 0.5]}))
                .mark_line(strokeDash=[4, 4])
                .encode(x="x", y="y")
            )
            st.altair_chart(diag + pts, use_container_width=True)
            st.caption("Points on the dashed line mean a 30% score really is right about 30% of the time.")
    with right:
        st.subheader("What the model relies on (test)")
        imp = card.get("permutation_importance_test", {})
        if imp:
            df_imp = pd.DataFrame(
                [{"feature": FEATURE_LABELS.get(k, k), "drop in avg precision": v["mean"]} for k, v in imp.items()]
            )
            st.altair_chart(
                alt.Chart(df_imp)
                .mark_bar(color="#FF6B35")
                .encode(x="drop in avg precision", y=alt.Y("feature", sort="-x", title=None)),
                use_container_width=True,
            )
            st.caption("How much average precision falls when one feature is shuffled.")

    exp_path = MODELS_DIR / "experiment_history_features.json"
    if exp_path.exists():
        exp = json.loads(exp_path.read_text())["results"]
        a, b = exp["single_approach"]["test"], exp["with_history"]["test"]
        st.subheader("The clearest next step")
        st.write(
            "Adding each asteroid's approach history (its closest recorded pass, number of passes, fastest speed) "
            f"lifts test average precision from **{a['average_precision']:.2f} to {b['average_precision']:.2f}**. "
            "It is not shipped because it changes the input from one observation to a full history. "
            "Reproduce with `python -m neo_sentinel experiment`."
        )

    st.info(
        "Scope: a triage aid, not NASA's hazard assessment. "
        "It does not model orbits, uncertainty or future perturbations."
    )

st.divider()
st.caption("Data: NASA NeoWs via Kaggle (sameepvani/nasa-nearest-earth-objects). NASA data is public domain.")
