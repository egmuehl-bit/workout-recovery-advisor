"""
app.py — Workout Recovery Advisor
Run locally with:  streamlit run app.py
Requires env var: ANTHROPIC_API_KEY  (or set via Streamlit secrets)
"""
from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import streamlit as st

# Pull API key from Streamlit secrets if present, before importing llm_components
# (llm_components reads ANTHROPIC_API_KEY from env at import time)
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    # No secrets file — assume env var set by user
    pass

from llm_components import (  # noqa: E402  (import after env setup is intentional)
    parse_workout,
    generate_recommendation,
    to_feature_row,
)


# --------------------------------------------------------------------------
# Resource loading (cached once per session)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model_and_meta() -> tuple[Any, dict]:
    model = joblib.load("rpe_xgb_model.joblib")
    with open("rpe_model_meta.json") as f:
        meta = json.load(f)
    return model, meta


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Workout Recovery Advisor",
    page_icon="💪",
    layout="centered",
)

st.title("Workout Recovery Advisor")
st.caption(
    "Describe your workout in plain English. The model predicts your perceived "
    "exertion (1–10) and an AI advisor suggests recovery actions."
)

# Surface API-key issues up front rather than at submit time
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY is not set. On Streamlit Community Cloud, add it under "
        "**Settings → Secrets**. Locally, export it before running the app."
    )
    st.stop()

# Load model
try:
    model, meta = load_model_and_meta()
except Exception as e:
    st.error(f"Failed to load the model files: {e}")
    st.stop()


# --------------------------------------------------------------------------
# Input UI
# --------------------------------------------------------------------------

personal_baseline = st.slider(
    "On a 1–10 scale, how hard does a typical moderate workout feel for you?",
    min_value=1, max_value=10, value=5,
    help="This calibrates the prediction to how you personally use the 1–10 scale.",
)

workout_text = st.text_area(
    "Describe your workout",
    placeholder=(
        "e.g. ran 45 minutes this morning, slept badly last night, felt tired going in"
    ),
    height=120,
)

submitted = st.button("Get My Recovery Prediction", type="primary")


# --------------------------------------------------------------------------
# Submit handler
# --------------------------------------------------------------------------

def predict_rpe(parsed: dict, baseline: int) -> float:
    row = to_feature_row(
        parsed=parsed,
        personal_baseline=baseline,
        defaults=meta["feature_defaults"],
        feature_cols=meta["feature_cols"],
    )
    return float(model.predict(np.array([row]))[0])


def difficulty_label(predicted: float, baseline: float) -> str:
    diff = predicted - baseline
    if diff >= 1.0:
        return "🔴 Harder than usual"
    if diff <= -1.0:
        return "🟢 Easier than usual"
    return "🟡 About average for you"


if submitted:
    if not workout_text.strip():
        st.warning("Please describe your workout first.")
        st.stop()

    rmse = meta["cv_metrics"]["xgb_rmse_mean"]

    with st.spinner("Parsing your workout..."):
        try:
            parsed = parse_workout(workout_text)
        except Exception as e:
            st.error(f"Couldn't parse the workout description ({type(e).__name__}). Try rephrasing.")
            st.stop()

    # If user mentioned their own baseline in the text, prefer that over the slider
    baseline_used = parsed.get("personal_baseline") or personal_baseline

    with st.spinner("Predicting recovery..."):
        try:
            pred = predict_rpe(parsed, baseline_used)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

    # Clamp display to 1–10 so we don't show "11.2" or "0.3"
    pred_display = float(np.clip(pred, 1.0, 10.0))

    # Headline metric
    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric(
            label="Predicted Perceived Exertion",
            value=f"{pred_display:.1f} / 10",
            delta=f"±{rmse:.1f} (typical error)",
            delta_color="off",
        )
    with col2:
        st.markdown(f"### {difficulty_label(pred_display, baseline_used)}")
        st.caption(f"Compared to your baseline of {baseline_used}/10")

    # Recommendation
    with st.spinner("Generating recommendation..."):
        try:
            rec = generate_recommendation(
                parsed=parsed,
                predicted_rpe=pred_display,
                personal_baseline=baseline_used,
            )
        except Exception as e:
            rec = (
                f"Predicted exertion is {pred_display:.1f}/10. "
                "Hydrate, eat a balanced meal with protein, and aim for 7–9 hours of sleep tonight.\n\n"
                "_Recommendation generator unavailable — showing a generic message._"
            )

    st.markdown("### Recovery recommendation")
    st.markdown(rec)

    # Transparency: show what the AI extracted
    with st.expander("What did the AI parse from your description?"):
        # Friendlier display: split into activity, duration, wellness, baseline
        activity_flags = {k: v for k, v in parsed.items() if k.startswith("is_")}
        wellness = {
            k: parsed.get(k)
            for k in ["fatigue", "mood", "sleep_quality", "soreness", "stress"]
        }
        st.markdown("**Activity flags** (1 = detected):")
        st.json(activity_flags)
        st.markdown(f"**Duration:** {parsed.get('duration_min') or 'not mentioned'}")
        st.markdown("**Wellness fields** (1–5; null = not mentioned, model uses median):")
        st.json(wellness)
        st.markdown(
            f"**Personal baseline used:** {baseline_used}/10 "
            f"({'extracted from text' if parsed.get('personal_baseline') else 'from slider'})"
        )

    # Honest model-quality note
    st.caption(
        f"Model: XGBoost, cross-validated RMSE ≈ {rmse:.2f} on the 1–10 RPE scale. "
        "Predicted exertion is highly individual; treat the number as a guide, not a verdict."
    )
