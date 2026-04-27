# Workout Recovery Advisor

A Streamlit web app that predicts your perceived exertion (RPE, 1–10 scale) for a workout described in plain English, then generates a personalized recovery recommendation. The pipeline combines a tuned XGBoost regressor (trained on the public PMData dataset, 772 sessions across 15 participants) with two Anthropic API calls — one to parse the free-text input into structured features, and one to generate a context-aware recovery recommendation.

## Files

- `app.py` — Streamlit UI and prediction pipeline
- `llm_components.py` — `parse_workout()`, `generate_recommendation()`, `to_feature_row()`
- `rpe_xgb_model.joblib` — trained XGBoost model (CV RMSE ≈ 1.59, R² ≈ 0.06)
- `rpe_model_meta.json` — feature column order, median fallbacks, CV metrics
- `requirements.txt` — pinned dependencies

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: set ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Push this folder (including the `.joblib` and `.json` model files) to a public GitHub repo.
2. Go to <https://share.streamlit.io>, click **New app**, point it at the repo, set the main file to `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Click **Deploy**. First build takes ~3 minutes.

The app reads the API key from `st.secrets` if available, otherwise falls back to the `ANTHROPIC_API_KEY` environment variable.

## How it works

1. User sets a personal-baseline slider (1–10: how hard a typical moderate workout feels for them) — this calibrates the prediction to how they personally use the scale.
2. User types a free-text workout description.
3. The Anthropic API parses the text into structured fields (duration, activity tags, optional wellness ratings).
4. Parsed fields + slider value form a 19-element feature row; missing fields are filled with training-set medians.
5. The XGBoost model predicts RPE; the app shows it ± the model's typical error.
6. A second API call generates a 2–3 sentence recovery recommendation referencing the user's specific inputs.

## Honest model notes

The model's R² under group-level cross-validation (held-out participants) is ~0.06 — small but real. The personal-baseline slider does most of the predictive work; workout duration, activity type, and wellness fields contribute small additional signal. Without the slider, the model can't beat predicting the global mean. This is a fundamental property of perceived exertion: it is highly individual, and the dataset has no labels for new users to bootstrap from.

The recommendation text is AI-generated and is not medical advice.
