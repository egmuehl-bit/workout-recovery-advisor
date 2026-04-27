"""
llm_components.py — drop into the Streamlit app.

Two functions:
  parse_workout(text) -> dict        # free text -> structured fields
  generate_recommendation(...) -> str # prediction + context -> short recovery advice

Plus:
  to_feature_row(parsed, baseline, defaults, feature_cols) -> list[float]
    glues the parser output to the model's expected feature vector.

Requires:  pip install anthropic
Env var:   ANTHROPIC_API_KEY
"""
from __future__ import annotations
import os
import json
import re
from typing import Optional

import anthropic

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1000

# ---------------------------------------------------------------------------
# Part 1 — workout parser
# ---------------------------------------------------------------------------

PARSE_SYSTEM_PROMPT = """You extract structured data from free-text workout descriptions.

Output STRICT JSON matching this exact schema. No preamble, no markdown fences, no commentary:

{
  "duration_min": int or null,
  "is_running": 0 or 1,
  "is_soccer": 0 or 1,
  "is_endurance": 0 or 1,
  "is_strength": 0 or 1,
  "is_individual": 0 or 1,
  "is_team": 0 or 1,
  "fatigue": int 1-5 or null,
  "mood": int 1-5 or null,
  "sleep_quality": int 1-5 or null,
  "soreness": int 1-5 or null,
  "stress": int 1-5 or null,
  "personal_baseline": int 1-10 or null
}

Rules:

ACTIVITY FLAGS (always 0 or 1, never null):
- is_running: jogging, running, sprints, treadmill running, track work
- is_soccer: soccer or football (the sport with a ball)
- is_strength: weights, lifting, resistance training, bodyweight strength work
- is_endurance: sustained aerobic activity that is NOT running -- cycling, swimming, hiking, rowing, long aerobic sessions. Do NOT set is_endurance=1 just because a run is long; use is_running for that.
- is_individual vs is_team: exactly ONE must be 1, never both, never neither.
  * Default to is_individual=1 unless the description clearly indicates team practice or a team game.
  * Soccer practice/match -> is_team=1; solo soccer drills -> is_individual=1.
- If no activity is mentioned at all, set all activity flags to 0 except is_individual=1.

WELLNESS FIELDS (1-5 scale, null if not mentioned):
- fatigue: 1=very fresh, 2=rested, 3=normal, 4=tired, 5=exhausted
- mood: 1=poor, 2=down, 3=neutral, 4=good, 5=great
- sleep_quality: 1=very poor, 2=poor, 3=ok, 4=good, 5=excellent
- soreness: 1=none, 2=slight, 3=moderate, 4=quite sore, 5=very sore
- stress: 1=very calm, 2=relaxed, 3=neutral, 4=stressed, 5=very stressed

Word -> scale mapping examples:
- "slept badly / barely slept / terrible sleep" -> sleep_quality=1, fatigue=4-5
- "slept ok / decent sleep" -> sleep_quality=3
- "well rested / great sleep" -> sleep_quality=5, fatigue=2
- "wrecked / exhausted / dead" -> fatigue=5
- "sore from yesterday" -> soreness=3, "really sore" -> soreness=4-5
- "stressed about work" -> stress=4
- "feeling great" -> mood=4-5, fatigue=2
- Only set a wellness field if the user actually mentions that dimension. If they only say "felt heavy on bench" that's not enough to set fatigue.

DURATION:
- "45 min", "45 minutes", "for 45" -> duration_min=45
- "an hour" -> 60, "half hour" -> 30, "two hours" -> 120
- If only sets/reps mentioned with no time, leave duration_min as null

PERSONAL BASELINE:
- Only set if the user explicitly states their typical RPE or perceived exertion baseline (e.g., "my usual is around a 6", "typical workout feels like a 7"). Otherwise null. Most inputs will not have this.

GARBAGE/EMPTY INPUT:
- If the input is gibberish, blank, or clearly not a workout description, return:
  {"duration_min": null, "is_running": 0, "is_soccer": 0, "is_endurance": 0, "is_strength": 0, "is_individual": 1, "is_team": 0, "fatigue": null, "mood": null, "sleep_quality": null, "soreness": null, "stress": null, "personal_baseline": null}

Output JSON only. Begin with { and end with }."""


# All keys the parser should return
PARSED_KEYS = {
    "duration_min", "is_running", "is_soccer", "is_endurance",
    "is_strength", "is_individual", "is_team",
    "fatigue", "mood", "sleep_quality", "soreness", "stress",
    "personal_baseline",
}
ACTIVITY_FLAGS = {"is_running", "is_soccer", "is_endurance", "is_strength",
                  "is_individual", "is_team"}
WELLNESS_FIELDS_1_5 = {"fatigue", "mood", "sleep_quality", "soreness", "stress"}


def _safe_default_parsed() -> dict:
    """Returned when parsing fails entirely."""
    return {
        "duration_min": None,
        "is_running": 0, "is_soccer": 0, "is_endurance": 0,
        "is_strength": 0, "is_individual": 1, "is_team": 0,
        "fatigue": None, "mood": None, "sleep_quality": None,
        "soreness": None, "stress": None,
        "personal_baseline": None,
    }


def _validate_parsed(d: dict) -> dict:
    """Coerce values into expected types / ranges. Unrecoverable -> default."""
    out = _safe_default_parsed()
    for k in PARSED_KEYS:
        if k not in d:
            continue
        v = d[k]
        if v is None:
            out[k] = None if k not in ACTIVITY_FLAGS else 0
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue  # keep default
        if k in ACTIVITY_FLAGS:
            out[k] = 1 if iv == 1 else 0
        elif k == "duration_min":
            out[k] = iv if 1 <= iv <= 600 else None
        elif k in WELLNESS_FIELDS_1_5:
            out[k] = iv if 1 <= iv <= 5 else None
        elif k == "personal_baseline":
            out[k] = iv if 1 <= iv <= 10 else None

    # Enforce is_individual XOR is_team
    if out["is_individual"] == 0 and out["is_team"] == 0:
        out["is_individual"] = 1
    elif out["is_individual"] == 1 and out["is_team"] == 1:
        out["is_team"] = 0
    return out


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first {...} block out of the response, tolerant of surrounding fluff."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_workout(text: str) -> dict:
    """
    Parse a free-text workout description into structured fields.
    Always returns a dict matching PARSED_KEYS (with nulls for missing).
    Never raises on bad input; returns safe defaults instead.
    """
    if not text or not text.strip():
        return _safe_default_parsed()

    try:
        resp = CLIENT.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=PARSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text.strip()}],
        )
        body = resp.content[0].text if resp.content else ""
    except Exception as e:
        print(f"[parse_workout] API error: {e}")
        return _safe_default_parsed()

    parsed = _extract_json(body)
    if parsed is None:
        print(f"[parse_workout] JSON parse failed. Raw: {body[:200]}")
        return _safe_default_parsed()

    return _validate_parsed(parsed)


# ---------------------------------------------------------------------------
# Glue: parser output + slider baseline -> model feature row
# ---------------------------------------------------------------------------

def to_feature_row(
    parsed: dict,
    personal_baseline: float,
    defaults: dict,
    feature_cols: list,
) -> list:
    """
    Build the feature row the trained model expects.

    parsed: dict from parse_workout()
    personal_baseline: 1-10 float from Streamlit slider
    defaults: meta['feature_defaults'] from rpe_model_meta.json
    feature_cols: meta['feature_cols'] from rpe_model_meta.json (preserves order)
    """
    row_dict = dict(defaults)

    for k, v in parsed.items():
        if k in row_dict and v is not None:
            row_dict[k] = float(v)

    for flag in ACTIVITY_FLAGS:
        row_dict[flag] = float(parsed[flag])

    row_dict["personal_baseline"] = float(personal_baseline)

    sor = parsed.get("soreness")
    if sor is not None:
        row_dict["has_soreness_area"] = 1.0 if sor >= 3 else 0.0

    any_wellness = any(parsed.get(f) is not None for f in WELLNESS_FIELDS_1_5)
    row_dict["wellness_missing"] = 0.0 if any_wellness else 1.0

    return [float(row_dict[c]) for c in feature_cols]


# ---------------------------------------------------------------------------
# Part 2 — recommendation
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "Not medical advice — this is an estimate from a general fitness model, "
    "not a substitute for guidance from a coach or clinician."
)

RECOMMEND_SYSTEM_PROMPT = """You are a fitness recovery advisor. You will receive a predicted perceived exertion (1-10 scale) and a workout/state summary. Write a personalized recovery recommendation.

Rules:
- 2-3 sentences total, under 80 words.
- Reference at least one SPECIFIC factor from the user's input (e.g., poor sleep, soreness, long duration, strength session). Don't be generic.
- If their predicted exertion is meaningfully above their personal baseline, this was a hard session -> emphasize recovery. If below baseline, this was easier than usual -> recovery is lighter.
- Suggest concrete recovery actions: hydration, protein, sleep tonight, mobility/stretching, or a rest day. Pick what fits the inputs.
- Friendly but factual. No emoji. Don't use the term "RPE" -- say "perceived exertion" or just describe the intensity in plain words.
- Do NOT include any disclaimer; that's added separately.
- Output the recommendation text only. No headings, no bullets."""


def _format_context_for_recommendation(
    parsed: dict, predicted_rpe: float, personal_baseline: float
) -> str:
    """Human-readable summary the recommender LLM consumes."""
    lines = [
        f"Predicted perceived exertion: {predicted_rpe:.1f} / 10",
        f"User's personal baseline (typical RPE): {personal_baseline:.0f} / 10",
    ]

    activity_bits = []
    if parsed["is_running"]:
        activity_bits.append("running")
    if parsed["is_soccer"]:
        activity_bits.append("soccer")
    if parsed["is_strength"]:
        activity_bits.append("strength training")
    if parsed["is_endurance"]:
        activity_bits.append("endurance work")
    if not activity_bits:
        activity_bits.append("a workout")
    setting = "team" if parsed["is_team"] else "solo"
    lines.append(f"Activity: {setting} {' + '.join(activity_bits)}")

    if parsed.get("duration_min") is not None:
        lines.append(f"Duration: {parsed['duration_min']} minutes")

    scale_words_5 = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
    sleep_words = {1: "very poor", 2: "poor", 3: "ok", 4: "good", 5: "excellent"}
    mood_words = {1: "poor", 2: "down", 3: "neutral", 4: "good", 5: "great"}

    if parsed.get("fatigue") is not None:
        lines.append(f"Pre-workout fatigue: {scale_words_5[parsed['fatigue']]}")
    if parsed.get("mood") is not None:
        lines.append(f"Pre-workout mood: {mood_words[parsed['mood']]}")
    if parsed.get("sleep_quality") is not None:
        lines.append(f"Last night's sleep: {sleep_words[parsed['sleep_quality']]}")
    if parsed.get("soreness") is not None:
        lines.append(f"Pre-workout soreness: {scale_words_5[parsed['soreness']]}")
    if parsed.get("stress") is not None:
        lines.append(f"Pre-workout stress: {scale_words_5[parsed['stress']]}")

    return "\n".join(lines)


def generate_recommendation(
    parsed: dict,
    predicted_rpe: float,
    personal_baseline: float,
    include_disclaimer: bool = True,
) -> str:
    """
    Generate a 2-3 sentence personalized recovery recommendation.
    Falls back to a generic message if the API call fails.
    """
    context = _format_context_for_recommendation(parsed, predicted_rpe, personal_baseline)
    user_msg = f"{context}\n\nWrite the recommendation now."

    try:
        resp = CLIENT.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=RECOMMEND_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        print(f"[generate_recommendation] API error: {e}")
        text = (
            f"Predicted perceived exertion is {predicted_rpe:.1f}/10. "
            "Hydrate, eat a balanced meal with protein, and aim for 7-9 hours of sleep tonight."
        )

    if include_disclaimer:
        text = f"{text}\n\n_{DISCLAIMER}_"
    return text


# ---------------------------------------------------------------------------
# Test harness — run this once with ANTHROPIC_API_KEY to verify
# ---------------------------------------------------------------------------

def _run_tests():
    """Manual smoke test. Run: python llm_components.py"""
    import joblib
    import numpy as np

    test_inputs = [
        ("DETAILED",
         "Did a 45-minute upper-body strength session this morning. Slept badly last night, only "
         "about 5 hours, felt pretty fatigued going in. Bench felt heavy. Mood was just ok."),
        ("VAGUE",
         "did some lifting"),
        ("GARBAGE",
         "asdfgh"),
        ("RUN_W_SLEEP",
         "Just finished a 60 min easy run. Felt great, slept like 8 hours, no soreness."),
        ("TEAM_GAME",
         "Played a 90 minute soccer match with the team. Pretty intense. Stressed all day at work."),
    ]

    print("=" * 70)
    print("LLM PARSER SMOKE TEST")
    print("=" * 70)
    for label, text in test_inputs:
        print(f"\n--- {label} ---")
        print(f"INPUT:  {text}")
        parsed = parse_workout(text)
        print(f"PARSED: {json.dumps(parsed, indent=2)}")

    print("\n" + "=" * 70)
    print("END-TO-END: parse -> features -> model prediction -> recommendation")
    print("=" * 70)
    model = joblib.load("rpe_xgb_model.joblib")
    with open("rpe_model_meta.json") as f:
        meta = json.load(f)

    sample_text = test_inputs[0][1]
    parsed = parse_workout(sample_text)
    user_baseline = 7
    row = to_feature_row(parsed, user_baseline, meta["feature_defaults"], meta["feature_cols"])
    pred = float(model.predict(np.array([row]))[0])
    rmse = meta["cv_metrics"]["xgb_rmse_mean"]

    print(f"\nInput: {sample_text}")
    print(f"User personal baseline (slider): {user_baseline}")
    print(f"\nPredicted RPE: {pred:.2f} (+/- {rmse:.2f})")
    rec = generate_recommendation(parsed, pred, user_baseline)
    print(f"\nRecommendation:\n{rec}")


if __name__ == "__main__":
    _run_tests()
