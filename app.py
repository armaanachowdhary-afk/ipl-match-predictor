"""
IPL Match Outcome Predictor - Streamlit App
Pick two teams, a venue, and toss details -> get a predicted winner + probability.
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    config = joblib.load("feature_config.pkl")
    state = joblib.load("team_state.pkl")
    teams = joblib.load("team_list.pkl")
    venues = joblib.load("venue_list.pkl")
    return model, config, state, teams, venues

model, config, state, teams, venues = load_artifacts()

FORM_STATS = ["recent_run_rate", "recent_pp_run_rate", "recent_death_run_rate",
              "recent_economy", "recent_death_economy",
              "recent_wickets_lost", "recent_wickets_taken"]

st.title("🏏 IPL Match Outcome Predictor")
st.caption(
    "Predicts a winner from pre-match team strength (Elo, recent form, head-to-head, "
    "venue history) — no in-game data. Trained on IPL 2008–2024."
)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
with col2:
    team2_options = [t for t in teams if t != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

venue = st.selectbox("Venue", venues)

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.radio("Toss winner", [team1, team2], horizontal=True)
with col4:
    toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)

def get_form(team, stat):
    return state["team_recent_form_stats"].get(team, {}).get(stat, 0.5)

def h2h_rate(a, b):
    key = tuple(sorted([a, b]))
    s = state["h2h"].get(key)
    if not s or (s.get(a, 0) + s.get(b, 0)) == 0:
        return 0.5
    return s.get(a, 0) / (s.get(a, 0) + s.get(b, 0))

def build_input_row():
    e1 = state["team_elo"].get(team1, 1500)
    e2 = state["team_elo"].get(team2, 1500)

    row = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "team1_win_rate": state["team_win_rate"].get(team1, 0.5),
        "team2_win_rate": state["team_win_rate"].get(team2, 0.5),
        "team1_recent_form": state["team_recent_form"].get(team1, 0.5),
        "team2_recent_form": state["team_recent_form"].get(team2, 0.5),
        "h2h_team1_win_rate": h2h_rate(team1, team2),
        "toss_winner_is_team1": int(toss_winner == team1),
        "toss_decision_bat": int(toss_decision == "bat"),
        "venue_bat_first_win_rate": state["venue_bat_first_win_rate"].get(venue, 0.5),
        "venue_matches_played": state["venue_matches_played"].get(venue, 0),
        "team1_elo": e1,
        "team2_elo": e2,
        "elo_diff": e1 - e2,
    }
    for c in FORM_STATS:
        row[f"team1_{c}"] = get_form(team1, c)
        row[f"team2_{c}"] = get_form(team2, c)

    # explicit diff features, matching training script
    row["win_rate_diff"] = row["team1_win_rate"] - row["team2_win_rate"]
    row["recent_form_diff"] = row["team1_recent_form"] - row["team2_recent_form"]
    for c in FORM_STATS:
        row[f"{c}_diff"] = row[f"team1_{c}"] - row[f"team2_{c}"]

    return pd.DataFrame([row])

st.divider()

if st.button("Predict Winner", type="primary", use_container_width=True):
    if team1 == team2:
        st.error("Pick two different teams.")
    else:
        X = build_input_row()
        proba = model.predict_proba(X)[0]
        p_team1 = proba[1]
        p_team2 = proba[0]

        predicted = team1 if p_team1 > p_team2 else team2
        confidence = max(p_team1, p_team2)

        st.subheader(f"Predicted winner: {predicted}")
        st.progress(float(confidence))
        st.write(f"**Confidence: {confidence:.1%}**")

        c1, c2 = st.columns(2)
        c1.metric(team1, f"{p_team1:.1%}")
        c2.metric(team2, f"{p_team2:.1%}")

        st.caption(
            "⚠️ Pre-match T20 prediction is inherently noisy (published benchmarks "
            "typically land 55-65% accuracy). Treat this as a probability estimate, "
            "not a certainty — cricket has high match-to-match variance."
        )

        with st.expander("See underlying stats used"):
            e1, e2 = state["team_elo"].get(team1, 1500), state["team_elo"].get(team2, 1500)
            st.write(f"**Elo rating** — {team1}: {e1:.0f} | {team2}: {e2:.0f}")
            st.write(f"**Overall win rate** — {team1}: {state['team_win_rate'].get(team1,0.5):.1%} "
                      f"| {team2}: {state['team_win_rate'].get(team2,0.5):.1%}")
            st.write(f"**Recent form (last 5)** — {team1}: {state['team_recent_form'].get(team1,0.5):.1%} "
                      f"| {team2}: {state['team_recent_form'].get(team2,0.5):.1%}")
            st.write(f"**Head-to-head ({team1} win rate)**: {h2h_rate(team1, team2):.1%}")
            st.write(f"**Venue bat-first win rate**: {state['venue_bat_first_win_rate'].get(venue,0.5):.1%} "
                      f"(over {state['venue_matches_played'].get(venue,0)} matches)")

st.divider()
st.caption("Model: XGBoost classifier, cross-validated via GridSearchCV on chronological IPL 2008-2024 data.")
