"""
Feature engineering for IPL Match Outcome Predictor.
Builds a model-ready dataset from matches.csv + deliveries.csv.
"""
import pandas as pd
import numpy as np
import joblib

TEAM_RENAME = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

def load_and_clean(matches_path="matches.csv", deliveries_path="deliveries.csv"):
    m = pd.read_csv(matches_path)
    d = pd.read_csv(deliveries_path)

    # Normalize team names across renamed franchises
    for col in ["team1", "team2", "toss_winner", "winner"]:
        m[col] = m[col].replace(TEAM_RENAME)
    d["batting_team"] = d["batting_team"].replace(TEAM_RENAME)
    d["bowling_team"] = d["bowling_team"].replace(TEAM_RENAME)

    # Drop matches with no result (abandoned/no-result) - can't have a target
    m = m[m["winner"].notnull()].copy()

    # Sort chronologically - critical for building "recent form" without leakage
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date").reset_index(drop=True)

    return m, d


def build_match_team_stats(d: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse ball-by-ball deliveries into one row per (match_id, team)
    with batting and bowling performance, overall and split by phase
    (powerplay = overs 0-5, death = overs 15-19).
    """
    d = d.copy()
    d["phase"] = np.select(
        [d["over"] <= 5, d["over"] >= 15],
        ["powerplay", "death"],
        default="middle",
    )

    # ---- Batting side ----
    bat = d.groupby(["match_id", "batting_team"]).agg(
        runs_scored=("total_runs", "sum"),
        balls_faced=("ball", "count"),
        wickets_lost=("is_wicket", "sum"),
    ).reset_index().rename(columns={"batting_team": "team"})

    bat_pp = d[d["phase"] == "powerplay"].groupby(["match_id", "batting_team"]).agg(
        pp_runs=("total_runs", "sum"), pp_balls=("ball", "count")
    ).reset_index().rename(columns={"batting_team": "team"})

    bat_death = d[d["phase"] == "death"].groupby(["match_id", "batting_team"]).agg(
        death_runs=("total_runs", "sum"), death_balls=("ball", "count")
    ).reset_index().rename(columns={"batting_team": "team"})

    # ---- Bowling side ----
    bowl = d.groupby(["match_id", "bowling_team"]).agg(
        runs_conceded=("total_runs", "sum"),
        balls_bowled=("ball", "count"),
        wickets_taken=("is_wicket", "sum"),
    ).reset_index().rename(columns={"bowling_team": "team"})

    bowl_death = d[d["phase"] == "death"].groupby(["match_id", "bowling_team"]).agg(
        death_runs_conceded=("total_runs", "sum"), death_balls_bowled=("ball", "count")
    ).reset_index().rename(columns={"bowling_team": "team"})

    stats = bat.merge(bat_pp, on=["match_id", "team"], how="left") \
        .merge(bat_death, on=["match_id", "team"], how="left") \
        .merge(bowl, on=["match_id", "team"], how="left") \
        .merge(bowl_death, on=["match_id", "team"], how="left")

    stats = stats.fillna(0)
    stats["run_rate"] = stats["runs_scored"] / (stats["balls_faced"] / 6).replace(0, np.nan)
    stats["pp_run_rate"] = stats["pp_runs"] / (stats["pp_balls"] / 6).replace(0, np.nan)
    stats["death_run_rate"] = stats["death_runs"] / (stats["death_balls"] / 6).replace(0, np.nan)
    stats["economy"] = stats["runs_conceded"] / (stats["balls_bowled"] / 6).replace(0, np.nan)
    stats["death_economy"] = stats["death_runs_conceded"] / (stats["death_balls_bowled"] / 6).replace(0, np.nan)
    stats = stats.fillna(stats.median(numeric_only=True))

    return stats[["match_id", "team", "run_rate", "pp_run_rate", "death_run_rate",
                   "economy", "death_economy", "wickets_lost", "wickets_taken"]]


def rolling_team_form(m: pd.DataFrame, match_team_stats: pd.DataFrame, n=5) -> pd.DataFrame:
    """
    For each match, compute each team's ROLLING AVERAGE of the above
    performance stats over their last n matches (strictly before this match).
    """
    stat_cols = ["run_rate", "pp_run_rate", "death_run_rate", "economy",
                 "death_economy", "wickets_lost", "wickets_taken"]

    # long format: one row per team per match, in chronological order
    long = []
    for _, row in m.iterrows():
        for team in [row["team1"], row["team2"]]:
            long.append({"match_id": row["id"], "date": row["date"], "team": team})
    long_df = pd.DataFrame(long).merge(match_team_stats, on=["match_id", "team"], how="left")
    long_df[stat_cols] = long_df[stat_cols].fillna(long_df[stat_cols].median())
    long_df = long_df.sort_values("date")

    # shift(1) ensures we only use PAST matches, then rolling mean over last n
    rolled = (
        long_df.groupby("team")[stat_cols]
        .apply(lambda g: g.shift(1).rolling(n, min_periods=1).mean())
    )
    rolled.columns = [f"recent_{c}" for c in stat_cols]
    out = pd.concat([long_df[["match_id", "team"]], rolled], axis=1)
    # fill first-ever matches (no history) with global median
    out = out.fillna(out.median(numeric_only=True))
    return out


def compute_elo_ratings(m: pd.DataFrame, k=32, base=1500):
    """
    Returns (pre_match, final_elo):
      pre_match: {match_id: {team1_elo, team2_elo}} - rating BEFORE that match
      final_elo: {team: rating} - rating AFTER all matches (for future predictions)
    Standard Elo: updated after every match using actual vs expected result.
    """
    elo = {}
    pre_match = {}

    def get_elo(t):
        return elo.setdefault(t, base)

    for _, match in m.iterrows():
        t1, t2, winner = match["team1"], match["team2"], match["winner"]
        e1, e2 = get_elo(t1), get_elo(t2)
        pre_match[match["id"]] = {"team1_elo": e1, "team2_elo": e2}

        expected1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        actual1 = 1 if winner == t1 else 0

        elo[t1] = e1 + k * (actual1 - expected1)
        elo[t2] = e2 + k * ((1 - actual1) - (1 - expected1))

    return pre_match, elo


def build_features(m: pd.DataFrame, team_form: pd.DataFrame = None, elo: dict = None, elo_final: dict = None, return_state=False):
    """
    Build leakage-free features: every feature for match i only uses
    information available BEFORE that match was played.
    """
    rows = []

    # index team_form for fast lookup: (match_id, team) -> row of recent_* stats
    form_lookup = {}
    if team_form is not None:
        for _, r in team_form.iterrows():
            form_lookup[(r["match_id"], r["team"])] = r

    # running stats, updated match by match
    team_stats = {}   # team -> {'wins':0,'losses':0}
    venue_stats = {}  # venue -> {'wins_bat_first':0, 'wins_field_first':0, 'total':0}
    h2h_stats = {}     # frozenset({teamA,teamB}) -> {teamA: wins, teamB: wins}
    recent_form = {}  # team -> list of last results (1=win,0=loss)

    def get_team(t):
        return team_stats.setdefault(t, {"wins": 0, "losses": 0})

    def win_rate(t):
        s = get_team(t)
        total = s["wins"] + s["losses"]
        return s["wins"] / total if total > 0 else 0.5  # neutral prior

    def recent_win_rate(t, n=5):
        hist = recent_form.get(t, [])
        if not hist:
            return 0.5
        last_n = hist[-n:]
        return sum(last_n) / len(last_n)

    def h2h_win_rate(a, b):
        key = tuple(sorted([a, b]))
        s = h2h_stats.get(key)
        if not s or (s[a] + s[b]) == 0:
            return 0.5
        return s[a] / (s[a] + s[b])

    for _, match in m.iterrows():
        t1, t2 = match["team1"], match["team2"]
        venue = match["venue"]
        toss_winner = match["toss_winner"]
        toss_decision = match["toss_decision"]
        winner = match["winner"]

        # ---- FEATURES (computed BEFORE updating stats with this match's result) ----
        feat = {
            "match_id": match["id"],
            "season": match["season"],
            "team1": t1,
            "team2": t2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1_win_rate": win_rate(t1),
            "team2_win_rate": win_rate(t2),
            "team1_recent_form": recent_win_rate(t1),
            "team2_recent_form": recent_win_rate(t2),
            "h2h_team1_win_rate": h2h_win_rate(t1, t2),
            "toss_winner_is_team1": int(toss_winner == t1),
            "toss_decision_bat": int(toss_decision == "bat"),
        }

        vs = venue_stats.setdefault(venue, {"bat_first_wins": 0, "field_first_wins": 0, "total": 0})
        total_v = vs["total"]
        feat["venue_bat_first_win_rate"] = (vs["bat_first_wins"] / total_v) if total_v > 0 else 0.5
        feat["venue_matches_played"] = total_v

        if elo is not None:
            e = elo[match["id"]]
            feat["team1_elo"] = e["team1_elo"]
            feat["team2_elo"] = e["team2_elo"]
            feat["elo_diff"] = e["team1_elo"] - e["team2_elo"]

        # ball-by-ball derived recent form (powerplay/death/economy/etc.)
        if form_lookup:
            for team_label, team_name in [("team1", t1), ("team2", t2)]:
                r = form_lookup.get((match["id"], team_name))
                if r is not None:
                    for c in ["recent_run_rate", "recent_pp_run_rate", "recent_death_run_rate",
                              "recent_economy", "recent_death_economy",
                              "recent_wickets_lost", "recent_wickets_taken"]:
                        feat[f"{team_label}_{c}"] = r[c]

        # target: did team1 win?
        feat["team1_won"] = int(winner == t1)

        rows.append(feat)

        # ---- UPDATE running stats using this match's actual result ----
        get_team(t1)
        get_team(t2)
        if winner == t1:
            team_stats[t1]["wins"] += 1
            team_stats[t2]["losses"] += 1
        else:
            team_stats[t2]["wins"] += 1
            team_stats[t1]["losses"] += 1

        recent_form.setdefault(t1, []).append(int(winner == t1))
        recent_form.setdefault(t2, []).append(int(winner == t2))

        key = tuple(sorted([t1, t2]))
        h2h = h2h_stats.setdefault(key, {t1: 0, t2: 0})
        h2h.setdefault(t1, 0)
        h2h.setdefault(t2, 0)
        h2h[winner] = h2h.get(winner, 0) + 1

        # winner batted first or fielded first?
        if (toss_winner == winner and toss_decision == "bat") or (toss_winner != winner and toss_decision == "field"):
            vs["bat_first_wins"] += 1
        else:
            vs["field_first_wins"] += 1
        vs["total"] += 1

    result = pd.DataFrame(rows)
    if not return_state:
        return result

    # Final state after processing all matches - needed for live predictions on future matchups
    state = {
        "team_win_rate": {t: (s["wins"] / (s["wins"] + s["losses"]) if (s["wins"] + s["losses"]) > 0 else 0.5)
                           for t, s in team_stats.items()},
        "team_recent_form": {t: recent_win_rate(t) for t in team_stats},
        "team_elo": elo_final if elo_final is not None else {},
        "h2h": h2h_stats,
        "venue_bat_first_win_rate": {v: (s["bat_first_wins"] / s["total"] if s["total"] > 0 else 0.5)
                                      for v, s in venue_stats.items()},
        "venue_matches_played": {v: s["total"] for v, s in venue_stats.items()},
        "team_recent_form_stats": {},  # filled below from team_form
    }
    if team_form is not None:
        latest_form = team_form.sort_values("match_id").groupby("team").last()
        for team, row in latest_form.iterrows():
            state["team_recent_form_stats"][team] = row.to_dict()

    return result, state


if __name__ == "__main__":
    m, d = load_and_clean()
    match_team_stats = build_match_team_stats(d)
    team_form = rolling_team_form(m, match_team_stats, n=5)
    elo, elo_final = compute_elo_ratings(m)
    feats, state = build_features(m, team_form=team_form, elo=elo, elo_final=elo_final, return_state=True)
    print(feats.shape)
    feats.to_csv("model_ready_data.csv", index=False)
    joblib.dump(state, "team_state.pkl")
    joblib.dump(sorted(set(m["team1"]) | set(m["team2"])), "team_list.pkl")
    joblib.dump(sorted(m["venue"].dropna().unique().tolist()), "venue_list.pkl")
    print("Saved model_ready_data.csv, team_state.pkl, team_list.pkl, venue_list.pkl")
