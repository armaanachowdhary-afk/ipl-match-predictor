"""
Train and compare Logistic Regression, Random Forest, and XGBoost
for IPL match outcome prediction. Uses GridSearchCV with cross-validation.
Saves the best model + preprocessing pipeline to disk for the Streamlit app.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("model_ready_data.csv")

# Drop very early matches where venue/team stats are still "cold start" (all 0.5 priors)
# Keep everything - cold start is realistic, but we track venue_matches_played as a feature
# so models can learn to trust it less early on.

CATEGORICAL = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
FORM_STATS = ["recent_run_rate", "recent_pp_run_rate", "recent_death_run_rate",
              "recent_economy", "recent_death_economy",
              "recent_wickets_lost", "recent_wickets_taken"]

NUMERIC = [
    "team1_win_rate", "team2_win_rate",
    "team1_recent_form", "team2_recent_form",
    "h2h_team1_win_rate", "toss_winner_is_team1", "toss_decision_bat",
    "venue_bat_first_win_rate", "venue_matches_played",
    "team1_elo", "team2_elo", "elo_diff",
] + [f"team1_{c}" for c in FORM_STATS] + [f"team2_{c}" for c in FORM_STATS]
TARGET = "team1_won"

# Explicit diff features (team1 - team2) so models don't have to learn subtraction
DIFF_PAIRS = [
    ("team1_win_rate", "team2_win_rate"),
    ("team1_recent_form", "team2_recent_form"),
] + [(f"team1_{c}", f"team2_{c}") for c in FORM_STATS]

for a, b in DIFF_PAIRS:
    diff_name = a.replace("team1_", "") + "_diff"
    df[diff_name] = df[a] - df[b]
    NUMERIC.append(diff_name)

X = df[CATEGORICAL + NUMERIC]
y = df[TARGET]

# Chronological split (matches are already sorted by date) - last 15% as holdout
# This simulates real deployment: train on past, predict future.
split_idx = int(len(df) * 0.85)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)}  Test: {len(X_test)}")

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# ---------------- Logistic Regression ----------------
lr_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
lr_grid = {"clf__C": [0.01, 0.1, 1, 10]}
lr_search = GridSearchCV(lr_pipe, lr_grid, cv=cv, scoring="accuracy", n_jobs=-1)
lr_search.fit(X_train, y_train)
results["Logistic Regression"] = lr_search

# ---------------- Random Forest ----------------
rf_pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=42))])
rf_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [4, 6, 10, None],
    "clf__min_samples_leaf": [1, 3, 5],
}
rf_search = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="accuracy", n_jobs=-1)
rf_search.fit(X_train, y_train)
results["Random Forest"] = rf_search

# ---------------- XGBoost ----------------
xgb_pipe = Pipeline([("prep", preprocessor), ("clf", XGBClassifier(
    eval_metric="logloss", random_state=42))])
xgb_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [3, 4, 6],
    "clf__learning_rate": [0.01, 0.05, 0.1],
}
xgb_search = GridSearchCV(xgb_pipe, xgb_grid, cv=cv, scoring="accuracy", n_jobs=-1)
xgb_search.fit(X_train, y_train)
results["XGBoost"] = xgb_search

# ---------------- Compare on holdout test set ----------------
print("\n" + "=" * 60)
print("MODEL COMPARISON (chronological holdout test set)")
print("=" * 60)

best_name, best_score, best_model = None, -1, None
for name, search in results.items():
    best_est = search.best_estimator_
    preds = best_est.predict(X_test)
    proba = best_est.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"\n{name}")
    print(f"  Best CV params: {search.best_params_}")
    print(f"  Best CV accuracy: {search.best_score_:.4f}")
    print(f"  Test accuracy:    {acc:.4f}")
    print(f"  Test ROC-AUC:     {auc:.4f}")
    if acc > best_score:
        best_name, best_score, best_model = name, acc, best_est

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_name} (test accuracy {best_score:.4f})")
print("=" * 60)

joblib.dump(best_model, "best_model.pkl")
joblib.dump({"categorical": CATEGORICAL, "numeric": NUMERIC}, "feature_config.pkl")
print("\nSaved best_model.pkl and feature_config.pkl")
