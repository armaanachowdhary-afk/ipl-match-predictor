# IPL Match Outcome Predictor

An end-to-end ML classification pipeline that predicts IPL match winners using
historical match and ball-by-ball data (2008–2024), with feature engineering on
team win rates, Elo ratings, venue statistics, toss decisions, and recent form
(including powerplay/death-overs batting & bowling performance).



## What it does

Given two teams, a venue, and toss details, the app predicts the likely winner
with a probability estimate, and shows the underlying stats (Elo, recent form,
head-to-head record, venue history) that drove the prediction.

## Approach

- **Data**: [IPL Complete Dataset (2008–2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
  from Kaggle — `matches.csv` (match-level results) and `deliveries.csv`
  (ball-by-ball data).
- **Feature engineering** (all leakage-free — every feature for a match only
  uses data available *before* that match was played):
  - Team win rate & recent form (last 5 matches)
  - **Elo ratings**, updated match-by-match (chess-style rating system)
  - Head-to-head record between the two teams
  - Venue history (bat-first vs field-first win rate)
  - Powerplay (overs 0-5) and death-overs (overs 15-19) batting/bowling form,
    derived from ball-by-ball data
  - Explicit team1-vs-team2 difference features
- **Modeling**: Logistic Regression, Random Forest, and XGBoost, compared via
  5-fold stratified cross-validation with `GridSearchCV`. Evaluated on a
  **chronological holdout** (train on past seasons, test on most recent
  matches) to simulate real deployment rather than random-split leakage.
- **Result**: XGBoost performed best (~57% accuracy, ROC-AUC ~0.55) on unseen
  future matches.

### A note on accuracy

Pre-match T20 cricket prediction (no in-game data — just historical team
strength) is a genuinely hard, high-variance problem. Published benchmarks for
this exact setup typically land in the 55–65% accuracy range. This project
prioritizes a methodologically sound pipeline (no data leakage, proper
chronological validation, real cross-validation) over inflated numbers from a
flawed evaluation setup.

## Project structure

```
├── app.py                  # Streamlit app (loads trained model + team state)
├── features.py              # Data cleaning + leakage-free feature engineering
├── train.py                  # Model training, GridSearchCV, model comparison
├── model_ready_data.csv      # Processed, model-ready dataset (output of features.py)
├── best_model.pkl            # Trained XGBoost model
├── feature_config.pkl        # Feature column config used at inference
├── team_state.pkl            # Latest team Elo/form/venue stats for live predictions
├── team_list.pkl / venue_list.pkl   # Dropdown options for the app
└── requirements.txt
```

## Running locally

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional — to retrain from scratch) Download `matches.csv` and
   `deliveries.csv` from the
   [Kaggle dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
   into the project root, then:
   ```bash
   python features.py   # builds model_ready_data.csv + team_state.pkl
   python train.py       # trains and compares models, saves best_model.pkl
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Tech stack

Python, scikit-learn, XGBoost, pandas, Streamlit
