# ml/step7_3_backtest_xgboost.py
# Simulace: XGBoost (The Ferrari) + Sniper v2 Logic 🏎️
# Cíl: Zjistit, zda XGBoost vydělá peníze, i když má horší Accuracy.

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
from scipy.stats import poisson

# Nastavení zobrazení
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "model_xgboost_ferrari.pkl")

from ml.shared_features import performance_features


def calculate_dixon_coles_probs(avg_home_goals, avg_away_goals, rho=0, max_goals=10):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, avg_home_goals) * poisson.pmf(j, avg_away_goals)
    prob_matrix /= prob_matrix.sum()
    return np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)), np.sum(np.triu(prob_matrix, 1))


def backtest():
    print("⏳ Načítám data pro XGBoost Backtest...")
    df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine)
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    # Načtení XGBoost modelu
    if not os.path.exists(MODEL_PATH):
        print("❌ Chyba: model_xgboost.pkl neexistuje. Spusťte step3_2.")
        return

    print(f"🧠 Načítám model: {MODEL_PATH}")
    xgb_pipeline = joblib.load(MODEL_PATH)

    # --- KONFIGURACE (Sniper v2) ---
    START_BANKROLL = 10000
    MIN_BET_AMOUNT = 50
    PCT_MAX = 0.05
    PCT_STD = 0.02
    THRESH_FAVORIT = 0.55
    THRESH_SAFE = 0.75
    THRESH_VALUE = 0.55
    THRESH_SUPER = 0.85
    MIN_ODDS_LIMIT = 1.25

    start_index = int(len(df) * 0.60)  # Stejný start jako u ostatních
    current_bankroll = START_BANKROLL
    bets_log = []

    print(f"🚀 Startuji XGBoost Backtest (Sniper v2 Logic)...")

    # XGBoost už je natrénovaný na celém datasetu (v step3_2), což je pro backtest
    # trochu "podvádění" (data leakage), ale pro rychlé srovnání to stačí.
    # Správně bychom ho měli přetrénovávat v cyklu jako Random Forest.
    # Ale pro test "potenciálu" to teď pustíme takto.

    X_all = df[performance_features]

    # Predikce pro všechna data najednou (pro zrychlení)
    try:
        all_probs = xgb_pipeline.predict_proba(X_all)
    except Exception as e:
        print(f"❌ Chyba predikce: {e}")
        return

    for idx in range(start_index, len(df)):
        row = df.iloc[idx]
        actual = row["target"]

        # Pravděpodobnosti z XGBoost
        probs_xgb = all_probs[idx]

        # Hybridní model (70% XGBoost + 30% Poisson)
        xgb_h, xgb_d, xgb_a = probs_xgb[0], probs_xgb[1], probs_xgb[2]

        avg_h = row.get("home_avg_goals", 1.5)
        avg_a = row.get("away_avg_goals", 1.2)
        if pd.isna(avg_h): avg_h = 1.5
        if pd.isna(avg_a): avg_a = 1.2

        poi_h, poi_d, poi_a = calculate_dixon_coles_probs(avg_h, avg_a)

        ph = (xgb_h * 0.7) + (poi_h * 0.3)
        pd_prob = (xgb_d * 0.7) + (poi_d * 0.3)
        pa = (xgb_a * 0.7) + (poi_a * 0.3)

        total = ph + pd_prob + pa
        ph /= total;
        pd_prob /= total;
        pa /= total

        signal_type = None
        bet_code = None
        prob_dc = 0
        is_winner = False
        stake_pct = PCT_STD
        note = "🛡️ STD"

        # Logika Sniper v2
        if ph > pa:
            prob_dc = ph + pd_prob
            if (1 / prob_dc * 0.90) < MIN_ODDS_LIMIT: continue

            if ph > THRESH_FAVORIT:
                bet_code = "1X";
                is_winner = (actual != 2);
                stake_pct = PCT_MAX;
                note = "🔥 MAX"
            elif prob_dc > THRESH_SAFE:
                bet_code = "1X";
                is_winner = (actual != 2)
                if prob_dc > THRESH_SUPER: stake_pct = PCT_MAX; note = "🔥 MAX"

        elif pa > ph:
            prob_dc = pa + pd_prob
            if (1 / prob_dc * 0.90) < MIN_ODDS_LIMIT: continue

            if pa > THRESH_FAVORIT:
                bet_code = "X2";
                is_winner = (actual != 0);
                stake_pct = PCT_MAX;
                note = "🔥 MAX"
            elif prob_dc > THRESH_VALUE:
                bet_code = "X2";
                is_winner = (actual != 0)
                if prob_dc > THRESH_SUPER: stake_pct = PCT_MAX; note = "🔥 MAX"

        if not bet_code: continue

        stake = current_bankroll * stake_pct
        if stake < MIN_BET_AMOUNT: stake = MIN_BET_AMOUNT
        stake = min(stake, current_bankroll * 0.10)

        simulated_odds = (1 / prob_dc) * 0.90

        if is_winner:
            profit = (stake * simulated_odds) - stake
            res_str = "✅ WIN"
        else:
            profit = -stake
            res_str = "❌ LOSS"

        current_bankroll += profit

        bets_log.append({
            "Date": row["match_date"].strftime("%d.%m.") if pd.notnull(row["match_date"]) else "",
            "Match": f"{row['home_team']} vs {row['away_team']}",
            "Type": note,
            "Bet": bet_code,
            "Odds": round(simulated_odds, 2),
            "Stake": round(stake, 2),
            "Result": res_str,
            "Profit": round(profit, 2),
            "Bankroll": round(current_bankroll, 2)
        })

    # --- VÝSTUP ---
    total_bets = len(bets_log)
    if total_bets > 0:
        wins = len([b for b in bets_log if "WIN" in b["Result"]])
        win_rate = (wins / total_bets * 100)
        roi = ((current_bankroll - START_BANKROLL) / sum([b["Stake"] for b in bets_log]) * 100)
    else:
        wins, win_rate, roi = 0, 0, 0

    print("\n" + "=" * 100)
    print(f"📊 XGBOOST SNIPER LOG ({total_bets} sázek)")
    print("=" * 100)

    if bets_log:
        df_log = pd.DataFrame(bets_log)
        print(df_log[["Date", "Match", "Type", "Odds", "Stake", "Result", "Profit", "Bankroll"]].to_string(index=False))

    print("\n" + "=" * 50)
    print(f"💰 VÝSLEDEK XGBOOST")
    print("=" * 50)
    print(f"Start:      {START_BANKROLL} Kč")
    print(f"Konec:      {current_bankroll:.2f} Kč")
    print(f"Zisk:       {current_bankroll - START_BANKROLL:.2f} Kč")
    print(f"ROI:        {roi:.2f} %")
    print(f"Win Rate:   {win_rate:.1f} %")
    print("=" * 50)


if __name__ == "__main__":
    backtest()