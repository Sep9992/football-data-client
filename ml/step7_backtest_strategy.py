# ml/step7_backtest_strategy.py
# Simulace: "THE SNIPER" (High Stakes on High Confidence) 🎯
# VÍTĚZNÁ STRATEGIE: Single Sázky + Agresivní Staking na Favority
# Opravena chyba v názvu proměnné MIN_BET_AMOUNT

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson

# Nastavení zobrazení
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

from ml.shared_features import performance_features


def calculate_dixon_coles_probs(avg_home_goals, avg_away_goals, rho=0, max_goals=10):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, avg_home_goals) * poisson.pmf(j, avg_away_goals)
    prob_matrix /= prob_matrix.sum()
    return np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)), np.sum(np.triu(prob_matrix, 1))


def backtest():
    print("⏳ Načítám data pro SNIPER Backtest (Final Version)...")
    df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine)
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    valid_features = [f for f in performance_features if f in df.columns]

    # --- KONFIGURACE BANKROLLU ---
    START_BANKROLL = 10000
    MIN_BET_AMOUNT = 50  # Minimální sázka

    # --- STAKING (Vítězné nastavení) ---
    PCT_MAX = 0.09  # 9% na Tutovky
    PCT_STD = 0.03  # 3% na Standard

    # --- PRAHY (Vítězné nastavení) ---
    THRESH_FAVORIT = 0.55
    THRESH_SAFE = 0.75
    THRESH_VALUE = 0.55
    THRESH_SUPER = 0.82

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42,
                                class_weight="balanced")
    calibrated_clf = CalibratedClassifierCV(rf, method='isotonic', cv=3)

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', calibrated_clf)
    ])

    start_index = int(len(df) * 0.60)
    retrain_step = 10

    current_bankroll = START_BANKROLL
    bets_log = []

    print(f"🚀 Startuji Backtest (MAX sázka: {PCT_MAX * 100}% | STD sázka: {PCT_STD * 100}%)...")

    for i in range(start_index, len(df), retrain_step):
        train_data = df.iloc[:i]
        end_idx = min(i + retrain_step, len(df))
        test_data = df.iloc[i:end_idx].copy()

        if test_data.empty: break

        X_train = train_data[valid_features]
        y_train = train_data["target"]

        try:
            model.fit(X_train, y_train)
        except:
            continue

        X_test = test_data[valid_features]
        y_probs_rf = model.predict_proba(X_test)

        for idx, probs_rf in enumerate(y_probs_rf):
            row = test_data.iloc[idx]
            actual = row["target"]

            # Hybridní model
            rf_h, rf_d, rf_a = probs_rf[0], probs_rf[1], probs_rf[2]
            avg_h = row.get("home_avg_goals", 1.5)
            avg_a = row.get("away_avg_goals", 1.2)
            if pd.isna(avg_h): avg_h = 1.5
            if pd.isna(avg_a): avg_a = 1.2

            poi_h, poi_d, poi_a = calculate_dixon_coles_probs(avg_h, avg_a)

            ph = (rf_h * 0.7) + (poi_h * 0.3)
            pd_prob = (rf_d * 0.7) + (poi_d * 0.3)
            pa = (rf_a * 0.7) + (poi_a * 0.3)

            total_prob = ph + pd_prob + pa
            ph /= total_prob
            pd_prob /= total_prob
            pa /= total_prob

            signal_type = None
            bet_code = None
            prob_dc = 0
            is_winner = False
            stake_pct = PCT_STD
            note = "🛡️ STD"

            # DOMÁCÍ
            if ph > pa:
                if ph > THRESH_FAVORIT:
                    signal_type = "🔥 Favorit"
                    bet_code = "1X"
                    prob_dc = ph + pd_prob
                    is_winner = (actual != 2)
                    stake_pct = PCT_MAX
                    note = "🔥 MAX"
                elif (ph + pd_prob) > THRESH_SAFE:
                    signal_type = "✅ Safe"
                    bet_code = "1X"
                    prob_dc = ph + pd_prob
                    is_winner = (actual != 2)
                    if prob_dc > THRESH_SUPER:
                        stake_pct = PCT_MAX
                        note = "💎 SAFE+"

            # HOSTÉ
            elif pa > ph:
                if pa > THRESH_FAVORIT:
                    signal_type = "🔥 Favorit"
                    bet_code = "X2"
                    prob_dc = pa + pd_prob
                    is_winner = (actual != 0)
                    stake_pct = PCT_MAX
                    note = "🔥 MAX"
                elif (pa + pd_prob) > THRESH_VALUE:
                    signal_type = "✨ Value"
                    bet_code = "X2"
                    prob_dc = pa + pd_prob
                    is_winner = (actual != 0)
                    if prob_dc > THRESH_SUPER:
                        stake_pct = PCT_MAX
                        note = "💎 SAFE+"

            if not signal_type: continue

            stake = current_bankroll * stake_pct
            # OPRAVA: Používáme správnou proměnnou MIN_BET_AMOUNT
            if stake < MIN_BET_AMOUNT: stake = MIN_BET_AMOUNT
            stake = min(stake, current_bankroll * 0.15)

            margin = 0.90
            simulated_odds = (1 / prob_dc) * margin

            if simulated_odds < 1.20: continue

            if is_winner:
                profit = (stake * simulated_odds) - stake
                result_str = "✅ WIN"
            else:
                profit = -stake
                result_str = "❌ LOSS"

            current_bankroll += profit

            date_str = row["match_date"].strftime("%d.%m.") if pd.notnull(row["match_date"]) else ""
            bets_log.append({
                "Date": date_str,
                "Match": f"{row['home_team']} vs {row['away_team']}",
                "Type": note,
                "Signal": signal_type,
                "Bet": bet_code,
                "Odds": round(simulated_odds, 2),
                "Stake": round(stake, 2),
                "Result": result_str,
                "Profit": round(profit, 2),
                "Bankroll": round(current_bankroll, 2)
            })

    # --- VÝSTUP ---
    total_bets = len(bets_log)
    if total_bets > 0:
        wins = len([b for b in bets_log if "WIN" in b["Result"]])
        losses = total_bets - wins
        win_rate = (wins / total_bets * 100)

        total_invested = sum([b["Stake"] for b in bets_log])
        roi = ((current_bankroll - START_BANKROLL) / total_invested * 100)

        total_won_cash = sum([b["Profit"] for b in bets_log if b["Profit"] > 0])
        total_lost_cash = sum([abs(b["Profit"]) for b in bets_log if b["Profit"] < 0])
    else:
        wins, losses, win_rate, roi, total_won_cash, total_lost_cash = 0, 0, 0, 0, 0, 0

    print("\n" + "=" * 100)
    print(f"📊 DYNAMICKÝ LOG SÁZEK (SNIPER MODE) - {total_bets} sázek")
    print("=" * 100)

    if bets_log:
        df_log = pd.DataFrame(bets_log)
        print(df_log[["Date", "Match", "Type", "Bet", "Odds", "Stake", "Result", "Profit", "Bankroll"]].to_string(
            index=False))

    print("\n" + "=" * 50)
    print(f"💰 VÝSLEDEK STRATEGIE (FINAL)")
    print("=" * 50)
    print(f"Start:          {START_BANKROLL} Kč")
    print(f"Konec:          {current_bankroll:.2f} Kč")
    print(f"Čistý Zisk:     {current_bankroll - START_BANKROLL:.2f} Kč")
    print(f"ROI:            {roi:.2f} %")
    print("-" * 50)
    print(f"✅ Výhry:        {wins} ({win_rate:.1f} %)")
    print(f"❌ Prohry:       {losses}")
    print("=" * 50)

    if bets_log:
        pd.DataFrame(bets_log).to_csv(os.path.join(DATA_DIR, "backtest_final.csv"), index=False)


if __name__ == "__main__":
    backtest()