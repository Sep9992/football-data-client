# ml/step7_backtest_strategy.py
# Simulace sázkové strategie: DOUBLE CHANCE (Sjednocená logika se Step 4)
# Thresholds: 1X > 0.78 | X2 > 0.55 | Odds > 1.20

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

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

from ml.shared_features import performance_features


def backtest():
    print("⏳ Načítám data pro Finální Backtest (Sjednocená logika)...")
    df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine)
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    valid_features = [f for f in performance_features if f in df.columns]

    # --- NASTAVENÍ ---
    START_PCT = 0.60
    BANKROLL = 10000
    BET_SIZE = 100

    # Používáme stejný základní model jako při tréninku
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42,
                                class_weight="balanced")
    calibrated_clf = CalibratedClassifierCV(rf, method='isotonic', cv=3)

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', calibrated_clf)
    ])

    start_index = int(len(df) * START_PCT)
    retrain_step = 10

    history = []
    balance = BANKROLL
    bets_log = []

    print(f"🚀 Startuji Backtest...")

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
        y_probs = model.predict_proba(X_test)

        for idx, probs in enumerate(y_probs):
            row = test_data.iloc[idx]
            actual = row["target"]  # 0=Home, 1=Draw, 2=Away

            p_home, p_draw, p_away = probs[0], probs[1], probs[2]

            bet_type = None
            prob_dc = 0

            # --- SJEDNOCENÁ LOGIKA (PODLE STEP 4) ---

            # 1. VĚTEV: DOMÁCÍ (Favorit nebo Safe)
            if p_home > p_away:
                if p_home > 0.60:
                    # Model křičí "Favorit 1".
                    # V backtestu DC strategie to bereme jako "Extrémně silnou 1X"
                    bet_type = "1X"
                    prob_dc = p_home + p_draw
                    won = (actual != 2)
                elif (p_home + p_draw) > 0.78:  # Změněno z 0.75 na 0.78 (dle step4)
                    # "Safe 1X"
                    bet_type = "1X"
                    prob_dc = p_home + p_draw
                    won = (actual != 2)

            # 2. VĚTEV: HOSTÉ (Favorit nebo Value)
            elif p_away > p_home:
                if p_away > 0.60:
                    # Model křičí "Favorit 2".
                    # Bereme jako silnou X2
                    bet_type = "X2"
                    prob_dc = p_away + p_draw
                    won = (actual != 0)
                elif (p_away + p_draw) > 0.55:  # Zůstává 0.55 (dle step4)
                    # "Value X2"
                    bet_type = "X2"
                    prob_dc = p_away + p_draw
                    won = (actual != 0)

            # 3. VĚTEV: REMÍZA (Ignorujeme, step4 dává "Risk Remíza")
            else:
                pass

            if bet_type is None: continue

            # --- SIMULACE KURZU A FILTR ---
            margin = 0.90
            simulated_odds = (1 / prob_dc) * margin

            # Filtr "Anti-Odpad" (Ochrana proti kurzům 1.15 a méně)
            if simulated_odds < 1.20:
                continue

            # Vyhodnocení sázky
            if won:
                profit = (BET_SIZE * simulated_odds) - BET_SIZE
                res_str = "WIN"
            else:
                profit = -BET_SIZE
                res_str = "LOSS"

            balance += profit
            history.append(profit)

            bets_log.append({
                "Date": row["match_date"],
                "Match": f"{row['home_team']} vs {row['away_team']}",
                "Bet": bet_type,
                "Prob_DC": round(prob_dc, 2),
                "Odds": round(simulated_odds, 2),
                "Result": res_str,
                "Profit": round(profit, 2)
            })

    # --- Vyhodnocení ---
    total = len(history)
    wins = len([p for p in history if p > 0])
    win_rate = (wins / total * 100) if total > 0 else 0
    roi = ((balance - BANKROLL) / (total * BET_SIZE) * 100) if total > 0 else 0

    print("\n📊 VÝSLEDKY BACKTESTU (LOGIKA STEP 4)")
    print("=" * 40)
    print(f"Sázky: {total} | Výhry: {wins}")
    print(f"Win Rate: {win_rate:.1f} %")
    print(f"ROI:      {roi:.2f} %")
    print(f"Zisk:     {balance - BANKROLL:.2f}")
    print("=" * 40)

    if bets_log:
        pd.DataFrame(bets_log).to_csv(os.path.join(DATA_DIR, "backtest_final_unified.csv"), index=False)


if __name__ == "__main__":
    backtest()