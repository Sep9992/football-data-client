# ml/step4_predict_next_round.py
# Predikce s Hybridním modelem (Voting + Dixon-Coles)
# FIX: Inteligentní rozbalení modelu (Smart Unpacking)

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy.stats import poisson

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# --- Dixon-Coles Probability Funkce ---
def calculate_dixon_coles_probs(avg_home_goals, avg_away_goals, rho, max_goals=10):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, avg_home_goals) * poisson.pmf(j, avg_away_goals)

    def correction(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - (lam * mu * rho)
        elif x == 0 and y == 1:
            return 1 + (lam * rho)
        elif x == 1 and y == 0:
            return 1 + (mu * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    for i in range(2):
        for j in range(2):
            tau = correction(i, j, avg_home_goals, avg_away_goals, rho)
            prob_matrix[i, j] = max(0, prob_matrix[i, j] * tau)

    prob_matrix /= prob_matrix.sum()

    prob_home_win = np.sum(np.tril(prob_matrix, -1))
    prob_draw = np.sum(np.diag(prob_matrix))
    prob_away_win = np.sum(np.triu(prob_matrix, 1))

    return prob_home_win, prob_draw, prob_away_win


def predict_next_round():
    print("📥 Načítám data pro predikci...")

    # 1. Načtení dat z DB - LIMIT 10 NEJBLIŽŠÍCH
    query = "SELECT * FROM prepared_fixtures ORDER BY match_date ASC LIMIT 10"
    df_fixt = pd.read_sql(query, engine)

    if df_fixt.empty:
        print("⚠️ Žádné zápasy k predikci.")
        return

    print(f"🔎 Nalezeno {len(df_fixt)} nejbližších zápasů k predikci.")

    predictions = []

    # ---------------------------------------------------------
    # 2. KLASIFIKAČNÍ MODELY (Voting Ensemble)
    # ---------------------------------------------------------
    voting_path = os.path.join(DATA_DIR, "model_voting_ensemble.pkl")

    voting_model = None
    clf_features = []
    v_imputer = None
    v_scaler = None

    if os.path.exists(voting_path):
        artifact = joblib.load(voting_path)

        # --- SMART UNPACKING (Inteligentní rozřazení) ---
        # Protože nevíme přesné pořadí, projdeme položky a poznáme je podle typu

        # První dvě jsou vždy Imputer a Scaler (to se nemění)
        v_imputer = artifact[0]
        v_scaler = artifact[1]

        # Zbytek (index 2 a 3) může být prohozený
        remaining = artifact[2:]

        for item in remaining:
            if hasattr(item, "predict"):  # Pokud to umí predikovat, je to MODEL
                voting_model = item
            elif isinstance(item, (list, np.ndarray, pd.Index)):  # Pokud je to seznam, jsou to FEATURES
                clf_features = item

        # Kontrola, zda jsme našli vše
        if voting_model is None or len(clf_features) == 0:
            # Fallback: Zkusíme natvrdo indexy, pokud detekce selhala
            if len(artifact) >= 4:
                if hasattr(artifact[2], "predict"):
                    voting_model = artifact[2]
                    clf_features = artifact[3]
                else:
                    clf_features = artifact[2]
                    voting_model = artifact[3]

        if voting_model:
            print(f"🔮 Používám klasifikátor: Voting Ensemble (Features: {len(clf_features)})")

            # Příprava dat
            # Filtrujeme jen potřebné sloupce
            try:
                X_clf_raw = df_fixt[clf_features].replace([np.inf, -np.inf], np.nan)
                X_clf_imp = v_imputer.transform(X_clf_raw)
                X_clf_scaled = v_scaler.transform(X_clf_imp)

                # Predikce
                voting_probs = voting_model.predict_proba(X_clf_scaled)
            except Exception as e:
                print(f"❌ Chyba při přípravě dat pro Voting: {e}")
                return
        else:
            print("❌ Chyba: Nepodařilo se identifikovat Voting model v souboru.")
            return
    else:
        print("⚠️ Voting model nenalezen!")
        return

    # ---------------------------------------------------------
    # 3. POISSON MODEL (Dixon-Coles)
    # ---------------------------------------------------------
    poisson_path = os.path.join(DATA_DIR, "model_poisson.pkl")

    dc_probs = [(0, 0, 0)] * len(df_fixt)
    goals_h = [0] * len(df_fixt)
    goals_a = [0] * len(df_fixt)

    if os.path.exists(poisson_path):
        print(f"🔮 Používám model: Dixon-Coles (Poisson)")
        try:
            artifact = joblib.load(poisson_path)

            if len(artifact) == 6:
                imputer, scaler, reg_home, reg_away, poi_features, rho = artifact
            else:
                imputer, scaler, reg_home, reg_away, poi_features = artifact
                rho = 0

            print(f"   ℹ️ Poisson vyžaduje {len(poi_features)} features. Rho={rho:.4f}")

            X_poi_raw = df_fixt[poi_features]
            X_poi_imp = imputer.transform(X_poi_raw)
            X_poi_scaled = scaler.transform(X_poi_imp)

            goals_h = reg_home.predict(X_poi_scaled)
            goals_a = reg_away.predict(X_poi_scaled)

            dc_probs_list = []
            for gh, ga in zip(goals_h, goals_a):
                ph, pd_prob, pa = calculate_dixon_coles_probs(gh, ga, rho)
                dc_probs_list.append((ph, pd_prob, pa))
            dc_probs = dc_probs_list

        except Exception as e:
            print(f"   ❌ Chyba u Poisson modelu: {e}")
    else:
        print("⚠️ Poisson model nenalezen!")

    # ---------------------------------------------------------
    # 4. ULOŽENÍ DO DATABÁZE
    # ---------------------------------------------------------
    with engine.begin() as conn:
        fix_ids = tuple(df_fixt["fixture_id"].tolist())
        if fix_ids:
            if len(fix_ids) == 1:
                conn.execute(text(f"DELETE FROM predictions WHERE fixture_id = {fix_ids[0]}"))
            else:
                conn.execute(text(f"DELETE FROM predictions WHERE fixture_id IN {fix_ids}"))

    new_rows = []

    for idx, row in df_fixt.iterrows():
        v_prob = voting_probs[idx]
        p_prob = dc_probs[idx]
        gh = goals_h[idx]
        ga = goals_a[idx]

        final_home = (v_prob[0] * 0.7) + (p_prob[0] * 0.3)
        final_draw = (v_prob[1] * 0.7) + (p_prob[1] * 0.3)
        final_away = (v_prob[2] * 0.7) + (p_prob[2] * 0.3)

        total = final_home + final_draw + final_away
        final_home /= total
        final_draw /= total
        final_away /= total

        if final_home > final_away and final_home > final_draw:
            tip = "1"
        elif final_away > final_home and final_away > final_draw:
            tip = "2"
        else:
            tip = "X"

        new_rows.append({
            "fixture_id": row["fixture_id"],
            "model_name": "hybrid_dixon_coles",
            "predicted_winner": tip,
            "proba_home_win": round(final_home, 4),
            "proba_draw": round(final_draw, 4),
            "proba_away_win": round(final_away, 4),
            "expected_goals_home": round(gh, 2),
            "expected_goals_away": round(ga, 2),
            "fair_odd_home": round(1 / final_home, 2) if final_home > 0 else 0,
            "fair_odd_draw": round(1 / final_draw, 2) if final_draw > 0 else 0,
            "fair_odd_away": round(1 / final_away, 2) if final_away > 0 else 0
        })

    if new_rows:
        pd.DataFrame(new_rows).to_sql("predictions", engine, if_exists="append", index=False)
        print("✅ Predikce (Hybridní Dixon-Coles) uloženy do DB.")

        report_path = os.path.join(DATA_DIR, "predictions_report.html")
        pd.DataFrame(new_rows).to_html(report_path)
        print(f"📄 Report vygenerován: {report_path}")


if __name__ == "__main__":
    predict_next_round()