""""
step4_final_prediction.py  v6
==============================
ZMĚNY v6 (KRITICKÁ OPRAVA):
  - Features čteny z prepared_fixtures (step2) namísto on-the-fly výpočtu
  - Důvod: step2 používá ELO-weighted rolling, coverage opravu a správnou
    home_x_elo definici. On-the-fly výpočet byl konzistentní jen na ~3/10 tipech.
  - Kód výrazně zjednodušen (odstraněno ~100 řádků rolling logiky)
  - Zachováno: blend 50% Voting + 50% Poisson, draw threshold, league tagy
"""

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

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def predict_with_draw_boost(proba, threshold):
    preds = []
    for p in proba:
        p_away, p_draw, p_home = p[0], p[1], p[2]
        if p_draw >= threshold and p_draw > min(p_away, p_home):
            preds.append(1)
        else:
            preds.append(int(np.argmax(p)))
    return np.array(preds)


def main():
    print("🚀 Spouštím FINÁLNÍ predikci (best_classifier + regressory)...")

    # --- NAČTENÍ MODELŮ ---
    try:
        best_clf         = joblib.load(os.path.join(MODEL_DIR, "best_classifier.pkl"))
        reg_h            = joblib.load(os.path.join(MODEL_DIR, "poisson_home_goals.pkl"))
        reg_a            = joblib.load(os.path.join(MODEL_DIR, "poisson_away_goals.pkl"))
        xgb_reg_h        = joblib.load(os.path.join(MODEL_DIR, "xgb_home_goals.pkl"))
        xgb_reg_a        = joblib.load(os.path.join(MODEL_DIR, "xgb_away_goals.pkl"))
        trained_features = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

        thr_path       = os.path.join(MODEL_DIR, "draw_threshold.pkl")
        draw_threshold = joblib.load(thr_path) if os.path.exists(thr_path) else 0.35

        print(f"✅ Modely načteny. Očekávají {len(trained_features)} features.")
        print(f"   Draw threshold: {draw_threshold:.2f}")

    except Exception as e:
        print(f"❌ Chyba při načítání modelů: {e}")
        return

    # --- NAČTENÍ ZÁPASŮ Z prepared_fixtures (step2 canonical source) ---
    with engine.begin() as conn:
        df_fixtures = pd.read_sql(text("""
            SELECT *
            FROM prepared_fixtures
            WHERE match_date >= CURRENT_DATE
              AND match_date <= CURRENT_DATE + INTERVAL '14 days'
            ORDER BY match_date ASC, league ASC
        """), conn)

    if df_fixtures.empty:
        print("📭 Žádné nadcházející zápasy.")
        return

    print(f"\n📅 Načteno {len(df_fixtures)} zápasů z prepared_fixtures")

    print("\n" + "=" * 100)
    print(f"{'Zápas':<45} | {'Tip':<5} | {'1':<7} | {'X':<7} | {'2':<7} | {'xG':<10} | {'Conf'}")
    print("-" * 100)

    for _, row in df_fixtures.iterrows():
        home = row.get('home_team', '?')
        away = row.get('away_team', '?')
        league = row.get('league', '?')
        match_name = f"[{league}] {home} vs {away}"

        try:
            # Doplň chybějící features nulou
            X_full = pd.DataFrame([row])
            for col in trained_features:
                if col not in X_full.columns:
                    X_full[col] = 0.0
            X_input = X_full[trained_features].astype(float)

            # A) Klasifikátor → pravděpodobnosti (Away=0, Draw=1, Home=2)
            probs_clf = best_clf.predict_proba(X_input)[0]
            p_clf_away, p_clf_draw, p_clf_home = probs_clf[0], probs_clf[1], probs_clf[2]

            # B) xG (Poisson + XGBoost hybrid, clamp 0.1–8.0)
            gh = np.clip((reg_h.predict(X_input)[0] + xgb_reg_h.predict(X_input)[0]) / 2, 0.1, 8.0)
            ga = np.clip((reg_a.predict(X_input)[0] + xgb_reg_a.predict(X_input)[0]) / 2, 0.1, 8.0)

            # C) Poisson distribuce
            p1_poi, px_poi, p2_poi = 0.0, 0.0, 0.0
            for h in range(10):
                for a in range(10):
                    p = poisson.pmf(h, gh) * poisson.pmf(a, ga)
                    if   h > a: p1_poi += p
                    elif h == a: px_poi += p
                    else:        p2_poi += p

            # D) Blend 50/50: klasifikátor + Poisson
            p1 = 0.5 * p_clf_home + 0.5 * p1_poi
            px = 0.5 * p_clf_draw + 0.5 * px_poi
            p2 = 0.5 * p_clf_away + 0.5 * p2_poi
            total = p1 + px + p2
            p1, px, p2 = p1 / total, px / total, p2 / total

            # E) Tip s Draw boost thresholdem
            pred_class = predict_with_draw_boost(np.array([[p2, px, p1]]), draw_threshold)[0]
            if pred_class == 2:    tip = "1" if p1 > 0.50 else ("1X" if px > p2 else "1")
            elif pred_class == 0:  tip = "2" if p2 > 0.50 else ("X2" if px > p1 else "2")
            else:                  tip = "X"

            # F) Confidence: o kolik % je nejsilnější výsledek nad random baseline (33%)
            max_p = max(p1, px, p2)
            confidence = max_p / (1/3)

            print(f"{match_name:<45} | {tip:<5} | {p1*100:>5.1f}% | {px*100:>5.1f}% | {p2*100:>5.1f}% | {gh:.2f}:{ga:.2f} | {confidence:.2f}x")

        except Exception as e:
            print(f"❌ Chyba {home}: {e}")
            import traceback; traceback.print_exc()

    print("=" * 100)


if __name__ == "__main__":
    main()