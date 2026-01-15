# ml/step7_calculate_fair_odds.py
# Výpočet férových kurzů (Value Betting)

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- Nastavení ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Můžeme filtrovat jen nejlepší model, nebo zobrazit všechny
MODEL_FILTER = "voting_ensemble"  # nebo "xgboost", "poisson_goals"


def main():
    print(f"💰 Počítám férové kurzy pro model: {MODEL_FILTER}...")

    # Načtení predikcí
    query = f"""
    SELECT * FROM predictions_next_round 
    WHERE model = '{MODEL_FILTER}'
    ORDER BY match_date ASC
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        print("⚠️ Žádná data nenalezena. Spusťte nejprve step4.")
        return

    # --- Výpočet férových kurzů ---
    # Fair Odd = 100 / Pravděpodobnost (%)
    # Přidáme malou rezervu (margin), např. 5%, abychom sázeli jen při jasné výhodě
    MARGIN = 0.05

    def calc_odd(prob_percent):
        if prob_percent <= 0: return 999.0
        fair_odd = 100 / prob_percent
        # Chceme kurz, který je o MARGIN lepší než fair odd, abychom pokryli chybu modelu
        target_odd = fair_odd * (1 + MARGIN)
        return round(fair_odd, 2), round(target_odd, 2)

    results = []

    print(f"\n{'MATCH':<40} | {'TIP':<5} | {'PROB':<6} | {'FAIR':<6} | {'TARGET':<6} | {'NOTE'}")
    print("-" * 90)

    for _, row in df.iterrows():
        match_str = f"{row['home_team']} vs {row['away_team']}"

        # Zjistíme, co model predikuje jako nejpravděpodobnější
        probs = [row['proba_home_win'], row['proba_draw'], row['proba_away_win']]
        labels = ["1", "X", "2"]

        # Najdeme index nejvyšší pravděpodobnosti
        best_idx = probs.index(max(probs))
        prob = probs[best_idx]
        label = labels[best_idx]

        fair, target = calc_odd(prob)

        # Interpretace síly
        note = ""
        if prob > 65:
            note = "🔥 TOP"
        elif prob > 50:
            note = "✅ Good"
        else:
            note = "⚠️ Risky"

        print(f"{match_str:<40} | {label:<5} | {prob:<5}% | {fair:<6} | {target:<6} | {note}")

        results.append({
            "match": match_str,
            "bet_on": label,
            "probability": prob,
            "fair_odd": fair,
            "target_odd": target  # Kurz, který byste měl hledat u sázkovky
        })

    print("-" * 90)
    print("\n💡 LEGENDA:")
    print("FAIR   = Kurz, při kterém jste na nule (breakeven).")
    print("TARGET = Kurz, který byste měli hledat (zahrnuje 5% marži pro chybu modelu).")
    print("Pokud sázkovka nabízí kurz VYŠŠÍ než TARGET -> SÁZEJTE.")


if __name__ == "__main__":
    main()