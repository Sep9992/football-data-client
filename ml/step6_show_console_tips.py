# ml/step6_show_console_tips.py
# Výpočet férových kurzů (Value Betting)

# ml/step6_show_console_tips.py
# Zobrazení tipů a kurzů v konzoli (Console Dashboard)
# UPDATE: Čte z nové tabulky 'predictions' a spojuje ji s týmy.

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def show_tips():
    print("💰 Načítám nejnovější tipy z databáze...")

    # SQL DOTAZ: Spojíme predikce (kde jsou čísla) s fixtures (kde jsou jména týmů)
    query = """
    SELECT 
        f.match_date,
        f.home_team,
        f.away_team,
        p.predicted_winner,
        p.proba_home_win,
        p.proba_draw,
        p.proba_away_win,
        p.fair_odd_home,
        p.fair_odd_draw,
        p.fair_odd_away
    FROM predictions p
    JOIN prepared_fixtures f ON p.fixture_id = f.fixture_id
    ORDER BY f.match_date ASC
    LIMIT 15
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Chyba SQL: {e}")
        print("   (Ujistěte se, že proběhl step4 a existuje tabulka 'predictions')")
        return

    if df.empty:
        print("⚠️ Žádné predikce v databázi.")
        return

    # Hlavička výpisu
    print("\n" + "=" * 100)
    print(f"{'ZÁPAS':<35} | {'TIP':<5} | {'SÍLA':<8} | {'FAIR KURZ':<10} | {'POZNÁMKA'}")
    print("=" * 100)

    for _, row in df.iterrows():
        match_str = f"{row['home_team']} vs {row['away_team']}"

        # Logika pro doporučení (stejná jako v reportu)
        ph = row['proba_home_win']
        pd_prob = row['proba_draw']
        pa = row['proba_away_win']

        tip_label = row['predicted_winner']
        strength_pct = 0.0
        fair_odd = 0.0
        note = ""

        # Určení zobrazovaných hodnot
        if tip_label == "1":
            strength_pct = ph
            fair_odd = row['fair_odd_home']
            if ph > 0.60:
                note = "🔥 FAVORIT"
            elif (ph + pd_prob) > 0.80:
                tip_label = "1X"
                strength_pct = ph + pd_prob
                note = "✅ SAFE"

        elif tip_label == "2":
            strength_pct = pa
            fair_odd = row['fair_odd_away']
            if pa > 0.60:
                note = "🔥 FAVORIT"
            elif (pa + pd_prob) > 0.55:
                tip_label = "X2"
                strength_pct = pa + pd_prob
                note = "✨ VALUE"

        else:  # Remíza
            strength_pct = pd_prob
            fair_odd = row['fair_odd_draw']
            note = "⚖️ RISK"

        # Formátování výstupu
        print(f"{match_str:<35} | {tip_label:<5} | {strength_pct * 100:>5.1f}%  | {fair_odd:<10.2f} | {note}")

    print("=" * 100)
    print("ℹ️  Vysvětlivka: 'FAIR KURZ' je nejnižší kurz, který byste měli vsadit.")
    print("   Pokud sázkovka nabízí více, je to výhodné.")


if __name__ == "__main__":
    show_tips()