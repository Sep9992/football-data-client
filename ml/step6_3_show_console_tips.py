# ml/step6_3_show_console_tips.py
# Rychlý výpis tipů do konzole (Ferrari Dashboard) 🏎️
# VERZE: XGBoost Sniper (Zobrazuje jen data z modelu 'xgboost_sniper')

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# --- KONFIGURACE (Musí sedět se Step 4_3) ---
THRESH_FAVORIT = 0.55
THRESH_SAFE = 0.75
THRESH_VALUE = 0.55
THRESH_SUPER = 0.85
MIN_ODDS_LIMIT = 1.25


def show_tips():
    print("🏎️  Načítám Ferrari tipy (XGBoost) z databáze...")

    # SQL DOTAZ: Filtrujeme jen model 'xgboost_sniper'
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
    WHERE p.model_name = 'xgboost_sniper' 
      AND f.match_date IS NOT NULL
    ORDER BY f.match_date ASC
    LIMIT 20
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Chyba SQL: {e}")
        print("   (Ujistěte se, že proběhl step4_3 a tabulka obsahuje sloupec 'model_name')")
        return

    if df.empty:
        print("⚠️ Žádné Ferrari tipy v databázi. Spusťte nejdříve step4_3_predict_xgboost.py.")
        return

    print("\n" + "=" * 115)
    print(f"🏎️  FERRARI CONSOLE (XGBoost) | Limit kurzu: {MIN_ODDS_LIMIT}+")
    print("=" * 115)
    print(f"{'DATUM':<12} | {'ZÁPAS':<35} | {'TIP':<5} | {'SÍLA':<8} | {'FÉR KURZ':<10} | {'POZNÁMKA'}")
    print("-" * 115)

    for _, row in df.iterrows():
        match_str = f"{row['home_team']} vs {row['away_team']}"
        date_str = row["match_date"].strftime("%d.%m %H:%M") if pd.notnull(row["match_date"]) else ""

        ph = row['proba_home_win']
        pd_prob = row['proba_draw']
        pa = row['proba_away_win']

        tip_label = row['predicted_winner']
        strength_pct = 0.0
        fair_odd = 0.0
        note = ""

        # --- LOGIKA SIGNÁLŮ (Stejná jako ve step4_3) ---

        # DOMÁCÍ
        if ph > pa:
            # 1. Favorit (Čistá 1)
            if ph > THRESH_FAVORIT:
                tip_label = "1"
                strength_pct = ph
                fair_odd = 1 / ph if ph > 0 else 0
                note = "🔥 FAVORIT"
            # 2. Safe (1X)
            elif (ph + pd_prob) > THRESH_SAFE:
                tip_label = "1X"
                strength_pct = ph + pd_prob
                fair_odd = 1 / strength_pct if strength_pct > 0 else 0
                note = "✅ SAFE"
                if strength_pct > THRESH_SUPER:
                    note = "💎 SAFE+"
            else:
                # Slabý favorit
                tip_label = "1 (X)"
                strength_pct = ph
                fair_odd = 1 / ph
                note = ""

        # HOSTÉ
        elif pa > ph:
            # 1. Favorit (Čistá 2)
            if pa > THRESH_FAVORIT:
                tip_label = "2"
                strength_pct = pa
                fair_odd = 1 / pa if pa > 0 else 0
                note = "🔥 FAVORIT"
            # 2. Value (X2)
            elif (pa + pd_prob) > THRESH_VALUE:
                tip_label = "X2"
                strength_pct = pa + pd_prob
                fair_odd = 1 / strength_pct if strength_pct > 0 else 0
                note = "✨ VALUE"
                if strength_pct > THRESH_SUPER:
                    note = "💎 SAFE+"
            else:
                tip_label = "2 (X)"
                strength_pct = pa
                fair_odd = 1 / pa
                note = ""

        else:  # Remíza
            tip_label = "X"
            strength_pct = pd_prob
            fair_odd = 1 / pd_prob
            note = "⚖️ REMÍZA"

        # Varování na nízký kurz (Anti-Odpad filtr)
        # Reálný kurz sázkovky bude cca o 10% nižší než Fair Odd
        est_market_odd = fair_odd * 0.90

        if est_market_odd < MIN_ODDS_LIMIT:
            if note:  # Pokud tam byl signál, přepíšeme ho na varování
                note = "❌ SKIP (Nízký kurz)"

        # Formátování
        print(
            f"{date_str:<12} | {match_str:<35} | {tip_label:<5} | {strength_pct * 100:>5.1f}%  | {fair_odd:<10.2f} | {note}")

    print("=" * 115)


if __name__ == "__main__":
    show_tips()