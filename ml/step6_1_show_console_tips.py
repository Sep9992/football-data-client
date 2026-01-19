# ml/step6_1_show_console_tips.py
# Rychlý výpis tipů do konzole (Dashboard)
# VERZE: SNIPER v2 (Sjednocená logika se Step 4 a Step 7)

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# --- KONFIGURACE (Sjednocená) ---
THRESH_FAVORIT = 0.55
THRESH_SAFE = 0.75
THRESH_VALUE = 0.55
THRESH_SUPER = 0.82
MIN_ODDS_LIMIT = 1.20


def show_tips():
    print("💰 Načítám nejnovější tipy z databáze (SNIPER v2 Logic)...")

    # Načteme predikce a rovnou je seřadíme podle času
    query = """
    SELECT 
        f.match_date,
        f.home_team,
        f.away_team,
        p.proba_home_win,
        p.proba_draw,
        p.proba_away_win
    FROM predictions p
    JOIN prepared_fixtures f ON p.fixture_id = f.fixture_id
    WHERE f.match_date IS NOT NULL
    ORDER BY f.match_date ASC
    LIMIT 20
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Chyba SQL: {e}")
        return

    if df.empty:
        print("⚠️ Žádná data. Spusťte nejprve step4.")
        return

    # Hlavička
    print("\n" + "=" * 110)
    print(f"{'ČAS':<15} | {'ZÁPAS':<40} | {'TIP':<5} | {'SÍLA':<8} | {'FÉR KURZ':<9} | {'SIGNÁL'}")
    print("=" * 110)

    for _, row in df.iterrows():
        match_str = f"{row['home_team']} vs {row['away_team']}"
        date_str = row['match_date'].strftime("%d.%m. %H:%M")

        # Hybridní pravděpodobnosti (už jsou uložené v DB z predikce step4)
        ph = row['proba_home_win']
        pd_prob = row['proba_draw']
        pa = row['proba_away_win']

        signal_note = ""
        tip_label = ""
        strength = 0.0
        fair_odd = 0.0

        # --- LOGIKA SNIPER v2 ---

        # DOMÁCÍ
        if ph > pa:
            if ph > THRESH_FAVORIT:
                signal_note = "🔥 FAVORIT"
                tip_label = "1"
                strength = ph
            elif (ph + pd_prob) > THRESH_SAFE:
                signal_note = "✅ SAFE"
                tip_label = "1X"
                strength = ph + pd_prob
                if strength > THRESH_SUPER:
                    signal_note = "💎 SAFE+"

        # HOSTÉ
        elif pa > ph:
            if pa > THRESH_FAVORIT:
                signal_note = "🔥 FAVORIT"
                tip_label = "2"
                strength = pa
            elif (pa + pd_prob) > THRESH_VALUE:
                signal_note = "✨ VALUE"
                tip_label = "X2"
                strength = pa + pd_prob
                if strength > THRESH_SUPER:
                    signal_note = "💎 SAFE+"

        # Výpočet kurzu pro daný tip
        if strength > 0:
            fair_odd = 1 / strength

            # Varování na nízký kurz (Anti-Odpad filtr)
            # Reálný kurz sázkovky bude cca o 10% nižší než Fair Odd
            est_market_odd = fair_odd * 0.90

            if est_market_odd < MIN_ODDS_LIMIT:
                signal_note = "❌ SKIP (Nízký kurz)"
                # I když je to favorit, pokud je kurz 1.10, nechceme ho vidět jako "Fire"

        else:
            tip_label = "-"
            signal_note = ""

        # Barvy (jen pro efekt v terminálu, pokud to podporuje, jinak text)
        print(
            f"{date_str:<15} | {match_str:<40} | {tip_label:<5} | {strength * 100:>5.1f}%  | {fair_odd:<9.2f} | {signal_note}")

    print("=" * 110)
    print("ℹ️  Legenda: 🔥 = Čistá výhra, ✅ = Neprohra, ✨ = Value na outsidera, 💎 = Tutovka")
    print(f"ℹ️  Filtr: Ignorujeme zápasy, kde odhadovaný kurz sázkovky < {MIN_ODDS_LIMIT}")


if __name__ == "__main__":
    show_tips()