# tools/update_kickoff.py
# Ruční oprava časů zápasů v databázi (Opravená logika)
# Identifikuje zápas podle DOMÁCÍHO i HOSTUJÍCÍHO týmu.

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def update_times():
    print("🕒 Aktualizuji časy výkopů...")

    # ZDE SI UPRAVTE SEZNAM ZÁPASŮ PRO TOTO KOLO
    # Formát: [Domácí, Hosté, Nový Čas]
    matches_to_update = [
        # Sobota 17.01.
        ("Manchester United", "Manchester City", "2026-01-17 13:30:00"),
        ("Chelsea FC", "Brentford", "2026-01-17 16:00:00"),
        ("Leeds United", "Fulham FC", "2026-01-17 16:00:00"),
        ("Liverpool FC", "Burnley FC", "2026-01-17 16:00:00"),
        ("AFC Sunderland", "Crystal Palace", "2026-01-17 16:00:00"),
        ("Tottenham Hotspur", "West Ham United", "2026-01-17 16:00:00"),
        ("Nottingham Forest", "Arsenal FC", "2026-01-17 18:30:00"),

        # Neděle 18.01.
        ("Wolverhampton Wanderers", "Newcastle United", "2026-01-18 15:00:00"),
        ("Aston Villa", "Everton FC", "2026-01-18 17:30:00"),

        # Pondělí 19.01.
        ("Brighton & Hove Albion", "AFC Bournemouth", "2026-01-19 21:00:00"),
    ]

    with engine.begin() as conn:
        count_ok = 0
        count_fail = 0

        for home, away, new_time in matches_to_update:
            # SQL update s podmínkou na OBA týmy
            sql = text("""
                UPDATE prepared_fixtures 
                SET match_date = :dt 
                WHERE home_team = :ht AND away_team = :at
            """)

            result = conn.execute(sql, {"dt": new_time, "ht": home, "at": away})

            if result.rowcount > 0:
                print(f"✅ Nastaveno: {home} vs {away} -> {new_time}")
                count_ok += 1
            else:
                print(f"⚠️ NENALEZENO: {home} vs {away} (Zkontrolujte jména týmů v DB)")
                count_fail += 1

    print(f"\n🏁 Hotovo. Úspěšně: {count_ok}, Chyby: {count_fail}")
    print("👉 Nyní spusťte 'ml/step4_1_predict_and_report.py', report by měl být čistý.")


if __name__ == "__main__":
    update_times()