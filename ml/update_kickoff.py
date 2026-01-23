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
        # Sobota 24.01.
        ("West Ham United", "AFC Sunderland", "2026-01-24 13:30:00"),
        ("Fulham FC", "Brighton & Hove Albion", "2026-01-24 16:00:00"),
        ("Burnley FC", "Tottenham Hotspur", "2026-01-24 16:00:00"),
        ("Manchester City", "Wolverhampton Wanderers", "2026-01-24 16:00:00"),
        ("AFC Bournemouth", "Liverpool FC", "2026-01-24 18:30:00"),
        # Neděle 25.01.
        ("Newcastle United", "Aston Villa", "2026-01-25 15:00:00"),
        ("Crystal Palace", "Chelsea FC", "2026-01-25 15:00:00"),
        ("Brentford", "Nottingham Forest", "2026-01-25 15:00:00"),
        ("Arsenal FC", "Manchester United", "2026-01-25 17:30:00"),
        # Pondělí 26.01.
        ("Everton FC", "Leeds United", "2026-01-26 21:00:00"),
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