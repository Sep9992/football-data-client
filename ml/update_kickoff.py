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
        # Střede 18.02.
        ("Wolverhampton Wanderers", "Arsenal FC", "2026-02-18 21:00:00"),
        # Sobota 21.02.
        ("Aston Villa", "Leeds United", "2026-02-21 16:00:00"),
        ("Brentford", "Brighton & Hove Albion", "2026-02-21 16:00:00"),
        ("Chelsea FC", "Burnley FC", "2026-02-21 16:00:00"),
        ("West Ham United", "AFC Bournemouth", "2026-02-21 18:30:00"),
        ("Manchester City", "Newcastle United", "2026-02-21 21:00:00"),
        # Neděle 22.02.
        ("Crystal Palace", "Wolverhampton Wanderers", "2026-02-22 15:00:00"),
        ("Nottingham Forest", "Liverpool FC", "2026-02-22 15:00:00"),
        ("AFC Sunderland", "Fulham FC", "2026-02-22 15:00:00"),
        ("Tottenham Hotspur", "Arsenal FC", "2026-02-22 17:30:00"),
        # Pondělí 23.02.
        ("Everton FC", "Manchester United", "2026-02-23 21:00:00"),
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