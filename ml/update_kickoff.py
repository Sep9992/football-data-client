# tools/update_kickoff.py
# Ruční oprava časů zápasů v databázi

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def update_times():
    print("🕒 Aktualizuji časy výkopů...")

    # ZDE SI UPRAVTE ČASY PODLE REALITY (Formát: RRRR-MM-DD HH:MM:00)
    # Stačí zadat Domácí tým a správný čas.
    updates = {
        "Liverpool FC": "2026-01-17 13:30:00",
        "Manchester City": "2026-01-24 18:30:00",
        "Tottenham Hotspur": "2026-01-17 16:00:00",
        "Brighton & Hove Albion": "2026-01-17 16:00:00",
        "Brentford": "2026-01-24 16:00:00",
        "Nottingham Forest": "2026-01-17 18:30:00",
        "Manchester United": "2026-01-17 13:30:00",
        "Wolverhampton Wanderers": "2026-01-17 16:00:00",
        "Crystal Palace": "2026-01-24 21:00:00",
        "AFC Sunderland": "2026-01-17 16:00:00",
        "Chelsea FC": "2026-01-17 16:00:00",
        "Leeds United": "2026-01-17 16:00:00",
        "Aston Villa": "2026-01-17 16:00:00",
        "Newcastle United": "2026-01-24 13:30:00",
        "Fulham FC": "2026-01-24 16:00:00"
    }

    with engine.begin() as conn:
        for home_team, new_time in updates.items():
            # SQL update
            sql = text("UPDATE prepared_fixtures SET match_date = :dt WHERE home_team = :ht")
            result = conn.execute(sql, {"dt": new_time, "ht": home_team})

            if result.rowcount > 0:
                print(f"✅ {home_team}: Čas změněn na {new_time}")
            else:
                print(f"⚠️ {home_team}: Zápas nenalezen (možná překlep v názvu?)")

    print("\n🏁 Hotovo. Nyní spusťte znovu step4_predict_and_report.py")


if __name__ == "__main__":
    update_times()