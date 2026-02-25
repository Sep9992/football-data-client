# tools/import_injuries_to_db.py
# Načte data/injuries.csv a nahraje je do databázové tabulky 'injuries'

import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def import_injuries():
    csv_path = os.path.join(DATA_DIR, "injuries.csv")

    if not os.path.exists(csv_path):
        print(f"❌ Soubor {csv_path} neexistuje!")
        return

    print("🚑 Načítám zranění z CSV...")
    try:
        df = pd.read_csv(csv_path)
        # Ošetření názvů sloupců (pro jistotu)
        df.columns = [c.strip().lower() for c in df.columns]

        if "team_name" not in df.columns or "missing_impact" not in df.columns:
            print("❌ Chyba: CSV musí obsahovat sloupce 'team_name' a 'missing_impact'")
            return

        print(f"📊 Nalezeno {len(df)} týmů s absencemi.")

        with engine.begin() as conn:
            # 1. Vytvoření tabulky
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS injuries (
                    team_name TEXT PRIMARY KEY,
                    missing_impact FLOAT
                )
            """))

            # 2. Vymazání starých dat (pro čistý start)
            conn.execute(text("DELETE FROM injuries"))

            # 3. Vložení nových dat
            # Používáme pandas to_sql pro jednoduchost
            df[["team_name", "missing_impact"]].to_sql("injuries", conn, if_exists="append", index=False)

        print("✅ Data úspěšně uložena do DB tabulky 'injuries'.")
        print("   Nyní můžete spouštět predikční skripty.")

    except Exception as e:
        print(f"❌ Chyba importu: {e}")


if __name__ == "__main__":
    import_injuries()