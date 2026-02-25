# tools/import_external_matches.py
import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def import_csv():
    csv_path = os.path.join(DATA_DIR, "external_matches.csv")

    if not os.path.exists(csv_path):
        print(f"❌ Soubor {csv_path} neexistuje! Vytvořte ho.")
        return

    print("📥 Načítám externí zápasy z CSV...")
    df = pd.read_csv(csv_path)

    # Konverze data
    df["match_date"] = pd.to_datetime(df["match_date"])

    print(f"📊 Nalezeno {len(df)} záznamů.")

    # Uložení do DB
    with engine.begin() as conn:
        # Vytvoření tabulky, pokud neexistuje
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS external_matches (
                match_date DATE,
                team_name TEXT,
                competition TEXT
            )
        """))

        # Vymazání starých dat (pro čistý import)
        conn.execute(text("DELETE FROM external_matches"))

        # Vložení nových
        df.to_sql("external_matches", conn, if_exists="append", index=False)

    print("✅ Data úspěšně importována do tabulky 'external_matches'.")

if __name__ == "__main__":
    import_csv()