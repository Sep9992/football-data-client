import os
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

BOOKIE_COLUMNS = [
    "book_1", "book_x", "book_2",
    "book_1x", "book_12", "book_x2",
    "book_over_25", "book_under_25",
    "book_btts_yes", "book_btts_no"
]

inspector = inspect(engine)
existing = {c['name'] for c in inspector.get_columns('bookmaker_odds')}

with engine.begin() as conn:
    for col in BOOKIE_COLUMNS:
        if col not in existing:
            conn.execute(text(f"ALTER TABLE bookmaker_odds ADD COLUMN {col} VARCHAR(50);"))
            print(f"➕ Přidán sloupec do bookmaker_odds: {col}")
        else:
            print(f"✅ Sloupec {col} již existuje.")

print("🎉 Tabulka bookmaker_odds je připravena na ukládání sázkovek!")