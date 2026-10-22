import os
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

EXTRA_COLUMNS = [
    "xgot_faced_home", "xgot_faced_away",
    "headed_goals_home", "headed_goals_away",
    "goal_kicks_home", "goal_kicks_away",
    "woodwork_home", "woodwork_away"
]

inspector = inspect(engine)
existing = {c['name'] for c in inspector.get_columns('match_statistics')}

with engine.begin() as conn:
    for col in EXTRA_COLUMNS:
        if col not in existing:
            conn.execute(text(f"ALTER TABLE match_statistics ADD COLUMN {col} FLOAT;"))
            print(f"➕ Přidán sloupec: {col}")
        else:
            print(f"✅ Sloupec {col} již existuje.")
print("🎉 Schéma DB je 100% kompletní!")