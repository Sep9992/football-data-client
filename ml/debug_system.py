# debug_system.py
# Diagnostika celého systému: DB, Voting Model, Poisson Model
# Ověřuje, zda jsou všechny části kompatibilní.

import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def check_system():
    print("🕵️  KOMPLEXNÍ DIAGNOSTIKA SYSTÉMU")
    print("=" * 40)

    # --- 1. KONTROLA DATABÁZE ---
    print("\n1. KONTROLA DATABÁZE (prepared_fixtures)")
    try:
        # Zkusíme načíst 1 řádek
        df_fixt = pd.read_sql("SELECT * FROM prepared_fixtures LIMIT 1", engine)
        cols = df_fixt.columns.tolist()
        print(f"   ✅ Tabulka existuje.")
        print(f"   📊 Celkem sloupců: {len(cols)}")

        # Klíčové nové features, které musí existovat
        critical_cols = ["market_value_diff", "home_goals_volatility", "elo_diff"]
        missing = [c for c in critical_cols if c not in cols]

        if missing:
            print(f"   ❌ CHYBA: V databázi chybí sloupce: {missing}")
            print("      -> Spusťte 'ml/step2_prepare_dataset.py'")
        else:
            print(f"   ✅ Klíčové sloupce nalezeny ({', '.join(critical_cols)})")

    except Exception as e:
        print(f"   ❌ Chyba při čtení DB: {e}")

    # --- 2. KONTROLA MODELŮ ---
    print("\n2. KONTROLA MODELŮ (Synchronizace)")

    voting_path = os.path.join(DATA_DIR, "model_voting_ensemble.pkl")
    poisson_path = os.path.join(DATA_DIR, "model_poisson.pkl")

    voting_feats = 0
    poisson_feats = 0

    # A) Voting Ensemble
    if os.path.exists(voting_path):
        try:
            artifact = joblib.load(voting_path)
            # Očekáváme 4 položky: (imputer, scaler, model, features)
            if len(artifact) == 4:
                features = artifact[3]
                voting_feats = len(features)
                print(f"   ✅ Voting Model: OK (Vyžaduje {voting_feats} features)")
            else:
                print(f"   ⚠️ Voting Model: Neznámý formát ({len(artifact)} položek)")
        except Exception as e:
            print(f"   ❌ Voting Model: Chyba ({e})")
    else:
        print("   ❌ Voting Model: Soubor chybí!")

    # B) Poisson Model
    if os.path.exists(poisson_path):
        try:
            artifact = joblib.load(poisson_path)
            # Očekáváme 6 položek: (imputer, scaler, reg_h, reg_a, features, rho)
            if len(artifact) == 6:
                features = artifact[4]
                poisson_feats = len(features)
                print(f"   ✅ Poisson Model: OK (Vyžaduje {poisson_feats} features)")
                print(f"      Rho parameter: {artifact[5]:.4f}")
            else:
                print(f"   ⚠️ Poisson Model: Neznámý formát ({len(artifact)} položek)")
        except Exception as e:
            print(f"   ❌ Poisson Model: Chyba ({e})")
    else:
        print("   ❌ Poisson Model: Soubor chybí!")

    # --- 3. ZÁVĚR ---
    print("-" * 40)
    if voting_feats > 0 and poisson_feats > 0:
        if voting_feats == poisson_feats:
            print(f"✅ VŠE V POŘÁDKU. Oba modely používají {voting_feats} features.")
            print("   Systém je připraven na predikce.")
        else:
            print(f"❌ NESHOODA! Voting chce {voting_feats}, ale Poisson chce {poisson_feats}.")
            print("   -> Spusťte 'ml/step3_train_model.py' pro sjednocení.")
    else:
        print("❌ Kritická chyba: Jeden nebo oba modely chybí.")


if __name__ == "__main__":
    check_system()