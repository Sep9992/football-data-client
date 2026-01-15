# debug_system.py
import os
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def check_system():
    print("🕵️ DIAGNOSTIKA SYSTÉMU")
    print("=" * 30)

    # 1. Kontrola Databáze (prepared_fixtures)
    print("\n1. KONTROLA DATABÁZE (prepared_fixtures)")
    try:
        df_fixt = pd.read_sql("SELECT * FROM prepared_fixtures LIMIT 1", engine)
        cols = df_fixt.columns.tolist()
        print(f"   ✅ Tabulka existuje.")
        print(f"   📊 Počet sloupců: {len(cols)}")

        # Hledáme klíčové nové sloupce
        missing = []
        for check in ["market_value_diff", "home_goals_volatility", "home_fatigue_index"]:
            if check in cols:
                print(f"      OK: Sloupec '{check}' nalezen.")
            else:
                print(f"      ❌ CHYBA: Sloupec '{check}' CHYBÍ!")
                missing.append(check)

        if missing:
            print("   -> Databáze je zastaralá. Je nutné spustit step2.")
    except Exception as e:
        print(f"   ❌ Chyba při čtení DB: {e}")

    # 2. Kontrola Modelu (model_poisson.pkl)
    print("\n2. KONTROLA MODELU (model_poisson.pkl)")
    path = os.path.join(DATA_DIR, "model_poisson.pkl")
    if os.path.exists(path):
        try:
            artifact = joblib.load(path)
            print(f"   ✅ Soubor nalezen. Obsahuje {len(artifact)} položek.")

            # Unpacking
            if len(artifact) == 6:
                imputer, scaler, reg_h, reg_a, features, rho = artifact
                print(f"   📊 Model očekává {len(features)} features.")

                # Kontrola Scaleru
                if hasattr(scaler, "n_features_in_"):
                    print(f"   ⚖️ Scaler byl natrénován na {scaler.n_features_in_} features.")

                if len(features) != scaler.n_features_in_:
                    print("   ❌ KRITICKÁ CHYBA: Seznam features nesedí se Scalerem!")
                else:
                    print("   ✅ Features a Scaler jsou synchronní.")

            else:
                print("   ⚠️ Varování: Model má starý formát (méně než 6 položek).")
        except Exception as e:
            print(f"   ❌ Chyba při načítání modelu: {e}")
    else:
        print("   ❌ Soubor model_poisson.pkl neexistuje!")


if __name__ == "__main__":
    check_system()