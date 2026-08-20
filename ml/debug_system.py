"""
debug_system_v3.py
==================
Komplexní diagnostika celého football pipeline systému.
Ověřuje: DB, step2 features, step3 modely, step4-8 kompatibilitu.
"""

import os
import pickle
import joblib
import json
import pandas as pd
from sqlalchemy import create_engine, inspect, text  # Přidán import text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

try:
    engine = create_engine(DATABASE_URL)
except Exception as e:
    print(f"❌ Chyba připojení DB: {e}")
    engine = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

print("\n" + "=" * 80)
print("🕵️  KOMPLEXNÍ DIAGNOSTIKA PIPELINE — v3")
print("=" * 80)
print(f"Project root: {BASE_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. KONTROLA DATABÁZE
# ============================================================================
print("1️⃣  KONTROLA DATABÁZE")
print("-" * 80)

if not engine:
    print("   ❌ Databáze není připojena (DATABASE_URL chybí v .env)\n")
else:
    try:
        # Test připojení - OPRAVENO pro SQLAlchemy 2.0+
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
        print("   ✅ Připojení k Supabase: OK\n")

        # Kontrola tabulek
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        print(f"   📋 Tabulky v DB: {len(tables)}")
        for table in ['prepared_fixtures', 'prepared_datasets', 'bookmaker_odds']:
            if table in tables:
                # OPRAVENO pro SQLAlchemy 2.0+
                count = pd.read_sql(text(f"SELECT COUNT(*) as cnt FROM {table}"), engine)
                rows = count['cnt'].values[0]
                print(f"      ✅ {table:<25s} | {rows:>6,} řádků")
            else:
                print(f"      ❌ {table:<25s} | CHYBÍ")

        # Kontrola sloupců v prepared_fixtures
        print(f"\n   📊 Kontrola prepared_fixtures sloupců:")
        # OPRAVENO pro SQLAlchemy 2.0+
        df_sample = pd.read_sql(text("SELECT * FROM prepared_fixtures LIMIT 1"), engine)
        all_cols = df_sample.columns.tolist()
        print(f"      Celkem sloupců: {len(all_cols)}")

        key_features = [
            'elo_diff', 'home_x_elo', 'market_value_diff_scaled', 'home_elo',
            'away_avg_possession_last5', 'away_avg_shots_last5',
            'home_avg_possession_last5', 'home_avg_shots_last5',
            'home_avg_shots_on_target_last5', 'away_avg_shots_on_target_last5',
            'home_avg_goals_last5', 'away_avg_goals_last5',
            'home_avg_passes_total_last5', 'expected_goals_home', 'expected_goals_away'
        ]

        found = sum(1 for c in key_features if c in all_cols)
        print(f"      Feature kompatibilita: {found}/{len(key_features)} ✅")

        if found < len(key_features):
            missing = [c for c in key_features if c not in all_cols]
            print(f"      ⚠️  Chybí features: {missing}")

        print()

    except Exception as e:
        print(f"   ❌ Chyba: {e}\n")

# ============================================================================
# 2. KONTROLA MODELŮ (11 souborů)
# ============================================================================
print("2️⃣  KONTROLA MODELŮ (step3 output)")
print("-" * 80)

models_to_check = {
    'voting_classifier.pkl': ('Voting Ensemble (RF+GBM+LR)', 'CRITICAL'),
    'best_classifier.pkl': ('Best Classifier (alias Voting)', 'CRITICAL'),
    'xgb_classifier.pkl': ('XGBoost Classifier', 'IMPORTANT'),
    'poisson_home_goals.pkl': ('Poisson Home Goals', 'CRITICAL'),
    'poisson_away_goals.pkl': ('Poisson Away Goals', 'CRITICAL'),
    'xgb_home_goals.pkl': ('XGBoost Home Goals', 'SECONDARY'),
    'xgb_away_goals.pkl': ('XGBoost Away Goals', 'SECONDARY'),
    'feature_cols.pkl': ('Feature Columns (20 features)', 'CRITICAL'),
    'draw_threshold.pkl': ('Draw Threshold (0.31)', 'CRITICAL'),
    'market_value_scaler.pkl': ('Market Value Scaler', 'IMPORTANT'),
    'training_log.json': ('Training Log (step3 výsledky)', 'INFO'),
}

found_models = {}
missing_models = []
total_size = 0

print("   Model Status:")
print()

for filename, (description, priority) in models_to_check.items():
    filepath = os.path.join(MODEL_DIR, filename)
    priority_icon = '🔴' if priority == 'CRITICAL' else '🟡' if priority == 'IMPORTANT' else '🟢'

    if os.path.exists(filepath):
        try:
            file_size = os.path.getsize(filepath) / 1024  # KB
            total_size += file_size

            if filename.endswith('.pkl'):
                obj = joblib.load(filepath)

                if filename == 'feature_cols.pkl':
                    if isinstance(obj, list):
                        status = f"| {len(obj)} features"
                        found_models[filename] = len(obj)
                    else:
                        status = f"| Typ: {type(obj).__name__}"
                        found_models[filename] = obj

                elif filename == 'draw_threshold.pkl':
                    status = f"| threshold={obj:.3f}"
                    found_models[filename] = obj

                elif filename == 'market_value_scaler.pkl':
                    status = f"| StandardScaler"
                    found_models[filename] = True

                else:
                    status = f"| {file_size:>7.1f} KB"
                    found_models[filename] = True

            elif filename.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                # OPRAVENO: Bezpečné čtení JSON logu bez ohledu na to, zda je to dict nebo list
                if isinstance(log_data, dict):
                    n_epochs = len(log_data.get('history', {}).get('loss', []))
                    status = f"| {n_epochs} epochs"
                elif isinstance(log_data, list):
                    status = f"| {len(log_data)} záznamů (list)"
                else:
                    status = "| Neznámý formát JSON"

                found_models[filename] = log_data

            print(f"   {priority_icon} ✅ {filename:<35s} {status}")

        except Exception as e:
            err_msg = str(e)
            # Vlastní odchycení verze knihoven
            if "STACK_GLOBAL" in err_msg:
                err_msg = "Nekompatibilní verze knihoven (nutný retrénink!)"
            print(f"   🔴 ⚠️  {filename:<35s} | Chyba: {err_msg[:60]}")
            missing_models.append(filename)

    else:
        print(f"   {priority_icon} ❌ {filename:<35s} | CHYBÍ")
        missing_models.append(filename)

print(f"\n   📦 Celková velikost modelů: {total_size:.1f} KB ({total_size / 1024:.2f} MB)\n")

# ============================================================================
# 3. KONTROLA STEP3 TRAINING LOGU
# ============================================================================
print("3️⃣  KONTROLA STEP3 TRAINING LOGU")
print("-" * 80)

training_log_path = os.path.join(MODEL_DIR, 'training_log.json')
if os.path.exists(training_log_path):
    try:
        with open(training_log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)

        print(f"   ✅ Training log nalezen:\n")

        # OPRAVENO: Kontrola typu dat
        if isinstance(log, dict):
            if 'timestamp' in log:
                print(f"      Čas tréninku: {log['timestamp']}")

            if 'dataset_info' in log:
                info = log['dataset_info']
                print(f"      Zápasů: {info.get('total_fixtures', '?')}")
                print(f"      Trénink: {info.get('training_matches', '?')} zápasů")
                print(f"      Ratio: {info.get('ratio', '?')}")

            if 'model_performance' in log:
                perf = log['model_performance']
                print(f"\n      Model Performance:")
                for model, metrics in perf.items():
                    if isinstance(metrics, dict):
                        print(f"      • {model}:")
                        for metric, val in list(metrics.items())[:3]:
                            if isinstance(val, float):
                                print(f"        - {metric}: {val:.4f}")
                            else:
                                print(f"        - {metric}: {val}")
        elif isinstance(log, list):
            print(f"      Upozornění: Log je uložen jako seznam (list) s {len(log)} prvky.")
            print(f"      Skript neočekává tuto strukturu pro detailní výpis.")

        print()

    except Exception as e:
        print(f"   ⚠️  Chyba čtení logu: {e}\n")
else:
    print(f"   ⚠️  training_log.json není v {MODEL_DIR}\n")

# ============================================================================
# 4. KONTROLA SYNCHRONIZACE
# ============================================================================
print("4️⃣  KONTROLA SYNCHRONIZACE (Model ↔ Features ↔ DB)")
print("-" * 80)

if 'feature_cols.pkl' in found_models:
    n_features = found_models['feature_cols.pkl']
    print(f"   Model očekává: {n_features} features")

    if n_features == 20:
        print(f"   ✅ Standard: 20 features (RUN-OK)\n")
    else:
        print(f"   ⚠️  Nestandard: {n_features} features (možný problém)\n")
else:
    print(f"   ⚠️  feature_cols.pkl chybí (nebo selhalo načtení)\n")

if 'draw_threshold.pkl' in found_models:
    threshold = found_models['draw_threshold.pkl']
    print(f"   Draw threshold: {threshold:.3f}")
    if 0.28 <= threshold <= 0.35:
        print(f"   ✅ Normální rozsah (0.28-0.35)\n")
    else:
        print(f"   ⚠️  Mimo standard (měl by být 0.28-0.35)\n")
else:
    print(f"   ⚠️  draw_threshold.pkl chybí (nebo selhalo načtení)\n")

# ============================================================================
# 5. SOUHRN A DOPORUČENÍ
# ============================================================================
print("=" * 80)
print("📋 ZÁVĚR")
print("=" * 80)

critical_files = [
    'voting_classifier.pkl',
    'poisson_home_goals.pkl',
    'poisson_away_goals.pkl',
    'feature_cols.pkl',
    'draw_threshold.pkl'
]

missing_critical = [f for f in critical_files if f in missing_models]
n_found = len(found_models)

print(f"\n   Modelů načteno bez chyb: {n_found}/11 ✅")
print(f"   Chybných nebo chybějících: {len(missing_models)} ❌")

if not missing_critical and n_found >= 5:
    print("\n   ✅ VŠE V POŘÁDKU!")
    print("\n   Systém je připraven pro:\n")
    print("      🚀 step4  — final_prediction.py         (predikat zápasy)")
    print("      🚀 step5  — analyze_features.py         (SHAP analýza)")
    print("      🚀 step6  — fetch_odds.py               (API The Odds)")
    print("      🚀 step6b — manual_odds.py              (ruční kurzy)")
    print("      🚀 step7  — show_console_tips.py        (tipy + value)")
    print("      🚀 step8  — backtest_strategy.py        (backtest)")
    print("\n   ✅ Doporučuji: Spustit step4 → step7 pro nové tipy\n")

elif missing_critical:
    print(f"\n   ❌ KRITICKÉ CHYBY:\n")
    print(f"      Modely nelze načíst ({len(missing_critical)} selhalo):")
    for m in missing_critical:
        print(f"         • {m}")
    print(f"\n      🔧 FIX: Modely jsou z jiné verze Pythonu/knihoven.")
    print(f"         Spusť step3 pro jejich retrénink:")
    print(f"         python ml/step3_train_models.py\n")

else:
    print(f"\n   ⚠️  Chybí {len(missing_models)} doplňkových souborů")
    print(f"      (nejsou kritické, ale doporučuji je mít)\n")

print("=" * 80 + "\n")