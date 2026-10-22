"""
system_debugger.py
Nástroj AI agenta pro komplexní diagnostiku zdraví modelů, schématu DB a připojení.
UPGRADE:
  - Doplněna kontrola tabulky match_statistics (včetně detekce prázdných řádků)
  - Kontrola přítomnosti Ligové DNA v prepared_datasets
  - Křížová kontrola počtu features mezi feature_cols.pkl a natrénovanými modely
"""

import os
import joblib
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv


class SystemDebugger:
    def __init__(self, db_url=None):
        load_dotenv()
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.model_dir = os.path.join(self.base_dir, "models")

        self.models_to_check = {
            'voting_classifier.pkl': ('Voting Ensemble (RF+GBM+LR)', 'CRITICAL'),
            'best_classifier.pkl': ('Best Classifier', 'CRITICAL'),
            'xgb_classifier.pkl': ('XGBoost Classifier', 'IMPORTANT'),
            'poisson_home_goals.pkl': ('Poisson Home Goals', 'CRITICAL'),
            'poisson_away_goals.pkl': ('Poisson Away Goals', 'CRITICAL'),
            'xgb_home_goals.pkl': ('XGBoost Home Goals', 'SECONDARY'),
            'xgb_away_goals.pkl': ('XGBoost Away Goals', 'SECONDARY'),
            'feature_cols.pkl': ('Feature Columns', 'CRITICAL'),
            'draw_threshold.pkl': ('Draw Threshold', 'CRITICAL'),
            'market_value_scaler.pkl': ('Market Value Scaler', 'IMPORTANT'),
            'training_log.json': ('Training Log', 'INFO'),
        }

    def run_diagnostics(self):
        print("\n" + "=" * 80)
        print("🕵️  KOMPLEXNÍ DIAGNOSTIKA AGENTA")
        print("=" * 80)

        db_issues = self._check_database()
        model_issues = self._check_models()

        print("\n" + "=" * 80)
        print("📋 ZÁVĚR")
        print("=" * 80)

        all_issues = db_issues + model_issues
        if all_issues:
            print("   ⚠️  ZJIŠTĚNÉ PROBLÉMY K ŘEŠENÍ:")
            for issue in all_issues:
                print(f"      • {issue}")
        else:
            print("   ✅ VŠE V POŘÁDKU! Databáze i AI modely jsou plně synchronizované a připravené.")

    def _check_database(self) -> list:
        print("\n1️⃣  KONTROLA DATABÁZE")
        print("-" * 80)
        issues = []

        if not self.db_url:
            print("   ❌ DATABASE_URL chybí v prostředí.")
            issues.append("Chybí DATABASE_URL v .env souboru.")
            return issues

        try:
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("   ✅ Připojení k databázi: OK\n")

            inspector = inspect(engine)
            tables = inspector.get_table_names()

            print(f"   📋 Tabulky v DB: {len(tables)}")
            core_tables = ['fixtures', 'match_statistics', 'prepared_datasets', 'prepared_fixtures', 'bookmaker_odds']
            for table in core_tables:
                if table in tables:
                    count = pd.read_sql(text(f"SELECT COUNT(*) as cnt FROM {table}"), engine)
                    rows = count['cnt'].values[0]
                    print(f"      ✅ {table:<25s} | {rows:>6,} řádků")
                else:
                    print(f"      ❌ {table:<25s} | CHYBÍ")
                    issues.append(f"V databázi chybí tabulka '{table}'.")

            # Kontrola prázdných statistik v match_statistics
            if 'match_statistics' in tables:
                null_stats = pd.read_sql(
                    text("SELECT COUNT(*) as cnt FROM match_statistics WHERE possession_home IS NULL OR possession_home = 0"),
                    engine
                )['cnt'].values[0]
                if null_stats > 0:
                    print(f"\n   ⚠️  V match_statistics je {null_stats} záznamů s nulovým/NULL držením míče.")
                else:
                    print(f"\n   ✅ Kvalita match_statistics: Žádné prázdné (NULL/0) záznamy držení míče.")

            # Kontrola přítomnosti Ligové DNA v prepared_datasets
            if 'prepared_datasets' in tables:
                pd_cols = {c['name'] for c in inspector.get_columns('prepared_datasets')}
                dna_cols = {'league_avg_goals', 'league_draw_rate', 'league_home_advantage', 'league_btts_rate'}
                missing_dna = dna_cols - pd_cols
                if missing_dna:
                    print(f"   ℹ️  Ligová DNA zatím není v prepared_datasets (spusťte prepare_datasets).")
                    issues.append("V tabulce prepared_datasets zatím chybí sloupce Ligové DNA -> spusťte prepare_datasets().")
                else:
                    print(f"   🧬 Ligová DNA v prepared_datasets: AKTIVNÍ (všechny 4 sloupce přítomny).")

        except Exception as e:
            print(f"   ❌ Chyba databáze: {e}")
            issues.append(f"Chyba připojení k DB: {e}")

        return issues

    def _check_models(self) -> list:
        print("\n2️⃣  KONTROLA MODELŮ (AI Paměť)")
        print("-" * 80)
        issues = []

        if not os.path.exists(self.model_dir):
            print(f"   ❌ Složka modelů neexistuje: {self.model_dir}")
            issues.append("Složka models neexistuje -> spusťte retrain_models().")
            return issues

        total_size = 0
        expected_n_features = None
        voting_n_features = None

        # Nejprve načteme počet features z feature_cols.pkl
        fc_path = os.path.join(self.model_dir, 'feature_cols.pkl')
        if os.path.exists(fc_path):
            try:
                expected_n_features = len(joblib.load(fc_path))
            except Exception:
                pass

        for filename, (desc, priority) in self.models_to_check.items():
            filepath = os.path.join(self.model_dir, filename)
            icon = '🔴' if priority == 'CRITICAL' else '🟡' if priority == 'IMPORTANT' else '🟢'

            if os.path.exists(filepath):
                try:
                    size = os.path.getsize(filepath) / 1024
                    total_size += size

                    status = f"| {size:>7.1f} KB"
                    if filename == 'feature_cols.pkl':
                        obj = joblib.load(filepath)
                        dna_in_fc = sum(1 for c in obj if str(c).startswith('league_'))
                        status = f"| {len(obj)} features (z toho {dna_in_fc}× Ligová DNA)"
                    elif filename == 'draw_threshold.pkl':
                        obj = joblib.load(filepath)
                        status = f"| threshold={obj:.3f}"
                    elif filename == 'voting_classifier.pkl':
                        obj = joblib.load(filepath)
                        if hasattr(obj, 'n_features_in_'):
                            voting_n_features = obj.n_features_in_

                    print(f"   {icon} ✅ {filename:<30s} {status}")
                except Exception as e:
                    print(f"   🔴 ⚠️  {filename:<30s} | Chyba: {str(e)[:40]}")
                    if priority == 'CRITICAL':
                        issues.append(f"Poškozený soubor modelu: {filename}")
            else:
                print(f"   {icon} ❌ {filename:<30s} | CHYBÍ")
                if priority == 'CRITICAL':
                    issues.append(f"Chybí kritický model: {filename} -> spusťte retrain_models().")

        if expected_n_features and voting_n_features and expected_n_features != voting_n_features:
            print(f"\n   ⚠️  NESOULAD FEATURES: feature_cols.pkl má {expected_n_features} sloupců, ale model čeká {voting_n_features}!")
            issues.append("Nesoulad počtu features mezi feature_cols.pkl a modely -> spusťte retrain_models().")

        print(f"\n   📦 Celková velikost modelů: {total_size / 1024:.2f} MB")
        return issues