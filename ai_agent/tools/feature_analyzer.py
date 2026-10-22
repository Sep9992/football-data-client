"""
feature_analyzer.py
Nástroj AI agenta pro analýzu predikcí pomocí SHAP values.
Vychází ze step5_analyze_features.py (v2).
"""

import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from ai_agent.config.market_values import MARKET_VALUES

# Zabrání padání na serveru/bez GUI
matplotlib.use('Agg')


class FeatureAnalyzer:
    def __init__(self, db_url=None):
        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")

        self.engine = create_engine(db_url)
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
        self.shap_dir = os.path.join(self.model_dir, "shap_reports")
        os.makedirs(self.shap_dir, exist_ok=True)

        self.models_loaded = False
        self.voting_clf = None
        self.xgb_clf = None
        self.poisson_h = None
        self.poisson_a = None
        self.xgb_reg_h = None
        self.xgb_reg_a = None
        self.feature_cols = None
        self.draw_threshold = 0.22
        self.mv_scaler = None

    def _load_models(self):
        """Líné načtení modelů pro analýzu."""
        if self.models_loaded:
            return True

        try:
            self.voting_clf = joblib.load(os.path.join(self.model_dir, "voting_classifier.pkl"))
            self.xgb_clf = joblib.load(os.path.join(self.model_dir, "xgb_classifier.pkl"))
            self.poisson_h = joblib.load(os.path.join(self.model_dir, "poisson_home_goals.pkl"))
            self.poisson_a = joblib.load(os.path.join(self.model_dir, "poisson_away_goals.pkl"))
            self.xgb_reg_h = joblib.load(os.path.join(self.model_dir, "xgb_home_goals.pkl"))
            self.xgb_reg_a = joblib.load(os.path.join(self.model_dir, "xgb_away_goals.pkl"))
            self.feature_cols = joblib.load(os.path.join(self.model_dir, "feature_cols.pkl"))

            thr_path = os.path.join(self.model_dir, "draw_threshold.pkl")
            if os.path.exists(thr_path):
                self.draw_threshold = joblib.load(thr_path)

            mv_path = os.path.join(self.model_dir, "market_value_scaler.pkl")
            if os.path.exists(mv_path):
                self.mv_scaler = joblib.load(mv_path)

            self.models_loaded = True
            return True
        except Exception as e:
            print(f"❌ Chyba při načítání modelů pro analýzu: {e}")
            return False

    def _prepare_features(self, row):
        """Z řádku prepared_fixtures doplní chybějící odvozené features včetně Ligové DNA."""
        X = pd.DataFrame([row])

        if 'home_x_elo' not in X.columns or pd.isna(X['home_x_elo'].iloc[0]):
            elo_diff_val = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
            X['home_x_elo'] = elo_diff_val

        if 'market_value_diff_scaled' not in X.columns or pd.isna(X['market_value_diff_scaled'].iloc[0]):
            mv_h = MARKET_VALUES.get(row.get('home_team', ''), 200.0)
            mv_a = MARKET_VALUES.get(row.get('away_team', ''), 200.0)
            mv_diff = mv_h - mv_a
            if self.mv_scaler is not None:
                mv_df = pd.DataFrame([[mv_diff]], columns=['market_value_diff'])
                X['market_value_diff_scaled'] = float(self.mv_scaler.transform(mv_df)[0][0])
            else:
                X['market_value_diff_scaled'] = mv_diff / 400.0

        if 'elo_x_market' not in X.columns or pd.isna(X['elo_x_market'].iloc[0]):
            elo_diff = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
            X['elo_x_market'] = elo_diff * float(X['market_value_diff_scaled'].iloc[0])

        # Bezpečné fallbacky pro Ligovou DNA
        dna_defaults = {
            'league_avg_goals': 2.70,
            'league_draw_rate': 0.25,
            'league_home_advantage': 0.44,
            'league_btts_rate': 0.53
        }
        for dna_col, default_val in dna_defaults.items():
            if dna_col not in X.columns or pd.isna(X[dna_col].iloc[0]) or float(X[dna_col].iloc[0]) == 0.0:
                X[dna_col] = default_val

        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0

        return X[self.feature_cols].astype(float)

    def _generate_shap_plot(self, fixture_id, match_name, X_input):
        """Vytvoří SHAP bar plot pro jeden zápas pomocí XGBoost pipeline."""
        try:
            prep_steps = self.xgb_clf[:-1]
            X_transformed = prep_steps.transform(X_input)
            xgb_model = self.xgb_clf.named_steps['xgb']

            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                vals = shap_values[2][0]
            else:
                if shap_values.ndim == 3:
                    vals = shap_values[0, :, 2]
                else:
                    vals = shap_values.flatten()

            df_shap = pd.DataFrame({"feat": self.feature_cols, "val": vals})
            df_shap["abs_val"] = df_shap["val"].abs()
            df_shap = df_shap.sort_values("abs_val", ascending=False).head(12).sort_values("val")

            colors = ['#e74c3c' if x < 0 else '#27ae60' for x in df_shap['val']]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df_shap['feat'], df_shap['val'], color=colors, edgecolor='white', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.set_title(f"SHAP – Vliv faktorů na výhru domácích\n{match_name}", fontsize=12, pad=12)
            ax.set_xlabel("SHAP hodnota (vliv na P(výhra domácích))")

            for i, (val, feat) in enumerate(zip(df_shap['val'], df_shap['feat'])):
                ax.text(val + (0.001 if val >= 0 else -0.001), i,
                        f" {val:+.3f}", va='center',
                        ha='left' if val >= 0 else 'right', fontsize=8)

            plt.tight_layout()

            safe_id = str(fixture_id).replace('/', '_')
            file_path = os.path.join(self.shap_dir, f"shap_{safe_id}.png")
            plt.savefig(file_path, dpi=120)
            plt.close()

            return vals, os.path.basename(file_path)

        except Exception as e:
            print(f"  ⚠️  SHAP chyba pro {match_name}: {e}")
            return None, None

    def analyze_upcoming_matches(self, days_ahead=14) -> dict:
        """Provede hlubokou analýzu nadcházejících zápasů a vygeneruje SHAP."""
        if not self._load_models():
            return {"status": "error", "message": "Modely nejsou dostupné."}

        with self.engine.begin() as conn:
            df_pred = pd.read_sql(text(f"""
                SELECT * FROM prepared_fixtures
                WHERE match_date >= CURRENT_DATE
                  AND match_date <= CURRENT_DATE + INTERVAL '{days_ahead} days'
                ORDER BY match_date ASC, league ASC
            """), conn)

        if df_pred.empty:
            return {"status": "success", "results": []}

        results = []
        all_shap_vals = []

        for _, row in df_pred.iterrows():
            home = row.get('home_team', '?')
            away = row.get('away_team', '?')
            match_name = f"[{row.get('league', '?')}] {home} vs {away}"
            fixture_id = row.get('fixture_id', f"{home}_{row.get('match_date', '')}")

            try:
                X = self._prepare_features(row)

                # Voting probabilities
                probs_v = self.voting_clf.predict_proba(X)[0]
                pv_a, pv_x, pv_h = probs_v[0], probs_v[1], probs_v[2]

                # XGBoost probabilities
                probs_xgb = self.xgb_clf.predict_proba(X)[0]
                px_a, px_x, px_h = probs_xgb[0], probs_xgb[1], probs_xgb[2]

                # Blend logic
                p1 = 0.5 * pv_h + 0.5 * px_h
                px = 0.5 * pv_x + 0.5 * px_x
                p2 = 0.5 * pv_a + 0.5 * px_a
                total = p1 + px + p2
                p1, px, p2 = p1 / total, px / total, p2 / total

                # Shoda a diference
                v_pred = np.argmax([pv_a, pv_x, pv_h])
                xgb_pred = np.argmax([px_a, px_x, px_h])
                shoda = "✅" if v_pred == xgb_pred else "❌"
                max_diff = max(abs(pv_h - px_h), abs(pv_x - px_x), abs(pv_a - px_a))
                confidence = max(p1, px, p2) / (1 / 3)

                # SHAP grafy
                shap_vals, shap_file = self._generate_shap_plot(fixture_id, match_name, X)
                if shap_vals is not None:
                    all_shap_vals.append(shap_vals)

                results.append({
                    'match': match_name, 'shoda': shoda,
                    'pv_h': pv_h, 'pv_x': pv_x, 'pv_a': pv_a,
                    'px_h': px_h, 'px_x': px_x, 'px_a': px_a,
                    'max_diff': max_diff, 'confidence': confidence,
                    'shap_file': shap_file
                })

            except Exception as e:
                print(f"❌ Chyba při analýze {match_name}: {e}")

        # Globální SHAP
        if all_shap_vals:
            self._generate_shap_summary(all_shap_vals)

        return {"status": "success", "results": results}

    def _generate_shap_summary(self, all_shap_vals):
        """Vygeneruje globální graf průměrné důležitosti."""
        try:
            matrix = np.vstack(all_shap_vals)
            mean_abs = np.abs(matrix).mean(axis=0)
            df = pd.DataFrame({"feat": self.feature_cols, "importance": mean_abs})
            df = df.sort_values("importance", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(df['feat'], df['importance'], color='#3498db', edgecolor='white')
            ax.set_title("SHAP – Globální importance features\n(průměr |SHAP| přes všechny zápasy)", fontsize=12,
                         pad=12)
            ax.set_xlabel("Průměrná |SHAP| hodnota")
            plt.tight_layout()

            plt.savefig(os.path.join(self.shap_dir, "shap_global_summary.png"), dpi=120)
            plt.close()
        except Exception as e:
            print(f"⚠️ Global SHAP chyba: {e}")

    def print_console_report(self, analysis_result: dict):
        """Vytiskne analytický report do konzole."""
        if analysis_result.get("status") != "success":
            return

        results = analysis_result.get("results", [])
        if not results:
            print("📭 Žádné zápasy k analýze.")
            return

        print("\n" + "=" * 110)
        print(f"{'Zápas':<40} | {'Voting':^20} | {'XGBoost':^20} | {'Shoda':^6} | {'Conf'}")
        print(f"{'':40} | {'1    X    2':^20} | {'1    X    2':^20} | {'':6} |")
        print("-" * 110)

        for r in results:
            print(f"{r['match']:<40} | {r['pv_h'] * 100:>4.0f}% {r['pv_x'] * 100:>4.0f}% {r['pv_a'] * 100:>4.0f}% |"
                  f" {r['px_h'] * 100:>4.0f}% {r['px_x'] * 100:>4.0f}% {r['px_a'] * 100:>4.0f}% |"
                  f"  {r['shoda']}   | {r['confidence']:.2f}x")
        print("=" * 110)

        shody = sum(1 for r in results if r['shoda'] == "✅")
        print(f"\n📊 SOUHRN ANALÝZY")
        print(f"  Shoda modelů:   {shody}/{len(results)} zápasů ({shody / len(results) * 100:.0f}%)")

        uncertain = sorted(results, key=lambda x: x['max_diff'], reverse=True)[:3]
        print(f"\n  ⚠️  Nejistější predikce (největší rozdíl Voting vs XGBoost):")
        for r in uncertain:
            print(f"     {r['match']:<40} diff={r['max_diff']:.1%}")

        confident = sorted(results, key=lambda x: x['confidence'], reverse=True)[:3]
        print(f"\n  🎯 Nejsilnější tipy (nejvyšší shoda a jistota):")
        for r in confident:
            print(f"     {r['match']:<40} conf={r['confidence']:.2f}x")

        print(f"\n  💡 SHAP grafy byly uloženy do: {self.shap_dir}")