"""
prediction_generator.py
Nástroj AI agenta pro generování finálních predikcí vč. 1X2, Neproher (1X/X2), Over/Under a BTTS.
UPGRADE:
  - Sjednocena pravidla a přísné prahy se strategy_backtester.py:
      * Neprohry (1X/X2) vyžadují šanci >= 75 %, max kurz 1.60, EV >= +8 % (poloviční vklad 1.75 %)
      * Over 2.5 vyžaduje celkové xG >= 3.15
      * Under 2.5 vyžaduje celkové xG <= 2.00
      * BTTS: Ano vyžaduje xG >= 1.30 u KAŽDÉHO týmu zvlášť a součet xG >= 2.90
  - Chronologické řazení podle datumu a bezpečné ošetření NaN kurzů
"""

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from scipy.stats import poisson
from ai_agent.config.market_values import MARKET_VALUES


class PredictionGenerator:
    def __init__(self, db_url=None):
        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")

        self.engine = create_engine(db_url)
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))

        self.models_loaded = False
        self.voting_clf = None
        self.xgb_clf = None
        self.reg_h, self.reg_a = None, None
        self.xgb_reg_h, self.xgb_reg_a = None, None
        self.trained_features = None
        self.mv_scaler = None

        # --- ŽIVÁ PRODUKČNÍ KONFIGURACE (Sjednoceno s backtesterem) ---
        self.BANKROLL = float(os.getenv("ODDS_BANKROLL", "10000"))
        self.KELLY_FRACTION = 0.25
        self.MAX_STAKE_PCT = 0.035             # 3.5% pro hlavní trh (1X2)
        self.MAX_STAKE_PCT_SECONDARY = 0.0175  # 1.75% pro vedlejší trhy (DC a Góly)

        self.MIN_PROB_THRESHOLD = 0.50
        self.MIN_VALUE_MARGIN_HOME = 0.04
        self.MIN_VALUE_MARGIN_AWAY = 0.06
        self.MAX_VALUE_MARGIN = 0.25
        self.MAX_ODDS_SINGLE = 2.50
        self.MIN_AKO_PROB = 0.65

        # --- PŘÍSNÉ PRAHY PRO NEPROHRY (1X a X2) ---
        self.DC_DRAW_WEIGHT = 0.80
        self.MIN_PROB_DC_VALUE = 0.75          # Value Single na 1X/X2 jen pro silné favority >= 75%
        self.MIN_MARGIN_DC_VALUE = 0.08        # EV aspoň +8%
        self.MAX_ODD_DC_VALUE = 1.60           # Max kurz 1.60
        self.MIN_AKO_PROB_DC = 0.84            # AKO tutovka na 1X/X2 jen při šanci >= 84%
        self.MIN_ODD_DC_AKO = 1.18

        # --- PŘÍSNÉ GÓLOVÉ FILTRY ---
        self.MIN_XG_OVER_25 = 3.15             # Over 2.5 jen při součtu xG >= 3.15
        self.MAX_XG_UNDER_25 = 2.00            # Under 2.5 jen při součtu xG <= 2.00
        self.MIN_PROB_GOALS_VALUE = 0.65
        self.MIN_MARGIN_GOALS_VALUE = 0.08
        self.MIN_AKO_PROB_GOALS = 0.74
        self.MIN_TEAM_XG_BTTS_YES = 1.30       # Oba týmy musí mít >= 1.30 xG
        self.MIN_TOTAL_XG_BTTS_YES = 2.90      # Součet xG >= 2.90
        self.MAX_TEAM_XG_BTTS_NO = 0.75

    def _load_models(self):
        if self.models_loaded:
            return True
        try:
            self.voting_clf = joblib.load(os.path.join(self.model_dir, "voting_classifier.pkl"))
            self.xgb_clf = joblib.load(os.path.join(self.model_dir, "xgb_classifier.pkl"))
            self.reg_h = joblib.load(os.path.join(self.model_dir, "poisson_home_goals.pkl"))
            self.reg_a = joblib.load(os.path.join(self.model_dir, "poisson_away_goals.pkl"))
            self.xgb_reg_h = joblib.load(os.path.join(self.model_dir, "xgb_home_goals.pkl"))
            self.xgb_reg_a = joblib.load(os.path.join(self.model_dir, "xgb_away_goals.pkl"))
            self.trained_features = joblib.load(os.path.join(self.model_dir, "feature_cols.pkl"))

            mv_path = os.path.join(self.model_dir, "market_value_scaler.pkl")
            if os.path.exists(mv_path):
                self.mv_scaler = joblib.load(mv_path)

            self.models_loaded = True
            return True
        except Exception as e:
            print(f"❌ Chyba při načítání modelů: {e}")
            return False

    @staticmethod
    def _safe_odd(odds_row, key: str) -> float:
        val = odds_row.get(key) if hasattr(odds_row, 'get') else None
        if val is None or pd.isna(val):
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_bookie(odds_row, key: str) -> str:
        if not hasattr(odds_row, 'get'):
            return "-"
        val = odds_row.get(key)
        if val is not None and not pd.isna(val) and str(val).strip():
            return str(val).strip()
        fallback = odds_row.get('book_1')
        if fallback is not None and not pd.isna(fallback) and str(fallback).strip():
            return str(fallback).strip()
        return "-"

    def _prepare_features(self, row):
        X = pd.DataFrame([row])
        if 'home_x_elo' not in X.columns or pd.isna(X['home_x_elo'].iloc[0]):
            X['home_x_elo'] = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)

        if 'market_value_diff_scaled' not in X.columns or pd.isna(X['market_value_diff_scaled'].iloc[0]):
            mv_h = MARKET_VALUES.get(row.get('home_team', ''), 200.0)
            mv_a = MARKET_VALUES.get(row.get('away_team', ''), 200.0)
            mv_diff = mv_h - mv_a
            if self.mv_scaler:
                mv_df = pd.DataFrame([[mv_diff]], columns=['market_value_diff'])
                X['market_value_diff_scaled'] = float(self.mv_scaler.transform(mv_df)[0][0])
            else:
                X['market_value_diff_scaled'] = mv_diff / 400.0

        if 'elo_x_market' not in X.columns or pd.isna(X['elo_x_market'].iloc[0]):
            elo_diff = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
            X['elo_x_market'] = elo_diff * float(X['market_value_diff_scaled'].iloc[0])

        dna_defaults = {
            'league_avg_goals': 2.70,
            'league_draw_rate': 0.25,
            'league_home_advantage': 0.44,
            'league_btts_rate': 0.53
        }
        for dna_col, default_val in dna_defaults.items():
            if dna_col not in X.columns or pd.isna(X[dna_col].iloc[0]) or float(X[dna_col].iloc[0]) == 0.0:
                X[dna_col] = default_val

        for col in self.trained_features:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.trained_features].astype(float)

    def _evaluate_bet(self, prob, odd, req_margin, min_ako_prob=None, is_secondary=False):
        ako_thr = min_ako_prob if min_ako_prob is not None else self.MIN_AKO_PROB

        if pd.isna(odd) or odd <= 1.0:
            if prob >= ako_thr:
                return "⏳ ČEKÁ NA KURZ (AKO šance)", 0, 0.0
            return "⏳ ČEKÁ NA KURZ", 0, 0.0

        ev = (prob * odd) - 1.0
        if ev > self.MAX_VALUE_MARGIN:
            return "❌ BEZ HODNOTY (Value Trap)", 0, ev
        elif ev >= req_margin and odd <= self.MAX_ODDS_SINGLE:
            kelly_full = ev / (odd - 1.0)
            cap = self.MAX_STAKE_PCT_SECONDARY if is_secondary else self.MAX_STAKE_PCT
            stake_pct = min(max(kelly_full * self.KELLY_FRACTION, 0), cap)
            stake = int(round((self.BANKROLL * stake_pct) / 50) * 50)
            if stake < 50:
                return "❌ BEZ HODNOTY (Nízký vklad)", 0, ev
            return "💎 VALUE SINGLE", stake, ev
        elif prob >= ako_thr:
            return "🔗 PREMIUM AKO", 0, ev
        else:
            return "❌ BEZ HODNOTY", 0, ev

    def generate_predictions(self, days_ahead=7) -> dict:
        if not self._load_models():
            return {"status": "error"}

        with self.engine.begin() as conn:
            df_fixtures = pd.read_sql(text(f"""
                SELECT * FROM prepared_fixtures
                WHERE match_date >= CURRENT_DATE
                  AND match_date <= CURRENT_DATE + INTERVAL '{days_ahead} days'
                ORDER BY match_date ASC, league ASC
            """), conn)

            try:
                odds_df = pd.read_sql(text("SELECT * FROM bookmaker_odds"), conn)
                odds_map = {row['fixture_id']: row for _, row in odds_df.iterrows()}
            except Exception:
                odds_map = {}

        if df_fixtures.empty:
            return {"status": "success", "predictions": pd.DataFrame()}

        results = []
        for _, row in df_fixtures.iterrows():
            try:
                mdate = pd.Timestamp(row['match_date']) if pd.notnull(row.get('match_date')) else None
                if mdate and mdate.month in (5, 6, 7, 8):
                    continue

                X_input = self._prepare_features(row)

                probs_v = self.voting_clf.predict_proba(X_input)[0]
                probs_x = self.xgb_clf.predict_proba(X_input)[0]
                probs_clf = 0.5 * probs_v + 0.5 * probs_x

                gh = float(np.clip((self.reg_h.predict(X_input)[0] + self.xgb_reg_h.predict(X_input)[0]) / 2, 0.1, 8.0))
                ga = float(np.clip((self.reg_a.predict(X_input)[0] + self.xgb_reg_a.predict(X_input)[0]) / 2, 0.1, 8.0))
                total_xg = gh + ga

                p1_poi, px_poi, p2_poi = 0.0, 0.0, 0.0
                p_under_25 = 0.0

                for h in range(10):
                    for a in range(10):
                        p = poisson.pmf(h, gh) * poisson.pmf(a, ga)
                        if h > a:
                            p1_poi += p
                        elif h == a:
                            px_poi += p
                        else:
                            p2_poi += p

                        if (h + a) < 2.5:
                            p_under_25 += p

                p1 = 0.5 * probs_clf[2] + 0.5 * p1_poi
                px = 0.5 * probs_clf[1] + 0.5 * px_poi
                p2 = 0.5 * probs_clf[0] + 0.5 * p2_poi
                tot = p1 + px + p2
                p1, px, p2 = p1 / tot, px / tot, p2 / tot

                p_1x = p1 + self.DC_DRAW_WEIGHT * px
                p_x2 = p2 + self.DC_DRAW_WEIGHT * px
                p_over_25 = 1.0 - p_under_25
                p_btts_yes = (1.0 - poisson.pmf(0, gh)) * (1.0 - poisson.pmf(0, ga))
                p_btts_no = 1.0 - p_btts_yes

                fid = row.get('fixture_id')
                o = odds_map.get(fid, {})
                xg_str = f"{gh:.2f}:{ga:.2f}"

                # 1. HLAVNÍ TRH (1X2)
                base_tip = "1" if p1 > p2 else "2"
                base_prob = p1 if base_tip == "1" else p2
                req_margin = self.MIN_VALUE_MARGIN_HOME if base_tip == "1" else self.MIN_VALUE_MARGIN_AWAY
                odd_base = self._safe_odd(o, 'odd_1') if base_tip == "1" else self._safe_odd(o, 'odd_2')
                book_base = self._safe_bookie(o, 'book_1' if base_tip == "1" else 'book_2')

                main_is_value = False
                if base_prob >= self.MIN_PROB_THRESHOLD:
                    sig, stake, ev = self._evaluate_bet(base_prob, odd_base, req_margin, is_secondary=False)
                    main_is_value = ("💎" in sig)
                    results.append({
                        'date': mdate, 'league': row.get('league'), 'home': row.get('home_team'), 'away': row.get('away_team'),
                        'market': '1X2', 'tip': base_tip, 'prob': base_prob, 'fair_odd': 1 / base_prob if base_prob > 0 else 0,
                        'actual_odd': odd_base, 'bookie': book_base if odd_base > 1.0 else '-',
                        'ev': ev if odd_base > 1.0 else 0.0, 'stake': stake, 'signal': sig, 'xg': xg_str
                    })

                # 2. DVOJITÁ ŠANCE / NEPROHRA (1X / X2)
                dc_tip = "1X" if p_1x >= p_x2 else "X2"
                dc_prob = p_1x if dc_tip == "1X" else p_x2
                dc_odd = self._safe_odd(o, 'odd_1x') if dc_tip == "1X" else self._safe_odd(o, 'odd_x2')
                dc_book = self._safe_bookie(o, 'book_1x' if dc_tip == "1X" else 'book_x2')
                dc_req_margin = self.MIN_VALUE_MARGIN_HOME if dc_tip == "1X" else self.MIN_VALUE_MARGIN_AWAY

                if dc_prob >= self.MIN_PROB_DC_VALUE and dc_odd >= self.MIN_ODD_DC_AKO:
                    dc_sig, dc_stake, dc_ev = self._evaluate_bet(
                        dc_prob, dc_odd, dc_req_margin, min_ako_prob=self.MIN_AKO_PROB_DC, is_secondary=True
                    )
                    if ("💎" in dc_sig and not main_is_value) or "🔗" in dc_sig:
                        results.append({
                            'date': mdate, 'league': row.get('league'), 'home': row.get('home_team'), 'away': row.get('away_team'),
                            'market': 'DC', 'tip': dc_tip, 'prob': dc_prob, 'fair_odd': 1 / dc_prob if dc_prob > 0 else 0,
                            'actual_odd': dc_odd, 'bookie': dc_book if dc_odd > 1.0 else '-',
                            'ev': dc_ev if dc_odd > 1.0 else 0.0, 'stake': dc_stake, 'signal': dc_sig, 'xg': xg_str
                        })

                # 3. OVER / UNDER 2.5 (S PŘÍSNÝM GÓLOVÝM POLŠTÁŘEM >= 3.15 / <= 2.00)
                ou_tip, ou_book = None, "-"
                if total_xg >= self.MIN_XG_OVER_25 and p_over_25 >= self.MIN_PROB_GOALS_VALUE:
                    ou_tip = "Over 2.5"
                    ou_prob = p_over_25
                    ou_odd = self._safe_odd(o, 'odd_over_25')
                    ou_book = self._safe_bookie(o, 'book_over_25')
                elif total_xg <= self.MAX_XG_UNDER_25 and p_under_25 >= self.MIN_PROB_GOALS_VALUE:
                    ou_tip = "Under 2.5"
                    ou_prob = p_under_25
                    ou_odd = self._safe_odd(o, 'odd_u25')
                    ou_book = self._safe_bookie(o, 'book_under_25')

                if ou_tip is not None:
                    ou_sig, ou_stake, ou_ev = self._evaluate_bet(ou_prob, ou_odd, self.MIN_VALUE_MARGIN_HOME, is_secondary=True)
                    if "💎" in ou_sig or "🔗" in ou_sig or "AKO šance" in ou_sig:
                        results.append({
                            'date': mdate, 'league': row.get('league'), 'home': row.get('home_team'), 'away': row.get('away_team'),
                            'market': 'Góly', 'tip': ou_tip, 'prob': ou_prob, 'fair_odd': 1 / ou_prob if ou_prob > 0 else 0,
                            'actual_odd': ou_odd, 'bookie': ou_book if ou_odd > 1.0 else '-',
                            'ev': ou_ev if ou_odd > 1.0 else 0.0, 'stake': ou_stake, 'signal': ou_sig, 'xg': xg_str
                        })

                # 4. BTTS - OBA TÝMY DAJÍ GÓL
                btts_tip, btts_book = None, "-"
                if gh >= self.MIN_TEAM_XG_BTTS_YES and ga >= self.MIN_TEAM_XG_BTTS_YES and total_xg >= self.MIN_TOTAL_XG_BTTS_YES:
                    btts_tip = "BTTS: Ano"
                    btts_prob = p_btts_yes
                    btts_odd = self._safe_odd(o, 'odd_btts_yes')
                    btts_book = self._safe_bookie(o, 'book_btts_yes')
                elif min(gh, ga) <= self.MAX_TEAM_XG_BTTS_NO and total_xg <= 2.40 and p_btts_no >= self.MIN_PROB_GOALS_VALUE:
                    btts_tip = "BTTS: Ne"
                    btts_prob = p_btts_no
                    btts_odd = self._safe_odd(o, 'odd_btts_no')
                    btts_book = self._safe_bookie(o, 'book_btts_no')

                if btts_tip is not None:
                    btts_sig, btts_stake, btts_ev = self._evaluate_bet(btts_prob, btts_odd, self.MIN_VALUE_MARGIN_HOME, is_secondary=True)
                    if "💎" in btts_sig or "🔗" in btts_sig or "AKO šance" in btts_sig:
                        results.append({
                            'date': mdate, 'league': row.get('league'), 'home': row.get('home_team'), 'away': row.get('away_team'),
                            'market': 'BTTS', 'tip': btts_tip, 'prob': btts_prob, 'fair_odd': 1 / btts_prob if btts_prob > 0 else 0,
                            'actual_odd': btts_odd, 'bookie': btts_book if btts_odd > 1.0 else '-',
                            'ev': btts_ev if btts_odd > 1.0 else 0.0, 'stake': btts_stake, 'signal': btts_sig, 'xg': xg_str
                        })

            except Exception as e:
                print(f"  ⚠️ Chyba při predikci zápasu {row.get('home_team')} vs {row.get('away_team')}: {e}")

        df_results = pd.DataFrame(results)
        return {"status": "success", "predictions": df_results}

    def print_console_report(self, result_dict: dict):
        if result_dict.get("status") != "success":
            return

        df = result_dict.get("predictions")
        if df is None or df.empty:
            print("\n📭 Aktuálně žádné zápasy nesplňují přísná kritéria (nebo je léto).")
            return

        print("\n" + "=" * 162)
        print(f"  {'DATUM':<8} {'LIGA':<4} {'ZÁPAS':<36} {'TRH':<5} {'TIP':<10} {'ŠANCE':>6} {'FÉR':>5} {'KURZ':>6} {'SÁZKOVKA':^15} {'EDGE':>7} {'VKLAD':>8}  {'xG':^9}  SIGNÁL")
        print("=" * 162)

        value_df = df[df['signal'] == "💎 VALUE SINGLE"].sort_values(by=['date', 'ev'], ascending=[True, False])
        ako_df = df[df['signal'] == "🔗 PREMIUM AKO"].sort_values(by=['date', 'prob'], ascending=[True, False])
        info_df = df[df['signal'].str.startswith("❌")].sort_values(by=['date', 'league'], ascending=[True, True])
        wait_df = df[df['signal'].str.startswith("⏳")].sort_values(by=['date', 'league'], ascending=[True, True])

        groups = [
            ("💎 DOPORUČENÉ SÁZKY (Seřazeno podle datumu a matematické výhody)", value_df),
            ("🔗 PREMIUM AKO MATERIÁL (Tutovky pro spojení do tiketu)", ako_df),
            ("ℹ️ STATISTICKÉ TIPY (Hlavní predikce bez sázkové hodnoty - Pouze pro info)", info_df),
            ("⏳ ČEKÁRNA (Tipy čekající na vypsání kurzů sázkovkou)", wait_df)
        ]

        for group_label, group_df in groups:
            if group_df.empty:
                continue
            print(f"\n  {group_label}")
            print("  " + "-" * 160)

            for _, r in group_df.iterrows():
                date_str = pd.Timestamp(r['date']).strftime("%d.%m.") if pd.notnull(r['date']) else "???"
                match_str = f"{r['home']} - {r['away']}"[:36]
                prob_str = f"{r['prob']*100:.1f}%"
                fair_str = f"{r['fair_odd']:.2f}"

                odd_str = f"{r['actual_odd']:.2f}" if r['actual_odd'] > 1.0 else " - "
                bookie_str = str(r.get('bookie', '-'))[:15] if r['actual_odd'] > 1.0 else "-"
                ev_str = f"{r['ev']*100:+.1f}%" if r['actual_odd'] > 1.0 else " - "
                stake_str = f"{r['stake']} Kč" if r['stake'] > 0 else " - "

                print(f"  {date_str:<8} {r['league']:<4} {match_str:<36} {r['market']:<5} {r['tip']:<10} {prob_str:>6} {fair_str:>5} "
                      f"{odd_str:>6} {bookie_str:^15} {ev_str:>7} {stake_str:>8}  {r['xg']:^9}  {r['signal']}")

        print("\n" + "=" * 162)
        print(f"  💰 Aktuální nastavení Bankrollu: {self.BANKROLL:,.0f} Kč (Quarter Kelly | Max sázka: {self.MAX_STAKE_PCT*100:.1f} %)")
        print(f"  🛡️ Gólové filtry: Over 2.5 (xG >= {self.MIN_XG_OVER_25:.1f}) | Under 2.5 (xG <= {self.MAX_XG_UNDER_25:.1f}) | BTTS Ano (oba >= {self.MIN_TEAM_XG_BTTS_YES:.2f} xG)")