"""
strategy_backtester.py
Nástroj AI agenta pro otestování strategie na historických datech.
ARCHITEKTURA: Asymmetric Value Singles + Premium AKO Tips (1X2 + přísné 1X/X2 + Góly).
UPGRADE:
  - Obnoven čistý 5zápasový xG Poisson (avg_expected_goals_last5) jako nezávislá kontrola formy vůči ELO
  - Ponechán 80% filtr pro Neprohry (1X/X2 >= 75 %, korekce remízy 0.80 * px, max kurz 1.60)
  - Poloviční max. expozice (1.75 % bankrollu) pro vedlejší trhy (1X/X2 a Góly), plná (3.5 %) pro hlavní trh 1X2
"""

import os
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from scipy.stats import poisson
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


class StrategyBacktester:
    def __init__(self, db_url=None):
        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")

        self.engine = create_engine(db_url)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.model_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # --- QUANTITATIVE CONFIGURATION ---
        self.START_BANKROLL = 10_000
        self.KELLY_FRACTION = 0.25             # Quarter Kelly
        self.MAX_STAKE_PCT = 0.035             # Max 3.5% bankrollu pro hlavní trh (1X2)
        self.MAX_STAKE_PCT_SECONDARY = 0.0175  # Max 1.75% bankrollu pro vedlejší trhy (1X/X2 a Góly)

        self.MIN_PROB_THRESHOLD = 0.50         # Sázíme jen favority > 50% (pro čisté 1/2)
        self.MAX_ODDS_SINGLE = 2.50            # Zákaz obřích kurzů pro singly
        self.MAX_VALUE_MARGIN = 0.25           # EV nesmí přesáhnout +25% (Value Traps)

        # Asymetrická Value (Ochrana proti selhání favoritů venku)
        self.MIN_VALUE_MARGIN_HOME = 0.04      # Domácí favorit (1) potřebuje výhodu +4%
        self.MIN_VALUE_MARGIN_AWAY = 0.06      # Hostující favorit (2) musí mít výhodu +6%

        self.MIN_AKO_PROB = 0.65               # AKO TIP (1/2): Zápas musí mít šanci aspoň 65%

        # --- PŘÍSNÉ PRAHY PRO NEPROHRY (1X a X2) - 80% WIN RATE NASTAVENÍ ---
        self.DC_DRAW_WEIGHT = 0.80             # Korekce přifouknuté remízy z balanced vah modelů
        self.MIN_PROB_DC_VALUE = 0.75          # Value Single na 1X/X2 jen pro silné favority >= 75%
        self.MIN_MARGIN_DC_VALUE = 0.08        # Požadovaná výhoda (EV) pro 1X/X2 aspoň +8%
        self.MAX_ODD_DC_VALUE = 1.60           # Max. kurz pro 1X/X2 (nad 1.60 jde o vyrovnané remízové pasti)
        self.MIN_AKO_PROB_DC = 0.84            # AKO TIP pro 1X/X2: vyžadujeme aspoň 84% šanci
        self.MIN_ODD_DC_AKO = 1.18             # Ignorujeme mikro-kurzy pod 1.18

        # --- PŘÍSNÉ GÓLOVÉ FILTRY (Bezpečnostní polštář +/- 0.5 gólu) ---
        self.MIN_XG_OVER_25 = 3.15             # Over 2.5 jen když součet xG >= 3.15 gólu
        self.MAX_XG_UNDER_25 = 2.00            # Under 2.5 jen když součet xG <= 2.00 gólu
        self.MIN_PROB_GOALS_VALUE = 0.65       # Min. pravděpodobnost pro Value Single na góly
        self.MIN_MARGIN_GOALS_VALUE = 0.08     # Min. EV pro Value Single na góly (+8%)
        self.MIN_AKO_PROB_GOALS = 0.74         # AKO TIP na góly jen při šanci >= 74%
        self.MIN_TEAM_XG_BTTS_YES = 1.30       # BTTS: Ano jen když OBA týmy mají >= 1.30 xG
        self.MIN_TOTAL_XG_BTTS_YES = 2.90      # a zároveň celkový součet xG >= 2.90 gólu
        self.MAX_TEAM_XG_BTTS_NO = 0.75        # BTTS: Ne jen když aspoň 1 tým má <= 0.75 xG

        self.INITIAL_TRAIN_PCT = 0.60
        self.RETRAIN_STEP = 10

        self._cached_main_result = None
        self._cached_summer_result = None

    def _build_voting_pipeline(self):
        prep = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        rf  = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
        lr  = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=42)
        voting = VotingClassifier(estimators=[('rf', rf), ('gbm', gbm), ('lr', lr)], voting='soft')
        return Pipeline([('prep', prep), ('clf', voting)])

    def _build_xgb_pipeline(self):
        prep = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', objective='multi:softprob', num_class=3,
            random_state=42, n_jobs=-1
        )
        return Pipeline([('prep', prep), ('xgb', xgb)])

    @staticmethod
    def _safe_float(val) -> float:
        if val is None or pd.isna(val):
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    def _get_real_odds(self, row, odds_map):
        fid = row.get('fixture_id')
        if fid in odds_map:
            o = odds_map[fid]
            return {
                'odd_1': self._safe_float(o.get('odd_1')),
                'odd_x': self._safe_float(o.get('odd_x')),
                'odd_2': self._safe_float(o.get('odd_2')),
                'odd_1x': self._safe_float(o.get('odd_1x')),
                'odd_x2': self._safe_float(o.get('odd_x2')),
                'odd_o25': self._safe_float(o.get('odd_over_25')),
                'odd_u25': self._safe_float(o.get('odd_under_25')),
                'odd_btts_y': self._safe_float(o.get('odd_btts_yes')),
                'odd_btts_n': self._safe_float(o.get('odd_btts_no')),
            }
        return {
            'odd_1': 0.0, 'odd_x': 0.0, 'odd_2': 0.0,
            'odd_1x': 0.0, 'odd_x2': 0.0,
            'odd_o25': 0.0, 'odd_u25': 0.0,
            'odd_btts_y': 0.0, 'odd_btts_n': 0.0
        }

    def _calc_kelly_stake(self, ev, odd, current_bankroll, is_secondary=False):
        kelly_full = ev / (odd - 1.0)
        cap = self.MAX_STAKE_PCT_SECONDARY if is_secondary else self.MAX_STAKE_PCT
        stake_pct = min(max(kelly_full * self.KELLY_FRACTION, 0), cap)
        return current_bankroll * stake_pct

    def _evaluate_bet(self, p1, px, p2, p_over25, p_under25, p_btts_y, p_btts_n, gh, ga, odds, current_bankroll):
        total_xg = gh + ga

        # --- Příprava 1X2 (Hlavní trh) ---
        base_bet = "1" if p1 >= p2 else "2"
        base_prob = p1 if base_bet == "1" else p2
        base_odd = odds['odd_1'] if base_bet == "1" else odds['odd_2']
        base_req_margin = self.MIN_VALUE_MARGIN_HOME if base_bet == "1" else self.MIN_VALUE_MARGIN_AWAY
        base_ev = (base_prob * base_odd) - 1.0 if base_odd > 1.0 else -1.0

        # --- Příprava DC (1X / X2 s korekcí přifouknuté remízy) ---
        p_1x = p1 + self.DC_DRAW_WEIGHT * px
        p_x2 = p2 + self.DC_DRAW_WEIGHT * px
        dc_bet = "1X" if p_1x >= p_x2 else "X2"
        dc_prob = p_1x if dc_bet == "1X" else p_x2
        dc_odd = odds['odd_1x'] if dc_bet == "1X" else odds['odd_x2']
        dc_ev = (dc_prob * dc_odd) - 1.0 if dc_odd > 1.0 else -1.0

        # --- Příprava Over/Under 2.5 ---
        ou_bet, ou_prob, ou_odd, ou_ev = None, 0.0, 0.0, -1.0
        if total_xg >= self.MIN_XG_OVER_25 and p_over25 >= self.MIN_PROB_GOALS_VALUE:
            ou_bet, ou_prob, ou_odd = "Over 2.5", p_over25, odds['odd_o25']
        elif total_xg <= self.MAX_XG_UNDER_25 and p_under25 >= self.MIN_PROB_GOALS_VALUE:
            ou_bet, ou_prob, ou_odd = "Under 2.5", p_under25, odds['odd_u25']
        if ou_bet and ou_odd > 1.0:
            ou_ev = (ou_prob * ou_odd) - 1.0

        # --- Příprava BTTS ---
        btts_bet, btts_prob, btts_odd, btts_ev = None, 0.0, 0.0, -1.0
        if gh >= self.MIN_TEAM_XG_BTTS_YES and ga >= self.MIN_TEAM_XG_BTTS_YES and total_xg >= self.MIN_TOTAL_XG_BTTS_YES and p_btts_y >= self.MIN_PROB_GOALS_VALUE:
            btts_bet, btts_prob, btts_odd = "BTTS: Ano", p_btts_y, odds['odd_btts_y']
        elif min(gh, ga) <= self.MAX_TEAM_XG_BTTS_NO and total_xg <= 2.40 and p_btts_n >= self.MIN_PROB_GOALS_VALUE:
            btts_bet, btts_prob, btts_odd = "BTTS: Ne", p_btts_n, odds['odd_btts_n']
        if btts_bet and btts_odd > 1.0:
            btts_ev = (btts_prob * btts_odd) - 1.0

        # =====================================================================
        # FÁZE A: HLEDÁNÍ VALUE SINGLE SÁZKY
        # =====================================================================
        # 1. 1X2 Value Single (Hlavní disciplína - plný Kelly limit 3.5 %)
        if (base_prob >= self.MIN_PROB_THRESHOLD
                and 1.0 < base_odd <= self.MAX_ODDS_SINGLE
                and base_req_margin <= base_ev <= self.MAX_VALUE_MARGIN):
            stake = self._calc_kelly_stake(base_ev, base_odd, current_bankroll, is_secondary=False)
            if stake >= 20:
                return base_bet, base_prob, base_odd, "💎 VALUE SINGLE", stake, base_ev

        # 2. DC (1X / X2) Value Single (jen silní favorité >= 75 %, kurz <= 1.60, poloviční vklad)
        if (dc_prob >= self.MIN_PROB_DC_VALUE
                and self.MIN_ODD_DC_AKO <= dc_odd <= self.MAX_ODD_DC_VALUE
                and self.MIN_MARGIN_DC_VALUE <= dc_ev <= self.MAX_VALUE_MARGIN):
            stake = self._calc_kelly_stake(dc_ev, dc_odd, current_bankroll, is_secondary=True)
            if stake >= 20:
                return dc_bet, dc_prob, dc_odd, "💎 VALUE SINGLE", stake, dc_ev

        # 3. Over / Under 2.5 Value Single (jen při xG >= 3.15 / <= 2.00 a šanci >= 65 %)
        if (ou_bet is not None
                and 1.0 < ou_odd <= self.MAX_ODDS_SINGLE
                and self.MIN_MARGIN_GOALS_VALUE <= ou_ev <= self.MAX_VALUE_MARGIN):
            stake = self._calc_kelly_stake(ou_ev, ou_odd, current_bankroll, is_secondary=True)
            if stake >= 20:
                return ou_bet, ou_prob, ou_odd, "💎 VALUE SINGLE", stake, ou_ev

        # 4. BTTS Value Single
        if (btts_bet is not None
                and 1.0 < btts_odd <= self.MAX_ODDS_SINGLE
                and self.MIN_MARGIN_GOALS_VALUE <= btts_ev <= self.MAX_VALUE_MARGIN):
            stake = self._calc_kelly_stake(btts_ev, btts_odd, current_bankroll, is_secondary=True)
            if stake >= 20:
                return btts_bet, btts_prob, btts_odd, "💎 VALUE SINGLE", stake, btts_ev

        # =====================================================================
        # FÁZE B: HLEDÁNÍ PREMIUM AKO TIPU
        # =====================================================================
        if base_prob >= self.MIN_AKO_PROB and base_odd > 1.0 and base_ev <= self.MAX_VALUE_MARGIN:
            return base_bet, base_prob, base_odd, "🔗 AKO TIP", 0, base_ev

        if dc_prob >= self.MIN_AKO_PROB_DC and dc_odd >= self.MIN_ODD_DC_AKO and dc_ev <= self.MAX_VALUE_MARGIN:
            return dc_bet, dc_prob, dc_odd, "🔗 AKO TIP", 0, dc_ev

        if ou_bet is not None and ou_prob >= self.MIN_AKO_PROB_GOALS and ou_odd > 1.0 and ou_ev <= self.MAX_VALUE_MARGIN:
            return ou_bet, ou_prob, ou_odd, "🔗 AKO TIP", 0, ou_ev

        if btts_bet is not None and btts_prob >= self.MIN_AKO_PROB_GOALS and btts_odd > 1.0 and btts_ev <= self.MAX_VALUE_MARGIN:
            return btts_bet, btts_prob, btts_odd, "🔗 AKO TIP", 0, btts_ev

        return None, 0, 0, "❌ SKIP", 0, 0

    def _check_winner(self, bet_code, actual, goals_h, goals_a):
        if bet_code == "1": return actual == 2
        if bet_code == "X": return actual == 1
        if bet_code == "2": return actual == 0
        if bet_code == "1X": return actual in (2, 1)
        if bet_code == "X2": return actual in (0, 1)
        if bet_code == "Over 2.5": return (goals_h + goals_a) > 2.5
        if bet_code == "Under 2.5": return (goals_h + goals_a) < 2.5
        if bet_code == "BTTS: Ano": return (goals_h > 0) and (goals_a > 0)
        if bet_code == "BTTS: Ne": return (goals_h == 0) or (goals_a == 0)
        return False

    def run_backtest(self, diagnostic_months=None):
        if diagnostic_months is not None and self._cached_summer_result is not None:
            print("\n  ⚡ Výsledky letních měsíců načteny z mezipaměti (spočítáno v 1. průchodu)...")
            return self._cached_summer_result

        try:
            feature_cols = joblib.load(os.path.join(self.model_dir, "feature_cols.pkl"))
        except Exception as e:
            return {"status": "error", "message": f"Chybí feature_cols.pkl: {e}"}

        print("\n  📥 Načítám historická data a REÁLNÉ kurzy (Walk-Forward: Voting + XGBoost + Čistá 5zápasová xG forma)...")
        df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", self.engine)
        df = df.dropna(subset=["match_date"]).reset_index(drop=True)

        try:
            fx_df = pd.read_sql("SELECT id, league FROM fixtures", self.engine)
            league_map = dict(zip(fx_df['id'], fx_df['league']))
        except Exception:
            league_map = {}

        if 'league' not in df.columns:
            df['league'] = df['fixture_id'].map(league_map).fillna("?")
        else:
            df['league'] = df['league'].fillna(df['fixture_id'].map(league_map)).fillna("?")

        if 'goals_home' in df.columns and 'goals_away' in df.columns:
            df['target'] = np.where(df['goals_home'] > df['goals_away'], 2,
                           np.where(df['goals_home'] < df['goals_away'], 0, 1))

        df = df.dropna(subset=['target']).reset_index(drop=True)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        try:
            odds_df = pd.read_sql("SELECT * FROM bookmaker_odds", self.engine)
            odds_map = {row['fixture_id']: row for _, row in odds_df.iterrows()}
        except Exception:
            odds_map = {}

        start_index = int(len(df) * self.INITIAL_TRAIN_PCT)

        bankroll_main = self.START_BANKROLL
        bets_main = []

        bankroll_summer = self.START_BANKROLL
        bets_summer = []

        summer_months_set = set(diagnostic_months) if diagnostic_months is not None else {5, 6, 7, 8}
        voting_model = self._build_voting_pipeline()
        xgb_model = self._build_xgb_pipeline()

        print(f"  🚀 Walk-forward start: zápas {start_index}/{len(df)} (Konsenzus Voting + XGBoost + 5zápasové xG)")

        for i in range(start_index, len(df), self.RETRAIN_STEP):
            X_tr, y_tr = df.iloc[:i][feature_cols], df.iloc[:i]['target'].astype(int)

            try:
                voting_model.fit(X_tr, y_tr)
                sw = compute_sample_weight('balanced', y_tr)
                xgb_model.fit(X_tr, y_tr, xgb__sample_weight=sw)
            except Exception:
                continue

            end = min(i + self.RETRAIN_STEP, len(df))
            test_slice = df.iloc[i:end]
            X_te = test_slice[feature_cols]

            try:
                probs_v_all = voting_model.predict_proba(X_te)
                probs_x_all = xgb_model.predict_proba(X_te)
                probs_all = 0.5 * probs_v_all + 0.5 * probs_x_all
            except Exception:
                continue

            for idx_local, (_, row) in enumerate(test_slice.iterrows()):
                odds = self._get_real_odds(row, odds_map)
                if odds['odd_1'] == 0.0 and odds['odd_1x'] == 0.0:
                    continue

                mdate = pd.Timestamp(row['match_date']) if pd.notnull(row.get('match_date')) else None
                is_summer = (mdate.month in summer_months_set) if mdate else False

                actual = int(row['target'])
                actual_gh = self._safe_float(row.get('goals_home'))
                actual_ga = self._safe_float(row.get('goals_away'))

                pv_a, pv_x, pv_h = probs_all[idx_local]

                # Čistá 5zápasová xG forma jako nezávislá kotva vůči ELO klasifikátorům
                avg_h_xg = float(np.clip(self._safe_float(row.get('home_avg_expected_goals_last5')) or self._safe_float(row.get('home_avg_goals_last5')) or 1.5, 0.3, 4.0))
                avg_a_xg = float(np.clip(self._safe_float(row.get('away_avg_expected_goals_last5')) or self._safe_float(row.get('away_avg_goals_last5')) or 1.2, 0.3, 4.0))

                p1_poi, px_poi, p2_poi = 0.0, 0.0, 0.0
                p_under_25 = 0.0
                for h in range(10):
                    for a in range(10):
                        p = poisson.pmf(h, avg_h_xg) * poisson.pmf(a, avg_a_xg)
                        if h > a: p1_poi += p
                        elif h == a: px_poi += p
                        else: p2_poi += p
                        if (h + a) < 2.5:
                            p_under_25 += p

                p1, pxb, p2 = 0.5 * pv_h + 0.5 * p1_poi, 0.5 * pv_x + 0.5 * px_poi, 0.5 * pv_a + 0.5 * p2_poi
                tot = p1 + pxb + p2
                p1, pxb, p2 = p1 / tot, pxb / tot, p2 / tot

                p_over_25 = 1.0 - p_under_25
                p_btts_y = (1.0 - poisson.pmf(0, avg_h_xg)) * (1.0 - poisson.pmf(0, avg_a_xg))
                p_btts_n = 1.0 - p_btts_y

                active_bankroll = bankroll_summer if is_summer else bankroll_main

                bet_code, prob, odd, signal_type, stake, ev = self._evaluate_bet(
                    p1, pxb, p2, p_over_25, p_under_25, p_btts_y, p_btts_n,
                    avg_h_xg, avg_a_xg, odds, active_bankroll
                )

                if bet_code is None:
                    continue

                won = self._check_winner(bet_code, actual, actual_gh, actual_ga)
                profit = 0.0

                if "VALUE" in signal_type:
                    profit = (stake * odd) - stake if won else -stake
                    if is_summer:
                        bankroll_summer += profit
                        active_bankroll = bankroll_summer
                    else:
                        bankroll_main += profit
                        active_bankroll = bankroll_main

                bet_record = {
                    "Date": mdate.strftime("%d.%m.%Y") if mdate else "",
                    "League": str(row.get('league', '?')),
                    "Match": f"{row.get('home_team','?')} vs {row.get('away_team','?')}",
                    "Signal": signal_type,
                    "Bet": bet_code,
                    "Prob": f"{prob*100:.1f}%",
                    "EV": f"{ev*100:+.1f}%",
                    "Odds": f"{odd:.2f}",
                    "Stake": round(stake, 0),
                    "Result": "✅ WIN" if won else "❌ LOSS",
                    "Profit": round(profit, 1),
                    "Bankroll": round(active_bankroll, 1) if "VALUE" in signal_type else "-"
                }

                if is_summer:
                    bets_summer.append(bet_record)
                else:
                    bets_main.append(bet_record)

        self._cached_main_result = {
            "status": "success",
            "bets_log": bets_main,
            "end_bankroll": bankroll_main
        }
        self._cached_summer_result = {
            "status": "success",
            "bets_log": bets_summer,
            "end_bankroll": bankroll_summer
        }

        return self._cached_summer_result if diagnostic_months is not None else self._cached_main_result

    def _print_league_breakdown(self, df_log, label):
        if 'League' not in df_log.columns or df_log.empty:
            return

        print("\n" + "=" * 96)
        print(f"  🌍 VÝKONNOST A ROI PODLE LIG — {label}")
        print("=" * 96)
        print(f"  {'Liga':<6} | {'Value Sázky':>11} | {'Win Rate':>9} | {'Vsazeno':>11} | {'Čistý Zisk':>11} | {'ROI':>9} | {'AKO Tipy (Hit %)'}")
        print("  " + "-" * 92)

        leagues = df_log['League'].unique()
        rows_data = []

        for lg in leagues:
            lg_df = df_log[df_log['League'] == lg]
            val_df = lg_df[lg_df['Signal'] == "💎 VALUE SINGLE"]
            ako_df = lg_df[lg_df['Signal'] == "🔗 AKO TIP"]

            v_cnt = len(val_df)
            v_wins = (val_df['Result'] == '✅ WIN').sum() if v_cnt > 0 else 0
            v_wr = (v_wins / v_cnt * 100) if v_cnt > 0 else 0.0
            v_stake = val_df['Stake'].sum() if v_cnt > 0 else 0.0
            v_prof = val_df['Profit'].sum() if v_cnt > 0 else 0.0
            v_roi = (v_prof / v_stake * 100) if v_stake > 0 else 0.0

            a_cnt = len(ako_df)
            a_wins = (ako_df['Result'] == '✅ WIN').sum() if a_cnt > 0 else 0
            a_wr = (a_wins / a_cnt * 100) if a_cnt > 0 else 0.0

            rows_data.append({
                "lg": lg, "v_cnt": v_cnt, "v_wins": v_wins, "v_wr": v_wr,
                "v_stake": v_stake, "v_prof": v_prof, "v_roi": v_roi,
                "a_cnt": a_cnt, "a_wins": a_wins, "a_wr": a_wr
            })

        rows_data.sort(key=lambda x: x['v_prof'], reverse=True)

        for r in rows_data:
            wr_str = f"{r['v_wr']:5.1f} %" if r['v_cnt'] > 0 else "    -  "
            roi_str = f"{r['v_roi']:+6.1f} %" if r['v_cnt'] > 0 else "     -  "
            ako_str = f"{r['a_wins']:>2}/{r['a_cnt']:<2} ({r['a_wr']:5.1f} %)" if r['a_cnt'] > 0 else " 0/0       (-)"

            print(f"  {r['lg']:<6} | {r['v_cnt']:>11} | {wr_str:>9} | {r['v_stake']:>8,.0f} Kč | {r['v_prof']:>+8,.0f} Kč | {roi_str:>9} |   {ako_str}")

        print("=" * 96)

    def print_report(self, result_dict, label="VALUE BETTING + AKO TIPS"):
        if result_dict.get("status") != "success":
            print(f"❌ Nelze zobrazit report: {result_dict.get('message')}")
            return

        bets_log = result_dict.get("bets_log", [])
        if not bets_log:
            print(f"\n⚠️  {label}: Nenalezena žádná data.")
            return

        df_log = pd.DataFrame(bets_log)
        value_bets = df_log[df_log['Signal'] == "💎 VALUE SINGLE"]
        ako_bets = df_log[df_log['Signal'] == "🔗 AKO TIP"]

        print("\n" + "=" * 125)
        print(f"📊 LOG SÁZEK — {label}")
        print("=" * 125)
        print(df_log.tail(30).to_string(index=False))

        print("\n" + "=" * 60)
        print(f"  💎 VALUE SINGLY (Řízeno Kellyho kritériem)")
        print("=" * 60)

        if not value_bets.empty:
            wins = (value_bets['Result'] == '✅ WIN').sum()
            invested = value_bets['Stake'].sum()
            net_profit = result_dict['end_bankroll'] - self.START_BANKROLL
            roi = (net_profit / invested * 100) if invested > 0 else 0

            print(f"  Počet sázek:      {len(value_bets)}")
            print(f"  Win rate:         {wins / len(value_bets) * 100:.1f} % ({wins} výher / {len(value_bets) - wins} proher)")
            print(f"  Celkem vsazeno:   {invested:,.0f} Kč")
            print(f"  Konečný Bankroll: {result_dict['end_bankroll']:,.0f} Kč")
            print(f"  Čistý zisk/ztráta:{net_profit:>+10,.0f} Kč")
            print(f"  ROI:              {roi:>+9.1f} %")

            v_1x2 = value_bets[value_bets['Bet'].isin(['1', '2'])]
            v_dc = value_bets[value_bets['Bet'].isin(['1X', 'X2'])]
            v_goals = value_bets[~value_bets['Bet'].isin(['1', '2', '1X', 'X2'])]
            print("  " + "-" * 56)
            if not v_1x2.empty:
                w_1x2 = (v_1x2['Result'] == '✅ WIN').sum()
                p_1x2 = v_1x2['Profit'].sum()
                print(f"  • Čisté tipy (1/2):    {len(v_1x2):>2}x | Win rate: {w_1x2/len(v_1x2)*100:5.1f} % | Zisk: {p_1x2:>+7,.0f} Kč")
            if not v_dc.empty:
                w_dc = (v_dc['Result'] == '✅ WIN').sum()
                p_dc = v_dc['Profit'].sum()
                print(f"  • Neprohry (1X/X2):    {len(v_dc):>2}x | Win rate: {w_dc/len(v_dc)*100:5.1f} % | Zisk: {p_dc:>+7,.0f} Kč")
            if not v_goals.empty:
                w_g = (v_goals['Result'] == '✅ WIN').sum()
                p_g = v_goals['Profit'].sum()
                print(f"  • Góly / BTTS:         {len(v_goals):>2}x | Win rate: {w_g/len(v_goals)*100:5.1f} % | Zisk: {p_g:>+7,.0f} Kč")

            bankrolls = pd.to_numeric(value_bets['Bankroll'], errors='coerce').dropna().values
            if len(bankrolls) > 0:
                max_b, min_b = bankrolls.max(), bankrolls.min()
                print(f"\n  📈 Equity křivka (min={min_b:.0f} Kč, max={max_b:.0f} Kč):")
                bar_w = 45
                for i in range(0, len(bankrolls), max(1, len(bankrolls) // 15)):
                    b = bankrolls[i]
                    pct = (b - min_b) / (max_b - min_b + 1) if max_b != min_b else 0.5
                    print(f"  {i:>4} | {'█' * int(pct * bar_w):<{bar_w}} {b:,.0f} Kč")
        else:
            print("  Žádné Value Singly nenalezeny.")

        print("\n" + "=" * 60)
        print(f"  🔗 PREMIUM AKO TIPY (1/2 > {self.MIN_AKO_PROB*100:.0f}% | 1X/X2 > {self.MIN_AKO_PROB_DC*100:.0f}%)")
        print("=" * 60)
        if not ako_bets.empty:
            ako_wins = (ako_bets['Result'] == '✅ WIN').sum()
            print(f"  Počet tipů:       {len(ako_bets)}")
            print(f"  Úspěšnost (Hit):  {ako_wins / len(ako_bets) * 100:.1f} % ({ako_wins} výher)")
            prob_mean = ako_bets['Prob'].str.rstrip('%').astype(float).mean()
            print(f"  Průměrná šance:   {prob_mean:.1f} %")

            a_1x2 = ako_bets[ako_bets['Bet'].isin(['1', '2'])]
            a_dc = ako_bets[ako_bets['Bet'].isin(['1X', 'X2'])]
            a_goals = ako_bets[~ako_bets['Bet'].isin(['1', '2', '1X', 'X2'])]
            if not a_dc.empty or not a_goals.empty:
                print("  " + "-" * 56)
                if not a_1x2.empty:
                    aw_1x2 = (a_1x2['Result'] == '✅ WIN').sum()
                    print(f"  • AKO Čisté (1/2):     {aw_1x2}/{len(a_1x2)} ({aw_1x2/len(a_1x2)*100:5.1f} %)")
                if not a_dc.empty:
                    aw_dc = (a_dc['Result'] == '✅ WIN').sum()
                    print(f"  • AKO Neprohry (1X/X2):{aw_dc}/{len(a_dc)} ({aw_dc/len(a_dc)*100:5.1f} %)")
                if not a_goals.empty:
                    aw_g = (a_goals['Result'] == '✅ WIN').sum()
                    print(f"  • AKO Góly / BTTS:     {aw_g}/{len(a_goals)} ({aw_g/len(a_goals)*100:5.1f} %)")
        else:
            print("  Žádné AKO tipy nenalezeny.")

        self._print_league_breakdown(df_log, label)