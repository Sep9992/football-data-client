"""
strategy_backtester.py
Nástroj AI agenta pro otestování strategie na historických datech.
ARCHITEKTURA: Asymmetric Value Singles (Home vs Away) + Premium AKO Tips.
UPGRADE: Jednoprůchodový Walk-Forward (Single-Pass Cache) + Rozpad ROI po ligách.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


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
        self.KELLY_FRACTION = 0.25       # Quarter Kelly
        self.MAX_STAKE_PCT = 0.035       # Max 3.5% bankrollu na sázku (Bezpečnější drawdown)

        self.MIN_PROB_THRESHOLD = 0.50   # Sázíme jen favority > 50%
        self.MAX_ODDS_SINGLE = 2.50      # Zákaz obřích kurzů pro singly
        self.MAX_VALUE_MARGIN = 0.25     # EV nesmí přesáhnout +25% (Value Traps)

        # Asymetrická Value (Ochrana proti selhání favoritů venku)
        self.MIN_VALUE_MARGIN_HOME = 0.04  # Domácí favorit potřebuje výhodu +4%
        self.MIN_VALUE_MARGIN_AWAY = 0.06  # Hostující favorit musí mít výhodu +6%

        self.MIN_AKO_PROB = 0.65         # AKO TIP: Zápas musí mít šanci aspoň 65%

        self.INITIAL_TRAIN_PCT = 0.60
        self.RETRAIN_STEP = 10

        # Interní cache pro jednoprůchodový výpočet (Single-Pass)
        self._cached_main_result = None
        self._cached_summer_result = None

    def _build_voting_pipeline(self):
        prep = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        rf  = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
        lr  = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=42)
        voting = VotingClassifier(estimators=[('rf', rf), ('gbm', gbm), ('lr', lr)], voting='soft')
        return Pipeline([('prep', prep), ('clf', voting)])

    def _get_real_odds(self, row, odds_map):
        fid = row.get('fixture_id')
        if fid in odds_map:
            o = odds_map[fid]
            return float(o.get('odd_1') or 0), float(o.get('odd_x') or 0), float(o.get('odd_2') or 0)
        return 0.0, 0.0, 0.0

    def _evaluate_bet(self, p1, px, p2, odd1, oddx, odd2, current_bankroll):
        options = {"1": (p1, odd1), "X": (px, oddx), "2": (p2, odd2)}

        best_bet = max(options, key=lambda k: options[k][0])
        best_prob, best_odd = options[best_bet]

        if best_odd <= 1.0 or best_bet == "X":  # Remízy ze singlů vynecháme úplně, je to loterie
            return None, 0, 0, "❌ SKIP (Chybí kurz/Remíza)", 0, 0

        if best_prob < self.MIN_PROB_THRESHOLD:
            return None, 0, 0, "❌ SKIP (Nízká důvěra)", 0, 0

        ev = (best_prob * best_odd) - 1.0

        # Požadovaná marže se liší podle toho, zda hraje tým doma nebo venku
        req_margin = self.MIN_VALUE_MARGIN_HOME if best_bet == "1" else self.MIN_VALUE_MARGIN_AWAY

        if ev > self.MAX_VALUE_MARGIN:
            return None, 0, 0, f"❌ SKIP (Podezřelý kurz, EV +{ev*100:.1f}%)", 0, ev

        elif ev >= req_margin and best_odd <= self.MAX_ODDS_SINGLE:
            kelly_full = ev / (best_odd - 1.0)
            stake_pct = min(max(kelly_full * self.KELLY_FRACTION, 0), self.MAX_STAKE_PCT)
            stake = current_bankroll * stake_pct

            if stake < 20:
                return None, 0, 0, "❌ SKIP (Malý vklad)", 0, ev

            return best_bet, best_prob, best_odd, "💎 VALUE SINGLE", stake, ev

        elif best_prob >= self.MIN_AKO_PROB:
            return best_bet, best_prob, best_odd, "🔗 AKO TIP", 0, ev

        else:
            reason = "Vysoký kurz" if best_odd > self.MAX_ODDS_SINGLE else "Malá hodnota"
            return None, 0, 0, f"❌ SKIP ({reason})", 0, ev

    def _check_winner(self, bet_code, actual):
        if bet_code == "1": return actual == 2
        if bet_code == "X": return actual == 1
        if bet_code == "2": return actual == 0
        return False

    def run_backtest(self, diagnostic_months=None):
        # Pokud už proběhl 1. průchod a agent nyní žádá letní diagnostiku, vrátíme ji okamžitě z cache!
        if diagnostic_months is not None and self._cached_summer_result is not None:
            print("\n  ⚡ Výsledky letních měsíců načteny z mezipaměti (spočítáno v 1. průchodu)...")
            return self._cached_summer_result

        try:
            feature_cols = joblib.load(os.path.join(self.model_dir, "feature_cols.pkl"))
        except Exception as e:
            return {"status": "error", "message": f"Chybí feature_cols.pkl: {e}"}

        print("\n  📥 Načítám historická data a REÁLNÉ kurzy (Jednoprůchodový Walk-Forward)...")
        df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", self.engine)
        df = df.dropna(subset=["match_date"]).reset_index(drop=True)

        # Pojistka: Doplnění kódu ligy z tabulky fixtures, pokud by v prepared_datasets chyběl
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

        # Dva oddělené účty pro jeden společný průchod historií
        bankroll_main = self.START_BANKROLL
        bets_main = []

        bankroll_summer = self.START_BANKROLL
        bets_summer = []

        summer_months_set = set(diagnostic_months) if diagnostic_months is not None else {5, 6, 7, 8}
        model = self._build_voting_pipeline()

        print(f"  🚀 Walk-forward start: zápas {start_index}/{len(df)} (počítám Hlavní i Letní sezónu zároveň)")

        for i in range(start_index, len(df), self.RETRAIN_STEP):
            X_tr, y_tr = df.iloc[:i][feature_cols], df.iloc[:i]['target'].astype(int)
            try:
                model.fit(X_tr, y_tr)
            except Exception:
                continue

            end = min(i + self.RETRAIN_STEP, len(df))
            test_slice = df.iloc[i:end]
            X_te = test_slice[feature_cols]

            try:
                probs_all = model.predict_proba(X_te)
            except Exception:
                continue

            for idx_local, (_, row) in enumerate(test_slice.iterrows()):
                odd1, oddx, odd2 = self._get_real_odds(row, odds_map)
                if odd1 == 0.0:
                    continue

                mdate = pd.Timestamp(row['match_date']) if pd.notnull(row.get('match_date')) else None
                is_summer = (mdate.month in summer_months_set) if mdate else False

                actual = int(row['target'])
                pv_a, pv_x, pv_h = probs_all[idx_local]

                avg_h_xg = float(np.clip(row.get('home_avg_expected_goals_last5') or row.get('home_avg_goals_last5', 1.5), 0.3, 4.0))
                avg_a_xg = float(np.clip(row.get('away_avg_expected_goals_last5') or row.get('away_avg_goals_last5', 1.2), 0.3, 4.0))

                p1_poi, px_poi, p2_poi = 0.0, 0.0, 0.0
                for h in range(10):
                    for a in range(10):
                        p = poisson.pmf(h, avg_h_xg) * poisson.pmf(a, avg_a_xg)
                        if h > a: p1_poi += p
                        elif h == a: px_poi += p
                        else: p2_poi += p

                p1, pxb, p2 = 0.5 * pv_h + 0.5 * p1_poi, 0.5 * pv_x + 0.5 * px_poi, 0.5 * pv_a + 0.5 * p2_poi
                tot = p1 + pxb + p2
                p1, pxb, p2 = p1 / tot, pxb / tot, p2 / tot

                # Rozhodnutí, se kterým bankrollem pracujeme (Hlavní sezóna vs. Léto)
                active_bankroll = bankroll_summer if is_summer else bankroll_main

                bet_code, prob, odd, signal_type, stake, ev = self._evaluate_bet(
                    p1, pxb, p2, odd1, oddx, odd2, active_bankroll
                )

                if bet_code is None:
                    continue

                won = self._check_winner(bet_code, actual)
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
        """Vypíše přehlednou tabulku úspěšnosti a ROI rozpadnutou po jednotlivých ligách."""
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

        # Seřadíme ligy od nejvyššího čistého zisku po nejnižší
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
        print(f"  🔗 PREMIUM AKO TIPY (Vysoká šance > {self.MIN_AKO_PROB*100}%)")
        print("=" * 60)
        if not ako_bets.empty:
            ako_wins = (ako_bets['Result'] == '✅ WIN').sum()
            print(f"  Počet tipů:       {len(ako_bets)}")
            print(f"  Úspěšnost (Hit):  {ako_wins / len(ako_bets) * 100:.1f} % ({ako_wins} výher)")
            prob_mean = ako_bets['Prob'].str.rstrip('%').astype(float).mean()
            print(f"  Průměrná šance:   {prob_mean:.1f} %")
        else:
            print("  Žádné AKO tipy nenalezeny.")

        # Vykreslení nového rozpadu po ligách
        self._print_league_breakdown(df_log, label)