"""
step8_backtest_strategy.py  v7
================================
Změny v6 oproti v5 (4 úpravy na základě analýzy backtestového logu v5):

  ÚPRAVA 1 — Oprava timing rebase:
    Problém v5: Rebase každých 30 sázek → nastane náhodně uprostřed série proher.
                Log: sázka 30 rebáze 500→610 Kč, hned 2× ztráta (-1220 Kč šok).
    Řešení: Rebase po každém KOLE (každých 14 dní kalendářního času),
            nikoliv po počtu sázek. Nikdy nerebázuje při aktivní ztrátové sérii
            (3+ po sobě jdoucí prohry).

  ÚPRAVA 2 — Cooling filtr (drawdown ochrana):
    Problém v5: Max drawdown 23.6% → po 3 prohrách model pokračuje plnou sazbou.
    Řešení: Po 3 po sobě jdoucích prohrách → sázka se sníží na 50% flat.
            Normální sázka se obnoví po 2 po sobě jdoucích výhrách.
            Výsledek: omezuje drawdown bez změny strategie výběru sázek.

  ÚPRAVA 3 — Vrácení THRESH_FAVORIT na 55%:
    Problém v6: Snížení na 53% zničilo win rate (66.7% → 61.2%).
                Break-even při kurzu 1.60 = 62.5%, model při 53% dosahuje jen 61.2%.
    Řešení: THRESH_FAVORIT zpět na 0.55 — ověřený edge, méně sázek ale kvalitnějších.

  ÚPRAVA 4 — Oprava sezónního filtru (červenec + celý srpen):
    Problém v6: Filtr pustil dovnitř červenec (19.07. Jablonec vs Sparta — předzápasové
                období, model nemá žádná formová data). Filtr měl jen horní hranici.
    Řešení: Zakázat sázky v červenci a srpnu úplně (měsíce 7 a 8).
            Platná sezóna = září–červen. Jednoduché, robustní, bez edge-case chyb.

ZACHOVÁNO z v5:
  - Pouze čisté výhry (1 nebo 2) — VALUE/SAFE odstraněny
  - Zpřísněný formový filtr (pts≥1.8 / goals≥1.3)
  - MIN_ODDS_1X2=1.50
  - Walk-forward bez data leakage
  - Blend 50/50 Voting + Poisson
"""

import os
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

# =============================================================================
# 1. KONFIGURACE
# =============================================================================

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- BANKROLL ---
START_BANKROLL  = 10_000
MIN_BET_AMOUNT  = 50

# --- FLAT STAKING ---
FLAT_STAKE_FAVORIT = 400   # Kč: výchozí flat sázka
# ÚPRAVA 1: Rebase po kole (14 dní), ne po počtu sázek
REBASE_DAYS        = 14    # Rebasuj max jednou za 14 dní
# ÚPRAVA 2: Cooling filtr
COOLING_LOSSES     = 3     # Po N prohrách → snižuj na COOLING_FACTOR sázky
COOLING_WINS       = 2     # Po N výhrách v cooling → obnov plnou sázku
COOLING_FACTOR     = 0.5   # 50% flat při aktivním cooling

# --- SIGNÁLOVÉ PRAHY ---
# ÚPRAVA 3: Snížení prahu z 0.55 → 0.53
THRESH_FAVORIT  = 0.55    # Minimální pravděpodobnost pro čistou výhru
THRESH_SUPER    = 0.85
BOOKMAKER_MARGIN = 0.10

# --- MIN ODDS ---
MIN_ODDS_1X2    = 1.50

# --- ÚPRAVA 4: Sezónní filtr ---
# Platná sezóna: září–červen (měsíce 7 a 8 = letní přestávka + předzápasové)

# --- WALK-FORWARD ---
INITIAL_TRAIN_PCT = 0.60
RETRAIN_STEP      = 10


# =============================================================================
# 2. DRAW BOOST (shodné se step4–step6)
# =============================================================================

def predict_with_draw_boost(proba, threshold):
    """proba: ndarray (n, 3) — pořadí [Away, Draw, Home]"""
    preds = []
    for p in proba:
        p_a, p_x, p_h = p[0], p[1], p[2]
        if p_x >= threshold and p_x > min(p_a, p_h):
            preds.append(1)
        else:
            preds.append(int(np.argmax(p)))
    return np.array(preds)


# =============================================================================
# 3. WALK-FORWARD VOTING PIPELINE (stejná architektura jako step3)
# =============================================================================

def build_voting_pipeline():
    """Stejná architektura jako v step3 — RF + GBM + LR, třída 'balanced'."""
    prep = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler())
    ])
    rf  = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                     max_depth=4, subsample=0.8, random_state=42)
    lr  = LogisticRegression(max_iter=2000, C=0.5,
                              class_weight='balanced', random_state=42)
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('lr', lr)],
        voting='soft'
    )
    return Pipeline([('prep', prep), ('clf', voting)])


# =============================================================================
# 4. FORMOVÝ FILTR — ZLEPŠENÍ 2
# =============================================================================

def form_ok(ph, pa, row):
    """
    Vrátí True pokud momentální forma POTVRZUJE strukturální výhodu.

    ÚPRAVA B — Zpřísněné prahy (odpovídají top polovina tabulky):
      Domácí favorit: h_pts >= 1.8  (bylo 1.5)  h_goals >= 1.3  (bylo 1.2)
      Hosté favorit:  a_pts >= 1.5  (bylo 1.2)  a_goals >= 1.2  (bylo 1.0)
    """
    if ph >= pa:
        h_pts   = float(row.get('home_avg_points_last5') or 0)
        h_goals = float(row.get('home_avg_goals_last5')  or 0)
        return h_pts >= 1.8 and h_goals >= 1.3
    else:
        a_pts   = float(row.get('away_avg_points_last5') or 0)
        a_goals = float(row.get('away_avg_goals_last5')  or 0)
        return a_pts >= 1.5 and a_goals >= 1.2


# =============================================================================
# 5. SIGNÁLOVÁ LOGIKA — ZLEPŠENÍ 1 (diferenciované MIN_ODDS)
# =============================================================================

def is_early_season(match_date) -> bool:
    """
    ÚPRAVA 4 (v7): Vrátí True pro červenec a srpen — letní přestávka + předzápasové.
    Platná sezóna = září (9) až červen (6).
    Jednodušší a robustnější než týdenní počítání od 1.8.
    """
    try:
        dt = pd.Timestamp(match_date)
        return dt.month in (7, 8)
    except Exception:
        return False



def classify_signal(ph, px, pa):
    """
    Vrátí (bet_code, prob_dc, sim_odds, signal_type, is_double_chance).

    Sázíme VÝHRADNĚ čisté výhry (1 nebo 2).
    VALUE/SAFE (1X, X2) odstraněny — systematicky ztrátové při kurzech 1.35–1.45.
    """
    bet_code    = None
    prob_dc     = 0.0
    signal_type = None

    if ph >= pa:
        if ph > THRESH_FAVORIT:
            bet_code    = "1"
            prob_dc     = ph
            signal_type = "💎 FAVORIT+" if ph > THRESH_SUPER else "🔥 FAVORIT"
    else:
        if pa > THRESH_FAVORIT:
            bet_code    = "2"
            prob_dc     = pa
            signal_type = "💎 FAVORIT+" if pa > THRESH_SUPER else "🔥 FAVORIT"

    if bet_code is None:
        return None, 0.0, 0.0, None, False

    sim_odds = (1.0 / prob_dc) * (1 - BOOKMAKER_MARGIN)

    if sim_odds < MIN_ODDS_1X2:
        signal_type = f"❌ SKIP ({signal_type})"

    return bet_code, prob_dc, sim_odds, signal_type, False  # is_dc vždy False


# =============================================================================
# 5. IS_WINNER — opravené kódování (Away=0, Draw=1, Home=2)
# =============================================================================

def check_winner(bet_code, actual):
    """
    actual: 0=Away win, 1=Draw, 2=Home win  (nová architektura)
    Vrátí True pokud sázka prošla.
    """
    if bet_code == "1":    return actual == 2          # Čistá výhra domácích
    if bet_code == "2":    return actual == 0          # Čistá výhra hostů
    if bet_code == "1X":   return actual in (1, 2)     # Domácí nevyhráli hosté
    if bet_code == "X2":   return actual in (0, 1)     # Hosté nevyhráli domácí
    return False


# =============================================================================
# 6. STATISTIKY A REPORTING
# =============================================================================

def compute_stats(bets_log, start_bankroll, end_bankroll):
    if not bets_log:
        return {}

    df = pd.DataFrame(bets_log)
    total   = len(df)
    wins    = (df['Profit'] > 0).sum()
    losses  = total - wins
    invested = df['Stake'].sum()
    net_profit = end_bankroll - start_bankroll
    roi     = net_profit / invested * 100 if invested > 0 else 0
    win_rate = wins / total * 100

    # Max drawdown
    bankroll_series = df['Bankroll'].values
    peak = bankroll_series[0]
    max_dd = 0.0
    for b in bankroll_series:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Per-signál statistiky
    per_signal = df.groupby('Signal').agg(
        Sázky   = ('Profit', 'count'),
        Výhry   = ('Profit', lambda x: (x > 0).sum()),
        Profit  = ('Profit', 'sum'),
        Avg_odd = ('Odds', 'mean')
    ).reset_index()
    per_signal['Win%'] = (per_signal['Výhry'] / per_signal['Sázky'] * 100).round(1)
    per_signal['ROI%'] = (per_signal['Profit'] / (df.groupby('Signal')['Stake'].sum().values) * 100).round(1)

    return {
        'total': total, 'wins': wins, 'losses': losses,
        'win_rate': win_rate, 'roi': roi, 'net_profit': net_profit,
        'invested': invested, 'max_drawdown': max_dd,
        'per_signal': per_signal
    }


def print_report(bets_log, stats, start_bankroll, end_bankroll, config_label):
    print("\n" + "=" * 110)
    print(f"📊 LOG SÁZEK — {config_label} ({stats.get('total', 0)} sázek)")
    print("=" * 110)

    if bets_log:
        df_log = pd.DataFrame(bets_log)
        print(df_log[["Date", "Match", "Signal", "Bet",
                       "Odds", "Stake", "Cooling", "Result", "Profit", "Bankroll"]].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"  💰 VÝSLEDEK: {config_label}")
    print("=" * 60)
    print(f"  Start bankroll:   {start_bankroll:>10,.0f} Kč")
    print(f"  Konec bankroll:   {end_bankroll:>10,.2f} Kč")
    print(f"  Čistý zisk:       {stats.get('net_profit', 0):>+10.2f} Kč")
    print(f"  ROI:              {stats.get('roi', 0):>+9.2f} %")
    print(f"  Win rate:         {stats.get('win_rate', 0):>9.1f} %")
    print(f"  Výhry/Prohry:     {stats.get('wins', 0)} / {stats.get('losses', 0)}")
    print(f"  Max drawdown:     {stats.get('max_drawdown', 0):>9.1f} %")
    print(f"  Investováno:      {stats.get('invested', 0):>10,.0f} Kč")
    print("=" * 60)

    per_signal = stats.get('per_signal')
    if per_signal is not None and not per_signal.empty:
        print("\n  📈 Per-signál statistiky:")
        print(f"  {'Signál':<20} {'Sázky':>6} {'Výhry':>6} {'Win%':>6} {'ROI%':>7} {'Avg kurz':>9}")
        print("  " + "-" * 58)
        for _, row in per_signal.iterrows():
            print(f"  {row['Signal']:<20} {row['Sázky']:>6} {row['Výhry']:>6} "
                  f"{row['Win%']:>5.1f}% {row['ROI%']:>6.1f}% {row['Avg_odd']:>8.2f}")


# =============================================================================
# 7. HLAVNÍ BACKTEST
# =============================================================================

def run_backtest(feature_cols, draw_threshold):
    """
    Walk-forward backtest na prepared_datasets.
    Retréninguje Voting pipeline každých RETRAIN_STEP zápasů.
    """
    print("  📥 Načítám historická data...")
    df = pd.read_sql(
        "SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine
    )
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)

    # Odvodit target z výsledků (nová arch nemá sloupec 'target')
    if 'target' not in df.columns:
        if 'goals_home' in df.columns and 'goals_away' in df.columns:
            df['target'] = np.where(df['goals_home'] > df['goals_away'], 2,
                           np.where(df['goals_home'] < df['goals_away'], 0, 1))
        else:
            print("  ❌ Chybí goals_home / goals_away pro odvození target!")
            return [], {}

    # Odfiltrovat zápasy bez výsledku
    df = df.dropna(subset=['target']).reset_index(drop=True)
    print(f"  ✅ {len(df)} zápasů s výsledkem | distribuce: "
          f"Away={( df['target']==0).sum()} "
          f"Draw={(df['target']==1).sum()} "
          f"Home={(df['target']==2).sum()}")

    # Doplň chybějící features nulou
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    start_index = int(len(df) * INITIAL_TRAIN_PCT)
    print(f"  🚀 Walk-forward start: zápas {start_index}/{len(df)} "
          f"(trénink na {start_index} zápasech, test na {len(df)-start_index})")

    current_bankroll = START_BANKROLL
    bets_log         = []
    model            = build_voting_pipeline()
    skipped_odds     = 0
    skipped_form     = 0
    skipped_season   = 0   # ÚPRAVA 4

    # ÚPRAVA 1: Rebase tracking (datum poslední rebáze)
    flat_favorit      = max(MIN_BET_AMOUNT, round(START_BANKROLL * 0.05 / 10) * 10)
    last_rebase_date  = None
    print(f"  💰 Počáteční flat sázka: FAVORIT={flat_favorit} Kč")

    # ÚPRAVA 2: Cooling stav
    cooling_active        = False
    consecutive_losses    = 0
    consecutive_wins_cool = 0

    for i in range(start_index, len(df), RETRAIN_STEP):
        X_tr = df.iloc[:i][feature_cols]
        y_tr = df.iloc[:i]['target'].astype(int)

        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            print(f"    ⚠️  Trénink selhal u idx {i}: {e}")
            continue

        end = min(i + RETRAIN_STEP, len(df))
        test_slice = df.iloc[i:end]
        X_te = test_slice[feature_cols]

        try:
            probs_all = model.predict_proba(X_te)
        except Exception:
            continue

        for idx_local, (_, row) in enumerate(test_slice.iterrows()):
            actual = int(row['target'])
            probs  = probs_all[idx_local]
            pv_a, pv_x, pv_h = probs[0], probs[1], probs[2]

            # ÚPRAVA 4: Sezónní filtr — přeskoč prvních 5 týdnů sezóny
            if is_early_season(row.get('match_date')):
                skipped_season += 1
                continue

            # Poisson složka
            avg_h_xg = row.get('home_avg_expected_goals_last5', None)
            avg_a_xg = row.get('away_avg_expected_goals_last5', None)
            if pd.isna(avg_h_xg) or avg_h_xg == 0:
                avg_h_xg = row.get('home_avg_goals_last5', 1.5)
            if pd.isna(avg_a_xg) or avg_a_xg == 0:
                avg_a_xg = row.get('away_avg_goals_last5', 1.2)
            if pd.isna(avg_h_xg): avg_h_xg = 1.5
            if pd.isna(avg_a_xg): avg_a_xg = 1.2
            avg_h_xg = float(np.clip(avg_h_xg, 0.3, 4.0))
            avg_a_xg = float(np.clip(avg_a_xg, 0.3, 4.0))

            p1_poi = px_poi = p2_poi = 0.0
            for h in range(10):
                for a in range(10):
                    p = poisson.pmf(h, avg_h_xg) * poisson.pmf(a, avg_a_xg)
                    if   h > a: p1_poi += p
                    elif h == a: px_poi += p
                    else:        p2_poi += p

            p1  = 0.5 * pv_h + 0.5 * p1_poi
            pxb = 0.5 * pv_x + 0.5 * px_poi
            p2  = 0.5 * pv_a + 0.5 * p2_poi
            tot = p1 + pxb + p2
            p1, pxb, p2 = p1/tot, pxb/tot, p2/tot

            bet_code, prob_dc, sim_odds, signal_type, is_dc = classify_signal(p1, pxb, p2)
            if bet_code is None or signal_type is None:
                continue
            if "❌ SKIP" in signal_type:
                skipped_odds += 1
                continue

            if not form_ok(p1, p2, row):
                skipped_form += 1
                continue

            # ÚPRAVA 1: Rebase po kole (14 dní), ne po počtu sázek
            # Nikdy nerebázujeme při aktivním cooling
            match_date = pd.Timestamp(row['match_date']) if pd.notnull(row.get('match_date')) else None
            if match_date and not cooling_active:
                if last_rebase_date is None or (match_date - last_rebase_date).days >= REBASE_DAYS:
                    new_flat = max(MIN_BET_AMOUNT, round(current_bankroll * 0.05 / 10) * 10)
                    if new_flat != flat_favorit:
                        flat_favorit = new_flat
                    last_rebase_date = match_date

            # ÚPRAVA 2: Cooling filtr — výpočet aktuální sázky
            if cooling_active:
                stake = max(MIN_BET_AMOUNT, round(flat_favorit * COOLING_FACTOR / 10) * 10)
            else:
                stake = flat_favorit
            stake = max(stake, MIN_BET_AMOUNT)

            # Výsledek
            won    = check_winner(bet_code, actual)
            profit = (stake * sim_odds) - stake if won else -stake
            current_bankroll += profit

            # ÚPRAVA 2: Aktualizace cooling stavu
            if won:
                consecutive_losses = 0
                if cooling_active:
                    consecutive_wins_cool += 1
                    if consecutive_wins_cool >= COOLING_WINS:
                        cooling_active        = False
                        consecutive_wins_cool = 0
            else:
                consecutive_wins_cool = 0
                consecutive_losses   += 1
                if consecutive_losses >= COOLING_LOSSES:
                    cooling_active = True

            cooling_flag = " 🧊" if cooling_active else ""
            date_str = match_date.strftime("%d.%m.%Y") if match_date else ""
            bets_log.append({
                "Date":     date_str,
                "Match":    f"{row.get('home_team','?')} vs {row.get('away_team','?')}",
                "Signal":   signal_type,
                "Bet":      bet_code,
                "Odds":     round(sim_odds, 2),
                "Stake":    round(stake, 2),
                "Cooling":  "🧊" if cooling_active else "",
                "Result":   "✅ WIN" if won else "❌ LOSS",
                "Profit":   round(profit, 2),
                "Bankroll": round(current_bankroll, 2)
            })

    return bets_log, current_bankroll, skipped_odds, skipped_form, skipped_season


# =============================================================================
# 8. HLAVNÍ FUNKCE
# =============================================================================

def main():
    print("=" * 70)
    print("🎯 STEP8 v7: WALK-FORWARD BACKTEST (Pure Favorit Strategy)")
    print("=" * 70)
    print(f"\n  Konfigurace:")
    print(f"    FLAT_FAVORIT={FLAT_STAKE_FAVORIT} Kč  REBASE každých {REBASE_DAYS} dní")
    print(f"    MIN_ODDS={MIN_ODDS_1X2}  THRESH_FAVORIT={THRESH_FAVORIT:.0%}")
    print(f"\n  Aktivní změny:")
    print(f"    ✅ Úprava 1: Rebase po kole ({REBASE_DAYS} dní), ne po počtu sázek")
    print(f"    ✅ Úprava 2: Cooling filtr ({COOLING_LOSSES} prohry → {COOLING_FACTOR*100:.0f}% sázka, reset po {COOLING_WINS} výhrách)")
    print(f"    ✅ Úprava 3: THRESH_FAVORIT={THRESH_FAVORIT:.0%} (bylo 53% v v6, zpět na 55%)")
    print(f"    ✅ Úprava 4: Sezónní filtr — zakázány měsíce 7 a 8 (červenec + srpen)")

    # Načtení features a threshold
    try:
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        thr_path     = os.path.join(MODEL_DIR, "draw_threshold.pkl")
        draw_threshold = joblib.load(thr_path) if os.path.exists(thr_path) else 0.37
        print(f"\n  ✅ feature_cols.pkl načten ({len(feature_cols)} features)")
        print(f"  ✅ Draw threshold: {draw_threshold:.2f}")
    except FileNotFoundError as e:
        print(f"  ❌ Chybí soubor: {e} — spusť step3 nejdřív")
        return

    print("\n" + "─" * 70)

    bets_log, end_bankroll, skipped_odds, skipped_form, skipped_season = run_backtest(feature_cols, draw_threshold)

    if not bets_log:
        print("⚠️  Žádné sázky nevznikly — zkontroluj data a prahy.")
        return

    stats = compute_stats(bets_log, START_BANKROLL, end_bankroll)
    print_report(bets_log, stats, START_BANKROLL, end_bankroll,
                 "SNIPER v7 Walk-Forward")

    # Cooling statistiky
    df_log    = pd.DataFrame(bets_log)
    n_cooling = (df_log['Cooling'] == '🧊').sum()

    total_candidates = len(bets_log) + skipped_odds + skipped_form + skipped_season
    print(f"\n  🔍 Filtrační souhrn (ze {total_candidates} kandidátních sázek):")
    print(f"     ❌ SKIP kurz    (MIN_ODDS filtr):      {skipped_odds:>4} sázek eliminováno")
    print(f"     ⛔ SKIP forma   (form_ok filtr):        {skipped_form:>4} sázek eliminováno")
    print(f"     📅 SKIP sezóna  (červenec + srpen):              {skipped_season:>4} sázek eliminováno")
    print(f"     ✅ Aktivní sázky:                       {len(bets_log):>4} sázek")
    print(f"     🧊 Z toho v cooling režimu:             {n_cooling:>4} sázek")

    # Uložení CSV
    csv_path = os.path.join(DATA_DIR, "backtest_sniper_v7.csv")
    pd.DataFrame(bets_log).to_csv(csv_path, index=False)
    print(f"\n  💾 Log uložen: {csv_path}")

    # Equity křivka (ASCII preview)
    df_log    = pd.DataFrame(bets_log)
    bankrolls = df_log['Bankroll'].values
    max_b     = bankrolls.max()
    min_b     = bankrolls.min()
    print(f"\n  📈 Equity křivka (min={min_b:.0f} Kč, max={max_b:.0f} Kč):")
    bar_width = 50
    for i in range(0, len(bankrolls), max(1, len(bankrolls) // 20)):
        b   = bankrolls[i]
        pct = (b - min_b) / (max_b - min_b + 1) if max_b != min_b else 0.5
        bar = "█" * int(pct * bar_width)
        print(f"  {i:>4} | {bar:<{bar_width}} {b:,.0f} Kč")


if __name__ == "__main__":
    main()