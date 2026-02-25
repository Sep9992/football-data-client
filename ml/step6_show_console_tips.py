"""
step6_show_console_tips.py  v3
================================
Sjednocení step6_1, step6_2, step6_3 pro architekturu v4 (step0–step5).

ZMĚNY oproti starým verzím:
  - Nečte z tabulky 'predictions' (ta v nové arch. neexistuje)
  - Čte features přímo z 'prepared_fixtures' (step2 tam ukládá 95 sloupců)
  - Načítá modely stejně jako step4 (voting_classifier, draw_threshold, mv_scaler)
  - Používá blended predikce (50% Voting + 50% Poisson) stejně jako step4
  - Přidává shodu modelů (Voting vs XGBoost) ze step5
  - Ukládá výsledky do tabulky 'predictions' pro auditní trail
  - Sjednocené prahy: THRESH_SUPER=0.85, MIN_ODDS_LIMIT=1.25 (konzervativnější z v2/v3)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy.stats import poisson

# =============================================================================
# 1. KONFIGURACE
# =============================================================================

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- SNIPER PRAHY (sjednoceno z v2 + v3, konzervativnější varianta) ---
THRESH_FAVORIT  = 0.55   # Čistý favorit (1 nebo 2)
THRESH_SAFE     = 0.75   # Neprohra (1X nebo X2)
THRESH_VALUE    = 0.55   # Value na outsidera (X2)
THRESH_SUPER    = 0.85   # Tutovka upgrade (SAFE → SAFE+)
MIN_ODDS_LIMIT  = 1.25   # Minimální odhadovaný tržní kurz
BOOKMAKER_MARGIN = 0.10  # Odhad marže sázkovky (10%)

MARKET_VALUES = {
    "Manchester City": 1290.0, "Arsenal FC": 1270.0, "Chelsea FC": 1160.0,
    "Liverpool FC": 1040.0, "Manchester United": 719.0, "Tottenham Hotspur": 877.0,
    "Newcastle United": 710.0, "Aston Villa": 532.0, "Brighton & Hove Albion": 510.0,
    "West Ham United": 339.0, "Nottingham Forest": 592.0, "Brentford": 434.0,
    "Crystal Palace": 536.0, "Wolverhampton Wanderers": 278.0, "Everton FC": 424.0,
    "Fulham FC": 373.0, "AFC Bournemouth": 447.0,
    "Leeds United": 321.0, "Burnley FC": 252.0, "AFC Sunderland": 327.0
}


# =============================================================================
# 2. DRAW BOOST (shodné se step4 a step5)
# =============================================================================

def predict_with_draw_boost(proba, threshold):
    preds = []
    for p in proba:
        p_away, p_draw, p_home = p[0], p[1], p[2]
        if p_draw >= threshold and p_draw > min(p_away, p_home):
            preds.append(1)
        else:
            preds.append(int(np.argmax(p)))
    return np.array(preds)


# =============================================================================
# 3. SIGNÁLOVÁ LOGIKA (sjednocení step6_1/2/3)
# =============================================================================

def classify_signal(ph, px, pa):
    """
    Vrátí (tip_label, strength, signal_note, skip).
    skip=True pokud odhadovaný tržní kurz < MIN_ODDS_LIMIT.
    """
    tip_label   = "-"
    strength    = 0.0
    signal_note = ""
    skip        = False

    if ph >= pa:
        # Domácí favorit nebo remíza nahoře
        if ph > THRESH_FAVORIT:
            tip_label   = "1"
            strength    = ph
            signal_note = "🔥 FAVORIT"
        elif (ph + px) > THRESH_SAFE:
            tip_label   = "1X"
            strength    = ph + px
            signal_note = "💎 SAFE+" if strength > THRESH_SUPER else "✅ SAFE"
        elif ph > 0:
            tip_label   = "1"
            strength    = ph
            signal_note = ""  # Slabý tip, bez signálu
    else:
        # Hosté favorit
        if pa > THRESH_FAVORIT:
            tip_label   = "2"
            strength    = pa
            signal_note = "🔥 FAVORIT"
        elif (pa + px) > THRESH_VALUE:
            tip_label   = "X2"
            strength    = pa + px
            signal_note = "💎 SAFE+" if strength > THRESH_SUPER else "✨ VALUE"
        elif pa > 0:
            tip_label   = "2"
            strength    = pa
            signal_note = ""

    # Anti-odpad filtr: odhadni tržní kurz = fair_odd * (1 - marže)
    if strength > 0:
        fair_odd         = 1.0 / strength
        est_market_odd   = fair_odd * (1 - BOOKMAKER_MARGIN)
        if est_market_odd < MIN_ODDS_LIMIT:
            signal_note = "❌ SKIP (nízký kurz)"
            skip        = True
    else:
        fair_odd = 0.0

    return tip_label, strength, fair_odd, signal_note, skip


# =============================================================================
# 4. PREDIKCE Z FEATURES ULOŽENÝCH V DB
# =============================================================================

def run_predictions(conn, feature_cols, voting_clf, xgb_clf,
                    poisson_h, poisson_a, xgb_reg_h, xgb_reg_a,
                    draw_threshold, mv_scaler):
    """
    Načte prepared_fixtures, doplní odvozené features, spustí modely.
    Vrátí DataFrame s výsledky.
    """
    fixtures = pd.read_sql(text("""
        SELECT *
        FROM prepared_fixtures
        WHERE match_date >= CURRENT_DATE
        ORDER BY match_date ASC
        LIMIT 20
    """), conn)

    if fixtures.empty:
        return pd.DataFrame()

    results = []

    for _, row in fixtures.iterrows():
        home = row.get('home_team', '?')
        away = row.get('away_team', '?')

        try:
            X = pd.DataFrame([row])

            # Dopočítej odvozené features pokud chybí v DB
            if 'market_value_diff_scaled' not in X.columns or pd.isna(X['market_value_diff_scaled'].iloc[0]):
                mv_h    = MARKET_VALUES.get(home, 200.0)
                mv_a    = MARKET_VALUES.get(away, 200.0)
                mv_diff = mv_h - mv_a
                if mv_scaler:
                    mv_df = pd.DataFrame([[mv_diff]], columns=['market_value_diff'])
                    X['market_value_diff_scaled'] = float(mv_scaler.transform(mv_df)[0][0])
                else:
                    X['market_value_diff_scaled'] = mv_diff / 400.0

            if 'home_x_elo' not in X.columns or pd.isna(X['home_x_elo'].iloc[0]):
                h_elo = float(X.get('home_elo', pd.Series([1500.0])).iloc[0] or 1500.0)
                h_pts = float(X.get('home_avg_points_last5', pd.Series([0.0])).iloc[0] or 0.0)
                X['home_x_elo'] = h_elo * (h_pts / 3.0)

            if 'elo_x_market' not in X.columns or pd.isna(X['elo_x_market'].iloc[0]):
                elo_diff = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
                X['elo_x_market'] = elo_diff * float(X['market_value_diff_scaled'].iloc[0])

            # Doplň chybějící features nulou
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X_input = X[feature_cols].astype(float)

            # A) Voting
            pv = voting_clf.predict_proba(X_input)[0]
            pv_a, pv_x, pv_h = pv[0], pv[1], pv[2]

            # B) XGBoost
            px_arr = xgb_clf.predict_proba(X_input)[0]
            px_a, px_x, px_h = px_arr[0], px_arr[1], px_arr[2]

            # C) xG (Poisson + XGBoost hybrid, clamp)
            gh = np.clip((poisson_h.predict(X_input)[0] + xgb_reg_h.predict(X_input)[0]) / 2, 0.1, 8.0)
            ga = np.clip((poisson_a.predict(X_input)[0] + xgb_reg_a.predict(X_input)[0]) / 2, 0.1, 8.0)

            # D) Poisson distribuce
            p1_poi = px_poi = p2_poi = 0.0
            for h in range(10):
                for a in range(10):
                    p = poisson.pmf(h, gh) * poisson.pmf(a, ga)
                    if   h > a: p1_poi += p
                    elif h == a: px_poi += p
                    else:        p2_poi += p

            # E) Blend 50/50 Voting + Poisson
            p1  = 0.5 * pv_h + 0.5 * p1_poi
            pxb = 0.5 * pv_x + 0.5 * px_poi
            p2  = 0.5 * pv_a + 0.5 * p2_poi
            total = p1 + pxb + p2
            p1, pxb, p2 = p1 / total, pxb / total, p2 / total

            # F) Finální tip s draw boost
            pred_class = predict_with_draw_boost(np.array([[p2, pxb, p1]]), draw_threshold)[0]

            # G) Shoda Voting vs XGBoost
            v_pred   = np.argmax([pv_a, pv_x, pv_h])
            xgb_pred = np.argmax([px_a, px_x, px_h])
            shoda    = "✅" if v_pred == xgb_pred else "❌"
            max_diff = max(abs(pv_h - px_h), abs(pv_x - px_x), abs(pv_a - px_a))

            # H) Signál (na blended pravděpodobnostech)
            tip_label, strength, fair_odd, signal_note, skip = classify_signal(p1, pxb, p2)

            results.append({
                'fixture_id':      row.get('fixture_id'),
                'match_date':      row.get('match_date'),
                'home_team':       home,
                'away_team':       away,
                'p1':              round(p1, 4),
                'px':              round(pxb, 4),
                'p2':              round(p2, 4),
                'xg_home':         round(gh, 2),
                'xg_away':         round(ga, 2),
                'tip':             tip_label,
                'strength':        round(strength, 4),
                'fair_odd':        round(fair_odd, 3),
                'signal':          signal_note,
                'skip':            skip,
                'shoda':           shoda,
                'max_diff':        round(max_diff, 4),
                'pred_class':      pred_class,
            })

        except Exception as e:
            print(f"  ⚠️  Chyba {home} vs {away}: {e}")

    return pd.DataFrame(results)


# =============================================================================
# 5. ULOŽENÍ DO DB (auditní trail)
# =============================================================================

def save_predictions(conn, df):
    """Uloží predikce do tabulky 'predictions' (vytvoří pokud neexistuje)."""
    if df.empty:
        return
    save_df = df[['fixture_id', 'p1', 'px', 'p2', 'xg_home', 'xg_away',
                  'tip', 'strength', 'fair_odd', 'signal', 'shoda']].copy()
    save_df.columns = ['fixture_id', 'proba_home_win', 'proba_draw', 'proba_away_win',
                       'xg_home', 'xg_away', 'predicted_tip', 'strength',
                       'fair_odd', 'signal', 'model_agreement']
    save_df['model_name'] = 'voting_blend_v4'
    save_df['created_at'] = pd.Timestamp.now()
    try:
        save_df.to_sql('predictions', conn, if_exists='replace', index=False)
        print(f"  💾 Predikce uloženy do tabulky 'predictions' ({len(save_df)} řádků)")
    except Exception as e:
        print(f"  ⚠️  Uložení selhalo: {e}")


# =============================================================================
# 6. ZOBRAZENÍ DASHBOARDU
# =============================================================================

def display_dashboard(df):
    if df.empty:
        print("📭 Žádné zápasy k zobrazení.")
        return

    print("\n" + "=" * 120)
    print(f"  {'DATUM':<14} {'ZÁPAS':<42} {'TIP':<5} {'SÍLA':>7} {'FÉR KURZ':>9}  "
          f"{'xG':^9}  {'SHODA':^5}  SIGNÁL")
    print("=" * 120)

    # Skupiny: nejdřív s jasným signálem, pak ostatní
    signaly   = df[df['signal'].str.startswith(('🔥', '💎', '✅', '✨'), na=False)]
    ostatni   = df[~df['fixture_id'].isin(signaly['fixture_id'])]

    for group_label, group_df in [("🎯 DOPORUČENÉ TIPY", signaly), ("📋 OSTATNÍ ZÁPASY", ostatni)]:
        if group_df.empty:
            continue
        print(f"\n  {group_label}")
        print("  " + "-" * 116)
        for _, r in group_df.iterrows():
            date_str  = pd.Timestamp(r['match_date']).strftime("%d.%m. %H:%M") if pd.notnull(r['match_date']) else "???"
            match_str = f"{r['home_team']} vs {r['away_team']}"
            xg_str    = f"{r['xg_home']:.2f}:{r['xg_away']:.2f}"

            # Zbarvení řádku podle signálu
            skip_mark = " ⚠️" if r['skip'] else ""

            print(f"  {date_str:<14} {match_str:<42} {r['tip']:<5} "
                  f"{r['strength']*100:>5.1f}%  {r['fair_odd']:>7.2f}  "
                  f"{xg_str:^9}  {r['shoda']:^5}  {r['signal']}{skip_mark}")

    print("\n" + "=" * 120)
    print(f"  ℹ️  Legenda signálů:")
    print(f"     🔥 FAVORIT  = P(výsledku) > {THRESH_FAVORIT:.0%}  →  sázej přímo")
    print(f"     💎 SAFE+    = P(neprohra) > {THRESH_SUPER:.0%}  →  nejsilnější pojistka")
    print(f"     ✅ SAFE     = P(neprohra) > {THRESH_SAFE:.0%}  →  sázej na jistotu")
    print(f"     ✨ VALUE    = P(neprohra host) > {THRESH_VALUE:.0%}  →  value outsider")
    print(f"     ❌ SKIP     = Odhadovaný tržní kurz < {MIN_ODDS_LIMIT}  →  nevýhodné")
    print(f"  ℹ️  Shoda modelů: ✅ = Voting i XGBoost tipují stejně  |  ❌ = vyšší nejistota")
    print(f"  ℹ️  Kurzy jsou férové (bez marže). Tržní kurz ≈ zobrazený × {1 - BOOKMAKER_MARGIN:.2f}")

    # Souhrn
    doporucene = df[df['signal'].str.startswith(('🔥', '💎', '✅', '✨'), na=False) & ~df['skip']]
    if not doporucene.empty:
        print(f"\n  📊 SOUHRN: {len(doporucene)} aktivních tipů z {len(df)} zápasů")
        print(f"     Shoda modelů: {(doporucene['shoda'] == '✅').sum()}/{len(doporucene)}")
        print(f"     Průměrný fér kurz: {doporucene['fair_odd'].mean():.2f}")


# =============================================================================
# 7. HLAVNÍ FUNKCE
# =============================================================================

def main():
    print("=" * 70)
    print("💰 STEP6 v3: DASHBOARD TIPŮ")
    print("=" * 70)

    # Načtení modelů
    print("\n📦 Načítám modely...")
    try:
        voting_clf   = joblib.load(os.path.join(MODEL_DIR, "voting_classifier.pkl"))
        xgb_clf      = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier.pkl"))
        poisson_h    = joblib.load(os.path.join(MODEL_DIR, "poisson_home_goals.pkl"))
        poisson_a    = joblib.load(os.path.join(MODEL_DIR, "poisson_away_goals.pkl"))
        xgb_reg_h    = joblib.load(os.path.join(MODEL_DIR, "xgb_home_goals.pkl"))
        xgb_reg_a    = joblib.load(os.path.join(MODEL_DIR, "xgb_away_goals.pkl"))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

        thr_path       = os.path.join(MODEL_DIR, "draw_threshold.pkl")
        draw_threshold = joblib.load(thr_path) if os.path.exists(thr_path) else 0.37

        mv_path   = os.path.join(MODEL_DIR, "market_value_scaler.pkl")
        mv_scaler = joblib.load(mv_path) if os.path.exists(mv_path) else None

        print(f"  ✅ Modely načteny | Features: {len(feature_cols)} | Draw thr: {draw_threshold:.2f}")
        print(f"  ✅ MV scaler: {'načten' if mv_scaler else '⚠️  chybí (fallback)'}")

    except FileNotFoundError as e:
        print(f"  ❌ Chybí model: {e}")
        print(f"     Spusť nejdřív step3.")
        return

    # Predikce a uložení
    with engine.begin() as conn:
        print("\n🔮 Spouštím predikce...")
        df = run_predictions(conn, feature_cols, voting_clf, xgb_clf,
                             poisson_h, poisson_a, xgb_reg_h, xgb_reg_a,
                             draw_threshold, mv_scaler)

        if df.empty:
            print("📭 Žádné nadcházející zápasy v DB.")
            return

        print(f"  ✅ Zpracováno {len(df)} zápasů")
        save_predictions(conn, df)

    # Dashboard výstup
    display_dashboard(df)


if __name__ == "__main__":
    main()