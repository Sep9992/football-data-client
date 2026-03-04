"""
step7_show_console_tips.py  v4
================================
Rozšíření v4: integrace reálných bookmaker kurzů (The Odds API via step7_fetch_odds).

NOVÉ v4:
  - Načítá kurzy z tabulky bookmaker_odds (plněna step7_fetch_odds.py)
  - Vypočítá VALUE % = model_prob × bookmaker_odd − 1  (>0 = value bet)
  - Doporučená sázka: Kelly criterion (frakční, 25% Kelly)
  - Porovnání best odds přes všechny dostupné bookmakers
  - Dashboard řadí doporučené tipy dle value% (od nejvyšší)
  - Bez kurzů: funguje stejně jako v3 (férové kurzy)

WORKFLOW:
  1. python step7_fetch_odds.py   ← stáhne kurzy 1× před kolem
  2. python step6_show_console_tips.py  ← dashboard s value betting

ENV (volitelné):
  ODDS_BANKROLL=10000   ← bankroll pro Kelly výpočet (default 10000 Kč)
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
    # --- PREMIER LEAGUE ---
    "Manchester City": 1290.0, "Arsenal FC": 1270.0, "Chelsea FC": 1160.0,
    "Liverpool FC": 1040.0, "Manchester United": 719.0, "Tottenham Hotspur": 877.0,
    "Newcastle United": 710.0, "Aston Villa": 532.0, "Brighton & Hove Albion": 510.0,
    "West Ham United": 339.0, "Nottingham Forest": 592.0, "Brentford": 434.0,
    "Crystal Palace": 536.0, "Wolverhampton Wanderers": 278.0, "Everton FC": 424.0,
    "Fulham FC": 373.0, "AFC Bournemouth": 447.0,
    "Leeds United": 321.0, "Burnley FC": 252.0, "AFC Sunderland": 327.0,
    # --- CHANCE LIGA ---
    "Sparta Praha": 82.0, "Slavia Praha": 75.0, "Viktoria Plzeň": 38.0,
    "Baník Ostrava": 22.0, "Mladá Boleslav": 18.0, "Bohemians": 14.0,
    "Slovácko": 13.0, "Sigma Olomouc": 14.0, "Hradec Králové": 12.0,
    "Slovan Liberec": 16.0, "Teplice": 11.0, "Jablonec": 12.0,
    "Pardubice": 15.0, "Zlín": 9.0, "Karviná": 10.0,
    "České Budějovice": 10.0, "Dukla Praha": 8.0, "Zbrojovka Brno": 9.0,
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
    Vrátí (tip_label, strength, fair_odd, original_signal, skip).
    original_signal zachován i když skip=True (pro zobrazení v dashboardu).
    """
    tip_label       = "-"
    strength        = 0.0
    original_signal = ""
    skip            = False

    if ph >= pa:
        if ph > THRESH_FAVORIT:
            tip_label       = "1"
            strength        = ph
            original_signal = "🔥 FAVORIT"
        elif (ph + px) > THRESH_SAFE:
            tip_label       = "1X"
            strength        = ph + px
            original_signal = "💎 SAFE+" if strength > THRESH_SUPER else "✅ SAFE"
        elif ph > 0:
            tip_label       = "1"
            strength        = ph
    else:
        if pa > THRESH_FAVORIT:
            tip_label       = "2"
            strength        = pa
            original_signal = "🔥 FAVORIT"
        elif (pa + px) > THRESH_VALUE:
            tip_label       = "X2"
            strength        = pa + px
            original_signal = "💎 SAFE+" if strength > THRESH_SUPER else "✨ VALUE"
        elif pa > 0:
            tip_label       = "2"
            strength        = pa

    fair_odd = 1.0 / strength if strength > 0 else 0.0

    # Anti-odpad filtr: odhadni tržní kurz
    if strength > 0:
        est_market_odd = fair_odd * (1 - BOOKMAKER_MARGIN)
        if est_market_odd < MIN_ODDS_LIMIT:
            skip = True

    return tip_label, strength, fair_odd, original_signal, skip


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
          AND match_date <= CURRENT_DATE + INTERVAL '14 days'
        ORDER BY match_date ASC, league ASC
    """), conn)

    if fixtures.empty:
        return pd.DataFrame()

    results = []

    for _, row in fixtures.iterrows():
        home   = row.get('home_team', '?')
        away   = row.get('away_team', '?')
        league = row.get('league', '?')

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

            # OPRAVA: home_x_elo = elo_diff (step2 definice), ne h_elo * pts/3.0
            if 'home_x_elo' not in X.columns or pd.isna(X['home_x_elo'].iloc[0]):
                elo_diff_val = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
                X['home_x_elo'] = elo_diff_val

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
            tip_label, strength, fair_odd, original_signal, skip = classify_signal(p1, pxb, p2)

            # I) Confidence score
            confidence = max(p1, pxb, p2) / (1/3)

            results.append({
                'fixture_id':      row.get('fixture_id'),
                'match_date':      row.get('match_date'),
                'league':          league,
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
                'signal':          original_signal,
                'skip':            skip,
                'shoda':           shoda,
                'max_diff':        round(max_diff, 4),
                'pred_class':      pred_class,
                'confidence':      round(confidence, 3),
            })

        except Exception as e:
            print(f"  ⚠️  Chyba {home} vs {away}: {e}")

    return pd.DataFrame(results)


# =============================================================================
# 5. ULOŽENÍ DO DB (auditní trail)
# =============================================================================

def save_predictions(conn, df):
    """Uloží predikce do tabulky 'predictions' (append = auditní trail zachován).
    Automaticky přidá nové sloupce pokud tabulka existuje ze staré verze.
    """
    if df.empty:
        return

    save_df = df[['fixture_id', 'p1', 'px', 'p2', 'xg_home', 'xg_away',
                  'tip', 'strength', 'fair_odd', 'signal', 'skip', 'shoda', 'confidence']].copy()
    save_df.columns = ['fixture_id', 'proba_home_win', 'proba_draw', 'proba_away_win',
                       'xg_home', 'xg_away', 'predicted_tip', 'strength',
                       'fair_odd', 'signal', 'skip', 'model_agreement', 'confidence']
    save_df['model_name'] = 'voting_blend_v4'
    save_df['created_at'] = pd.Timestamp.now()

    # Auto-migrace: přidej nové sloupce pokud chybí
    new_cols = {
        'skip':       'BOOLEAN',
        'confidence': 'NUMERIC',
    }
    for col, col_type in new_cols.items():
        try:
            conn.execute(text(
                f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {col_type}"
            ))
        except Exception:
            pass  # Tabulka ještě neexistuje → vytvoří ji to_sql níže

    try:
        fixture_ids = save_df['fixture_id'].dropna().tolist()
        if fixture_ids:
            placeholders = ','.join([':id' + str(i) for i in range(len(fixture_ids))])
            params = {'id' + str(i): fid for i, fid in enumerate(fixture_ids)}
            conn.execute(text(
                f"DELETE FROM predictions WHERE fixture_id IN ({placeholders})"
            ), params)
        save_df.to_sql('predictions', conn, if_exists='append', index=False)
        print(f"  💾 Predikce uloženy do tabulky 'predictions' ({len(save_df)} řádků)")
    except Exception as e:
        print(f"  ⚠️  Uložení selhalo: {e}")


# =============================================================================
# 6. ZOBRAZENÍ DASHBOARDU
# =============================================================================

def display_dashboard(df: pd.DataFrame):
    if df.empty:
        print("📭 Žádné zápasy k zobrazení.")
        return

    has_odds = df['has_odds'].any() if 'has_odds' in df.columns else False

    # Hlavička — se nebo bez kurz sloupců
    if has_odds:
        print("\n" + "=" * 148)
        print(f"  {'DATUM':<14} {'ZÁPAS':<47} {'TIP':<5} {'SÍLA':>7} {'FÉR':>5} "
              f"{'BM KURZ':>8} {'VALUE':>7} {'KELLY':>7}  {'xG':^9}  {'CONF':>5}  {'SHODA':^5}  SIGNÁL")
        print("=" * 148)
    else:
        print("\n" + "=" * 128)
        print(f"  {'DATUM':<14} {'ZÁPAS':<47} {'TIP':<5} {'SÍLA':>7} {'FÉR KURZ':>9}  "
              f"{'xG':^9}  {'CONF':>5}  {'SHODA':^5}  SIGNÁL")
        print("=" * 128)

    sep = "=" * (148 if has_odds else 128)

    # 3 skupiny
    # Přesuň SKIP tipy s reálnou pozitivní value do DOPORUČENÝCH
    # (MIN_ODDS_LIMIT filtruje odhadovaný kurz, ale pokud bookmaker nabídne vyšší → value existuje)
    has_pos_value = (
        df['has_odds'].fillna(False) &
        df['value_pct'].notna() &
        (df['value_pct'] > 0)
    ) if 'has_odds' in df.columns else pd.Series(False, index=df.index)

    strong_signal = df['signal'].str.startswith(('🔥', '💎', '✅', '✨'), na=False)

    doporucene = df[strong_signal & (~df['skip'] | has_pos_value)]
    skip_group = df[df['skip'] & ~has_pos_value]
    ostatni    = df[~df['fixture_id'].isin(doporucene['fixture_id']) &
                   ~df['fixture_id'].isin(skip_group['fixture_id'])]

    # Seřaď DOPORUČENÉ podle value_pct DESC (pokud jsou kurzy), pak podle confidence
    if has_odds and 'value_pct' in doporucene.columns:
        # Seřaď: nejdřív tipy s reálnou value (sestupně), pak ostatní dle confidence
        doporucene = doporucene.copy()
        doporucene['_sort_val'] = doporucene['value_pct'].infer_objects(copy=False).fillna(-999)
        doporucene = doporucene.sort_values('_sort_val', ascending=False).drop(columns='_sort_val')

    groups = [
        ("🎯 DOPORUČENÉ TIPY", doporucene, False),
        ("⏭️  PŘESKOČIT (dobrý signál, nízký kurz)", skip_group, True),
        ("📋 OSTATNÍ ZÁPASY", ostatni, False),
    ]

    for group_label, group_df, is_skip in groups:
        if group_df.empty:
            continue
        print(f"\n  {group_label}")
        print("  " + "-" * (146 if has_odds else 124))

        for _, r in group_df.iterrows():
            date_str  = pd.Timestamp(r['match_date']).strftime("%d.%m. %H:%M") if pd.notnull(r['match_date']) else "???"
            league    = r.get('league', '?')
            match_str = f"[{league}] {r['home_team']} vs {r['away_team']}"
            xg_str    = f"{r['xg_home']:.2f}:{r['xg_away']:.2f}"
            conf_str  = f"{r.get('confidence', 0):.2f}x"
            signal    = r['signal'] if r['signal'] else ""
            skip_sfx  = "  ❌ SKIP" if is_skip else ""

            if has_odds:
                bm_odd   = r.get('bm_odd')
                val_pct  = r.get('value_pct')
                kelly_kc = r.get('kelly_kc')

                bm_str  = f"{bm_odd:.2f}" if bm_odd else "  -  "
                val_str = f"{val_pct:+.1f}%" if val_pct is not None else "   -  "
                kel_str = f"{kelly_kc} Kč"  if kelly_kc else "  -  "

                # Zvýrazni value sázky
                val_flag = " 💰" if (val_pct is not None and val_pct > 0) else ""

                print(f"  {date_str:<14} {match_str:<47} {r['tip']:<5} "
                      f"{r['strength']*100:>5.1f}%  {r['fair_odd']:>4.2f} "
                      f"{bm_str:>8} {val_str:>7} {kel_str:>7}  "
                      f"{xg_str:^9}  {conf_str:>5}  {r['shoda']:^5}  {signal}{skip_sfx}{val_flag}")
            else:
                print(f"  {date_str:<14} {match_str:<47} {r['tip']:<5} "
                      f"{r['strength']*100:>5.1f}%  {r['fair_odd']:>7.2f}  "
                      f"{xg_str:^9}  {conf_str:>5}  {r['shoda']:^5}  {signal}{skip_sfx}")

    print("\n" + sep)

    # Legenda
    print(f"  ℹ️  Legenda signálů:")
    print(f"     🔥 FAVORIT  = P(výsledku) > {THRESH_FAVORIT:.0%}  →  sázej přímo")
    print(f"     💎 SAFE+    = P(neprohra) > {THRESH_SUPER:.0%}  →  nejsilnější pojistka")
    print(f"     ✅ SAFE     = P(neprohra) > {THRESH_SAFE:.0%}  →  sázej na jistotu")
    print(f"     ✨ VALUE    = P(neprohra host) > {THRESH_VALUE:.0%}  →  value outsider")
    print(f"     ❌ SKIP     = Odhadovaný tržní kurz < {MIN_ODDS_LIMIT}  →  nevýhodné")
    if has_odds:
        print(f"     💰          = Pozitivní value (model_prob × BM kurz > 1.0)")
        print(f"  ℹ️  VALUE % = model_prob × bookmaker_odd − 1  (>0 = value bet)")
        print(f"  ℹ️  KELLY   = Doporučená sázka ({KELLY_FRACTION*100:.0f}% Kelly, bankroll {BANKROLL:,.0f} Kč)")
    print(f"  ℹ️  Conf = P(nejsilnějšího výsledku) / baseline(33%)  →  >1.5x = silný tip")
    print(f"  ℹ️  Shoda modelů: ✅ = Voting i XGBoost tipují stejně  |  ❌ = vyšší nejistota")
    print(f"  ℹ️  FÉR = férový kurz (bez marže). Tržní kurz ≈ FÉR × {1 - BOOKMAKER_MARGIN:.2f}")

    # Souhrn
    doporucene_live = df[df['signal'].str.startswith(('🔥', '💎', '✅', '✨'), na=False) & ~df['skip']]
    if not doporucene_live.empty:
        if has_odds and 'value_pct' in doporucene_live.columns:
            value_bets = doporucene_live[
                doporucene_live['value_pct'].notna() & (doporucene_live['value_pct'] > 0)
            ]
        print(f"\n  📊 SOUHRN: {len(doporucene_live)} aktivních tipů  |  "
              f"{len(skip_group)} přeskočeno  |  "
              f"{len(ostatni)} ostatní  |  celkem {len(df)} zápasů")
        print(f"     Shoda modelů (aktivní): {(doporucene_live['shoda'] == '✅').sum()}/{len(doporucene_live)}")
        print(f"     Průměrný fér kurz: {doporucene_live['fair_odd'].mean():.2f}")
        print(f"     Průměrná confidence: {doporucene_live['confidence'].mean():.2f}x")
        if has_odds and not value_bets.empty:
            print(f"     💰 Value sázky: {len(value_bets)}/{len(doporucene_live)}  "
                  f"| Avg value: {value_bets['value_pct'].mean():+.1f}%  "
                  f"| Celkem Kelly: {value_bets['kelly_kc'].sum():,.0f} Kč")


# =============================================================================
# 7. HLAVNÍ FUNKCE
# =============================================================================

# =============================================================================
# 8. OBOHACENÍ O REÁLNÉ KURZY (bookmaker_odds z step7_fetch_odds)
# =============================================================================

BANKROLL = float(os.getenv("ODDS_BANKROLL", "10000"))
KELLY_FRACTION = 0.25   # Frakce Kelly (konzervativní: čtvrtinový Kelly)

def enrich_with_odds(df: pd.DataFrame, conn) -> pd.DataFrame:
    """
    Přidá k predikcím reálné bookmaker kurzy z tabulky bookmaker_odds.
    Vypočítá: value_pct, kelly_stake.
    Vrátí rozšířený DataFrame.
    """
    # Výchozí hodnoty
    df['bm_odd']    = None
    df['bm_book']   = '-'
    df['value_pct'] = None
    df['kelly_kc']  = None
    df['has_odds']  = False

    try:
        odds_df = pd.read_sql(text("""
            SELECT fixture_id, odd_1, odd_x, odd_2, book_1, book_x, book_2
            FROM bookmaker_odds
            WHERE fixture_id IS NOT NULL
        """), conn)
    except Exception:
        return df  # Tabulka ještě neexistuje → tiché pokračování

    if odds_df.empty:
        return df

    odds_map = {row['fixture_id']: row for _, row in odds_df.iterrows()}

    for idx, row in df.iterrows():
        fid = row.get('fixture_id')
        if fid not in odds_map:
            continue

        o = odds_map[fid]
        tip  = row['tip']
        prob = float(row['strength'])   # model pravděpodobnost pro tip

        # ----------------------------------------------------------------
        # DŮLEŽITÉ: The Odds API (free tier) poskytuje jen h2h trh (1/X/2).
        # Pro čisté výhry (1, 2) → přímý kurz, VALUE výpočet je přesný.
        # Pro double chance (1X, X2, X) → kurzy na DC trh NEJSOU dostupné.
        #   Odhadneme DC kurz ze sčítání pravděpodobností bookmakers:
        #   DC_odd_1X ≈ 1 / (1/odd_1 + 1/odd_x)   (bez marže)
        #   DC_odd_X2 ≈ 1 / (1/odd_x + 1/odd_2)
        # ----------------------------------------------------------------
        bm_odd = None
        bm_bk  = '-'

        if tip == '1':
            bm_odd = o.get('odd_1')
            bm_bk  = o.get('book_1', '-')
        elif tip == '2':
            bm_odd = o.get('odd_2')
            bm_bk  = o.get('book_2', '-')
        elif tip == 'X':
            bm_odd = o.get('odd_x')
            bm_bk  = o.get('book_x', '-')
        elif tip == '1X':
            # Odhadni DC kurz: 1 / (1/odd_1 + 1/odd_x)
            o1 = o.get('odd_1')
            ox = o.get('odd_x')
            if o1 and ox and float(o1) > 1.0 and float(ox) > 1.0:
                bm_odd = round(1.0 / (1.0/float(o1) + 1.0/float(ox)), 2)
                bm_bk  = f"DC({o.get('book_1','-')}+{o.get('book_x','-')})"
        elif tip == 'X2':
            # Odhadni DC kurz: 1 / (1/odd_x + 1/odd_2)
            ox = o.get('odd_x')
            o2 = o.get('odd_2')
            if ox and o2 and float(ox) > 1.0 and float(o2) > 1.0:
                bm_odd = round(1.0 / (1.0/float(ox) + 1.0/float(o2)), 2)
                bm_bk  = f"DC({o.get('book_x','-')}+{o.get('book_2','-')})"

        if not bm_odd or float(bm_odd) <= 1.0:
            continue

        bm_odd = float(bm_odd)

        # Value % = model_prob × bookmaker_odd − 1
        value  = prob * bm_odd - 1.0

        # Kelly stake (frakční): f = (edge / (odd - 1)) × bankroll
        # edge = value_pct, odd - 1 = čistý výnos na 1 Kč
        kelly_full = value / (bm_odd - 1.0) if bm_odd > 1.0 else 0.0
        kelly_frac = max(0.0, kelly_full * KELLY_FRACTION * BANKROLL)
        kelly_frac = round(kelly_frac / 50) * 50  # Zaokrouhli na 50 Kč

        df.at[idx, 'bm_odd']    = round(bm_odd, 2)
        df.at[idx, 'bm_book']   = bm_bk
        df.at[idx, 'value_pct'] = round(value * 100, 1)
        df.at[idx, 'kelly_kc']  = int(kelly_frac)
        df.at[idx, 'has_odds']  = True

    n_matched = df['has_odds'].sum()
    if n_matched > 0:
        print(f"  💹 Reálné kurzy: {n_matched}/{len(df)} zápasů spárováno "
              f"(bankroll: {BANKROLL:,.0f} Kč, Kelly ×{KELLY_FRACTION})")
    else:
        print(f"  ℹ️  Reálné kurzy nejsou k dispozici — spusť step7_fetch_odds.py")

    return df


# =============================================================================
# 9. HLAVNÍ FUNKCE
# =============================================================================

def main():
    print("=" * 70)
    print("💰 STEP7 v4: DASHBOARD TIPŮ + VALUE BETTING")
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

    # Predikce, obohacení o odds, uložení
    with engine.begin() as conn:
        print("\n🔮 Spouštím predikce...")
        df = run_predictions(conn, feature_cols, voting_clf, xgb_clf,
                             poisson_h, poisson_a, xgb_reg_h, xgb_reg_a,
                             draw_threshold, mv_scaler)

        if df.empty:
            print("📭 Žádné nadcházející zápasy v DB.")
            return

        print(f"  ✅ Zpracováno {len(df)} zápasů")

        # Obohacení o reálné kurzy
        df = enrich_with_odds(df, conn)

        save_predictions(conn, df)

    # Dashboard výstup
    display_dashboard(df)


if __name__ == "__main__":
    main()