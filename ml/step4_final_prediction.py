""""
step4_final_prediction.py  v5
==============================
ZMĚNY v5:
  - Načítá market_value_scaler.pkl (uložený step2) → opravena exploze xG
  - Počítá odvozené features (home_x_elo, elo_x_market) konzistentně se step2
  - Draw boost threshold načítá z draw_threshold.pkl (uložený step3)
  - Blendované pravděpodobnosti: 50% klasifikátor + 50% Poisson
  - Safety clamp: xG max 8.0 (ochrana před numerickými extrémy)
"""

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from scipy.stats import poisson

# --- 1. KONFIGURACE ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MARKET_VALUES = {
    "Manchester City": 1290.0, "Arsenal FC": 1270.0, "Chelsea FC": 1160.0,
    "Liverpool FC": 1040.0, "Manchester United": 719.0, "Tottenham Hotspur": 877.0,
    "Newcastle United": 710.0, "Aston Villa": 532.0, "Brighton & Hove Albion": 510.0,
    "West Ham United": 339.0, "Nottingham Forest": 592.0, "Brentford": 434.0,
    "Crystal Palace": 536.0, "Wolverhampton Wanderers": 278.0, "Everton FC": 424.0,
    "Fulham FC": 373.0, "AFC Bournemouth": 447.0,
    "Leeds United": 321.0, "Burnley FC": 252.0, "AFC Sunderland": 327.0
}

STATS_TO_ROLL = [
    'goals', 'possession',
    'shots', 'shots_on_target', 'shots_off_target', 'blocked_shots',
    'passes_total', 'passes_completed',
    'expected_goals', 'corners', 'free_kicks', 'throw_ins',
    'fouls', 'yellow_cards', 'red_cards',
    'saves', 'offsides', 'interceptions'
]

CONCEDED_STATS = ['expected_goals', 'shots', 'shots_on_target', 'goals']


# =============================================================================
# DRAW RECALL BOOST (zrcadlí step3)
# =============================================================================

def predict_with_draw_boost(proba, threshold):
    """
    Predikuj Draw pokud P(Draw) >= threshold AND P(Draw) > min(P(Away), P(Home)).
    Druhá podmínka zabrání přebití jasných Away/Home výsledků.
    """
    preds = []
    for p in proba:
        p_away, p_draw, p_home = p[0], p[1], p[2]
        if p_draw >= threshold and p_draw > min(p_away, p_home):
            preds.append(1)
        else:
            preds.append(int(np.argmax(p)))
    return np.array(preds)


# =============================================================================
# HISTORICKÁ DATA PRO ROLLING FEATURES
# =============================================================================

def get_history(conn, team, match_date):
    """Získá posledních 5 odehraných zápasů týmu před datem."""
    if isinstance(match_date, datetime):
        match_date = match_date.date()
    query = text("""
        SELECT f.match_date,
               CASE WHEN f.home_team = :team THEN 'home' ELSE 'away' END as side,
               s.*
        FROM fixtures f
        JOIN match_statistics s ON f.id = s.fixture_id
        WHERE (f.home_team = :team OR f.away_team = :team)
          AND f.match_date < :date
          AND s.goals_home IS NOT NULL
        ORDER BY f.match_date DESC
        LIMIT 5
    """)
    return pd.read_sql(query, conn, params={"team": team, "date": match_date})


def get_rolling_features(conn, home_team, away_team, m_date, fixture_id, mv_scaler):
    """
    Vypočítá rolling averages, ELO a odvozené features pro jeden zápas.
    mv_scaler: sklearn StandardScaler načtený z market_value_scaler.pkl — KRITICKÉ!
    """
    h_hist = get_history(conn, home_team, m_date)
    a_hist = get_history(conn, away_team, m_date)

    if h_hist.empty or a_hist.empty:
        return None

    feats = {}

    # --- ROLLING AVERAGES ---
    for prefix, hist in [("home", h_hist), ("away", a_hist)]:
        for stat in STATS_TO_ROLL:
            vals = []
            for _, r in hist.iterrows():
                side = r['side']
                v = r.get(f"{stat}_{side}", 0)
                if v is None:
                    v = 0
                if stat == 'possession' and float(v) > 1.0:
                    v = float(v) / 100.0
                vals.append(float(v))
            feats[f"{prefix}_avg_{stat}_last5"] = np.mean(vals) if vals else 0.0

            if stat in CONCEDED_STATS:
                conc_vals = []
                for _, r in hist.iterrows():
                    opp_side = 'away' if r['side'] == 'home' else 'home'
                    v_opp = r.get(f"{stat}_{opp_side}", 0)
                    conc_vals.append(float(v_opp) if v_opp is not None else 0.0)
                feats[f"{prefix}_avg_{stat}_conceded_last5"] = np.mean(conc_vals) if conc_vals else 0.0

    # --- MARKET VALUE ---
    mv_h = MARKET_VALUES.get(home_team, 200.0)
    mv_a = MARKET_VALUES.get(away_team, 200.0)
    mv_diff = mv_h - mv_a
    feats['market_value_home'] = mv_h
    feats['market_value_away'] = mv_a
    feats['market_value_diff'] = mv_diff

    # KRITICKÉ: použij stejný scaler jako step2, jinak Poisson dostane raw diff
    # (např. Liverpool vs Wolves: 1040-278 = 762) místo normalizované hodnoty ±2
    # → exploze xG na astronomická čísla
    if mv_scaler is not None:
        # Předej DataFrame se jménem sloupce — scaler byl natrénován s feature names
        mv_df = pd.DataFrame([[mv_diff]], columns=['market_value_diff'])
        feats['market_value_diff_scaled'] = float(mv_scaler.transform(mv_df)[0][0])
    else:
        # Fallback: hrubá normalizace (std tréninkové distribuce ≈ 400)
        feats['market_value_diff_scaled'] = mv_diff / 400.0

    # --- REST DAYS ---
    if isinstance(m_date, datetime):
        m_date = m_date.date()
    last_h = h_hist.iloc[0]['match_date']
    last_a = a_hist.iloc[0]['match_date']
    if isinstance(last_h, datetime): last_h = last_h.date()
    if isinstance(last_a, datetime): last_a = last_a.date()
    feats['home_rest_days'] = (m_date - last_h).days
    feats['away_rest_days'] = (m_date - last_a).days

    # --- ELO z prepared_fixtures ---
    elo_res = pd.read_sql(
        text("SELECT home_elo, away_elo, elo_diff FROM prepared_fixtures WHERE fixture_id = :fid"),
        conn, params={"fid": fixture_id}
    )
    if not elo_res.empty:
        feats['home_elo'] = float(elo_res.iloc[0]['home_elo'])
        feats['away_elo'] = float(elo_res.iloc[0]['away_elo'])
        feats['elo_diff'] = float(elo_res.iloc[0]['elo_diff'])
    else:
        h_elo_res = pd.read_sql(
            text("SELECT home_elo FROM prepared_fixtures WHERE home_team = :t ORDER BY match_date DESC LIMIT 1"),
            conn, params={"t": home_team}
        )
        a_elo_res = pd.read_sql(
            text("SELECT away_elo FROM prepared_fixtures WHERE away_team = :t ORDER BY match_date DESC LIMIT 1"),
            conn, params={"t": away_team}
        )
        h_elo = float(h_elo_res.iloc[0]['home_elo']) if not h_elo_res.empty else 1500.0
        a_elo = float(a_elo_res.iloc[0]['away_elo']) if not a_elo_res.empty else 1500.0
        feats['home_elo'] = h_elo
        feats['away_elo'] = a_elo
        feats['elo_diff'] = h_elo - a_elo

    # --- ODVOZENÉ FEATURES (odpovídá step2/create_enhanced_features) ---
    # home_x_elo: home_elo × (home_avg_points_last5 / 3.0)
    feats['home_x_elo']    = feats['home_elo'] * (feats.get('home_avg_points_last5', 0.0) / 3.0)
    # elo_x_market: elo_diff × market_value_diff_scaled
    feats['elo_x_market']  = feats['elo_diff'] * feats['market_value_diff_scaled']
    # home_advantage (konstanta, ale může být ve feature listu)
    feats['home_advantage'] = 1.0

    return feats


# =============================================================================
# HLAVNÍ FUNKCE
# =============================================================================

def main():
    print("🚀 Spouštím FINÁLNÍ predikci (best_classifier + regressory)...")

    # --- NAČTENÍ MODELŮ ---
    try:
        best_clf         = joblib.load(os.path.join(MODEL_DIR, "best_classifier.pkl"))
        reg_h            = joblib.load(os.path.join(MODEL_DIR, "poisson_home_goals.pkl"))
        reg_a            = joblib.load(os.path.join(MODEL_DIR, "poisson_away_goals.pkl"))
        xgb_reg_h        = joblib.load(os.path.join(MODEL_DIR, "xgb_home_goals.pkl"))
        xgb_reg_a        = joblib.load(os.path.join(MODEL_DIR, "xgb_away_goals.pkl"))
        trained_features = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

        thr_path       = os.path.join(MODEL_DIR, "draw_threshold.pkl")
        draw_threshold = joblib.load(thr_path) if os.path.exists(thr_path) else 0.22

        mv_scaler_path = os.path.join(MODEL_DIR, "market_value_scaler.pkl")
        mv_scaler_abs  = os.path.abspath(mv_scaler_path)
        if os.path.exists(mv_scaler_abs):
            mv_scaler = joblib.load(mv_scaler_abs)
            mv_status = f"✅  mean={mv_scaler.mean_[0]:.1f}, std={mv_scaler.scale_[0]:.1f}"
        else:
            mv_scaler = None
            mv_status = f"⚠️  CHYBÍ!\n   Hledám v: {mv_scaler_abs}\n   Spusť step2 znovu."

        print(f"✅ Modely načteny. Očekávají {len(trained_features)} features.")
        print(f"   Draw threshold:     {draw_threshold:.2f}")
        print(f"   MV scaler:          {mv_status}")

    except Exception as e:
        print(f"❌ Chyba při načítání modelů: {e}")
        return

    # --- NAČTENÍ ZÁPASŮ ---
    with engine.begin() as conn:
        fixtures = pd.read_sql(text("""
            SELECT fixture_id, match_date, home_team, away_team
            FROM prepared_fixtures
            WHERE match_date >= CURRENT_DATE
            ORDER BY match_date ASC
            LIMIT 15
        """), conn)

        if fixtures.empty:
            print("📭 Žádné nadcházející zápasy.")
            return

        rows = []
        for _, row in fixtures.iterrows():
            f = get_rolling_features(
                conn, row['home_team'], row['away_team'],
                row['match_date'], row['fixture_id'], mv_scaler
            )
            if f:
                f['home_team'] = row['home_team']
                f['away_team'] = row['away_team']
                f['match_date'] = row['match_date']
                rows.append(f)

    if not rows:
        print("❌ Žádná data pro predikci.")
        return

    df_pred = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(f"{'Zápas':<40} | {'Tip':<5} | {'1':<7} | {'X':<7} | {'2':<7} | {'xG':<10}")
    print("-" * 100)

    # --- PREDIKCE ---
    for _, row in df_pred.iterrows():
        try:
            X_full = pd.DataFrame([row])

            # Doplň chybějící features nulou
            for col in trained_features:
                if col not in X_full.columns:
                    X_full[col] = 0.0

            X_input = X_full[trained_features]

            # A) Klasifikátor → pravděpodobnosti (Away=0, Draw=1, Home=2)
            probs_clf = best_clf.predict_proba(X_input)[0]
            p_clf_away, p_clf_draw, p_clf_home = probs_clf[0], probs_clf[1], probs_clf[2]

            # B) xG s safety clampem 0.1–8.0
            # Bez clampu by model mohl extrapolovat na nesmyslné hodnoty
            XG_MIN, XG_MAX = 0.1, 8.0
            gh = np.clip((reg_h.predict(X_input)[0] + xgb_reg_h.predict(X_input)[0]) / 2, XG_MIN, XG_MAX)
            ga = np.clip((reg_a.predict(X_input)[0] + xgb_reg_a.predict(X_input)[0]) / 2, XG_MIN, XG_MAX)

            # C) Poisson distribuce
            p1_poi, px_poi, p2_poi = 0.0, 0.0, 0.0
            for h in range(10):
                for a in range(10):
                    p = poisson.pmf(h, gh) * poisson.pmf(a, ga)
                    if   h > a: p1_poi += p
                    elif h == a: px_poi += p
                    else:        p2_poi += p

            # D) Blend 50/50: klasifikátor (forma/ELO) + Poisson (xG/střely)
            p1 = 0.5 * p_clf_home + 0.5 * p1_poi
            px = 0.5 * p_clf_draw + 0.5 * px_poi
            p2 = 0.5 * p_clf_away + 0.5 * p2_poi
            total = p1 + px + p2
            p1, px, p2 = p1 / total, px / total, p2 / total

            # E) Tip s Draw boost thresholdem
            pred_class = predict_with_draw_boost(np.array([[p2, px, p1]]), draw_threshold)[0]

            if pred_class == 2:    # Home win
                tip = "1" if p1 > 0.50 else ("1X" if px > p2 else "1")
            elif pred_class == 0:  # Away win
                tip = "2" if p2 > 0.50 else ("X2" if px > p1 else "2")
            else:                  # Draw
                tip = "X"

            match_name = f"{row['home_team']} vs {row['away_team']}"
            print(f"{match_name:<40} | {tip:<5} | {p1*100:>5.1f}% | {px*100:>5.1f}% | {p2*100:>5.1f}% | {gh:.2f}:{ga:.2f}")

        except Exception as e:
            print(f"❌ Chyba {row.get('home_team', '?')}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 100)


if __name__ == "__main__":
    main()
