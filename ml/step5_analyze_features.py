"""
step5_analyze_features.py  v2
==============================
Aktualizováno pro pipeline v4 (step3 v4):

ZMĚNY oproti v1:
  - Opraveny názvy modelů:
      xgb_match_winner.pkl   → xgb_classifier.pkl
      match_winner_model.pkl → voting_classifier.pkl
      home_goals_model.pkl   → poisson_home_goals.pkl
      away_goals_model.pkl   → poisson_away_goals.pkl
      xgb_features.pkl       → feature_cols.pkl  (sdílené pro oba modely)
      model_features.pkl     → feature_cols.pkl
  - Oba klasifikátory sdílí stejný feature_cols.pkl (20 features)
  - SHAP správně extrahuje preprocessor z Pipeline (Imputer, pak XGB)
  - Načítá draw_threshold.pkl a market_value_scaler.pkl
  - Doplňuje chybějící features z DB (home_x_elo, elo_x_market)
  - Nový souhrnný výstup: tabulka všech zápasů + komentáře ke shodě modelů
  - Globální SHAP summary plot (importance přes všechny zápasy)
"""

import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy.stats import poisson

matplotlib.use('Agg')

# =============================================================================
# 1. KONFIGURACE
# =============================================================================

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SHAP_DIR  = os.path.join(MODEL_DIR, "shap_reports")
os.makedirs(SHAP_DIR, exist_ok=True)

# Tržní hodnoty (záloha pokud market_value_diff_scaled chybí v DB)
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
# 2. SHAP ANALÝZA — JEDEN ZÁPAS
# =============================================================================

def generate_shap_plot(fixture_id, match_name, xgb_pipeline, X_input, feature_names):
    """
    Vytvoří SHAP bar plot pro jeden zápas.
    Správně extrahuje preprocessor a XGB model z Pipeline.
    Vrátí (shap_values_home_win, filename).
    """
    try:
        # Pipeline struktura: [('prep', Imputer), ('xgb', XGBClassifier)]
        prep_steps = xgb_pipeline[:-1]           # všechny kroky kromě posledního
        X_transformed = prep_steps.transform(X_input)

        xgb_model = xgb_pipeline.named_steps['xgb']

        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        # shap_values je list [Away, Draw, Home] → index 2 = výhra domácích
        # Kompatibilita se starým i novým SHAP API:
        # SHAP < 0.40: list[n_classes] kde každý prvek je (n_samples, n_features)
        #              → shap_values[2][0]  (class=Home, sample=0)
        # SHAP ≥ 0.40: ndarray (n_samples, n_features, n_classes)
        #              → shap_values[0][:, 2]  (sample=0, class=Home)
        if isinstance(shap_values, list):
            # Starý formát: list of arrays per class
            vals = shap_values[2][0]
        else:
            # Nový formát: ndarray (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                vals = shap_values[0, :, 2]   # sample 0, všechny features, class=Home (idx 2)
            else:
                # Fallback: (n_features,) nebo (1, n_features) pro binary
                vals = shap_values.flatten()

        df_shap = pd.DataFrame({"feat": feature_names, "val": vals})
        df_shap["abs_val"] = df_shap["val"].abs()
        df_shap = df_shap.sort_values("abs_val", ascending=False).head(12).sort_values("val")

        colors = ['#e74c3c' if x < 0 else '#27ae60' for x in df_shap['val']]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_shap['feat'], df_shap['val'], color=colors, edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_title(f"SHAP – Vliv faktorů na výhru domácích\n{match_name}", fontsize=12, pad=12)
        ax.set_xlabel("SHAP hodnota (vliv na P(výhra domácích))")

        # Popisky hodnot
        for i, (val, feat) in enumerate(zip(df_shap['val'], df_shap['feat'])):
            ax.text(val + (0.001 if val >= 0 else -0.001), i,
                    f" {val:+.3f}", va='center',
                    ha='left' if val >= 0 else 'right', fontsize=8)

        plt.tight_layout()

        safe_id = str(fixture_id).replace('/', '_')
        file_path = os.path.join(SHAP_DIR, f"shap_{safe_id}.png")
        plt.savefig(file_path, dpi=120)
        plt.close()

        return vals, os.path.basename(file_path)

    except Exception as e:
        print(f"  ⚠️  SHAP chyba: {e}")
        return None, None


# =============================================================================
# 3. GLOBÁLNÍ SHAP SUMMARY
# =============================================================================

def generate_shap_summary(all_shap_vals, feature_names):
    """Průměrný |SHAP| přes všechny zápasy → globalní importance."""
    if not all_shap_vals:
        return
    try:
        matrix = np.vstack(all_shap_vals)
        mean_abs = np.abs(matrix).mean(axis=0)
        df = pd.DataFrame({"feat": feature_names, "importance": mean_abs})
        df = df.sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df['feat'], df['importance'], color='#3498db', edgecolor='white')
        ax.set_title("SHAP – Globální importance features\n(průměr |SHAP| přes všechny zápasy)", fontsize=12, pad=12)
        ax.set_xlabel("Průměrná |SHAP| hodnota")
        plt.tight_layout()

        path = os.path.join(SHAP_DIR, "shap_global_summary.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"\n  💾 Globální SHAP summary: {path}")
    except Exception as e:
        print(f"  ⚠️  Global SHAP chyba: {e}")


# =============================================================================
# 4. DRAW BOOST (shodné se step4)
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
# 5. PŘÍPRAVA VSTUPNÍHO VEKTORU Z DB ŘÁDKU
# =============================================================================

def prepare_features(row, feature_cols, mv_scaler):
    """
    Z řádku prepared_fixtures doplní případně chybějící odvozené features.
    Vrátí DataFrame (1 řádek) s přesně feature_cols sloupci.
    """
    X = pd.DataFrame([row])

    # Doplň odvozené features pokud nejsou v DB (step2 je ukládá, ale záloha)
    # OPRAVA: home_x_elo = home_advantage * elo_diff = elo_diff (step2 definice)
    # Původní fallback (h_elo * pts/3.0) byl špatná sémantika i rozsah
    if 'home_x_elo' not in X.columns or pd.isna(X['home_x_elo'].iloc[0]):
        elo_diff_val = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
        X['home_x_elo'] = elo_diff_val  # home_advantage=1.0, takže = elo_diff

    if 'market_value_diff_scaled' not in X.columns or pd.isna(X['market_value_diff_scaled'].iloc[0]):
        mv_h = MARKET_VALUES.get(row.get('home_team', ''), 200.0)
        mv_a = MARKET_VALUES.get(row.get('away_team', ''), 200.0)
        mv_diff = mv_h - mv_a
        if mv_scaler is not None:
            mv_df = pd.DataFrame([[mv_diff]], columns=['market_value_diff'])
            X['market_value_diff_scaled'] = float(mv_scaler.transform(mv_df)[0][0])
        else:
            X['market_value_diff_scaled'] = mv_diff / 400.0

    if 'elo_x_market' not in X.columns or pd.isna(X['elo_x_market'].iloc[0]):
        elo_diff = float(X.get('elo_diff', pd.Series([0.0])).iloc[0] or 0.0)
        X['elo_x_market'] = elo_diff * float(X['market_value_diff_scaled'].iloc[0])

    # Doplň chybějící features nulou (pojistka)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    return X[feature_cols].astype(float)


# =============================================================================
# 6. HLAVNÍ FUNKCE
# =============================================================================

def run_analysis():
    print("=" * 70)
    print("🔬 STEP5 v2: ANALÝZA PREDIKCÍ + SHAP VYSVĚTLENÍ")
    print("=" * 70)

    # --- NAČTENÍ MODELŮ ---
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
        draw_threshold = joblib.load(thr_path) if os.path.exists(thr_path) else 0.22

        mv_path   = os.path.join(MODEL_DIR, "market_value_scaler.pkl")
        mv_scaler = joblib.load(mv_path) if os.path.exists(mv_path) else None

        print(f"  ✅ Všechny modely načteny")
        print(f"  📌 Features: {len(feature_cols)}  |  Draw threshold: {draw_threshold:.2f}")
        print(f"  📌 MV scaler: {'✅' if mv_scaler else '⚠️  chybí'}")

    except FileNotFoundError as e:
        print(f"  ❌ Chybí model: {e}")
        print(f"     Ujisti se, že step3 proběhl úspěšně.")
        return

    # --- NAČTENÍ ZÁPASŮ ---
    df_pred = pd.read_sql(
        text("""SELECT * FROM prepared_fixtures
                WHERE match_date >= CURRENT_DATE
                  AND match_date <= CURRENT_DATE + INTERVAL '14 days'
                ORDER BY match_date ASC, league ASC"""),
        engine
    )
    print(f"\n📅 Načteno {len(df_pred)} nadcházejících zápasů")
    print(f"   Sloupce v DB: {len(df_pred.columns)} total")

    missing_feats = [f for f in feature_cols if f not in df_pred.columns]
    if missing_feats:
        print(f"   ⚠️  Chybí v DB (budou dopočítány): {missing_feats}")

    if df_pred.empty:
        print("📭 Žádné zápasy.")
        return

    # --- SHAP GLOBÁLNÍ SBÍRKA ---
    all_shap_vals = []

    # --- TABULKA VÝSLEDKŮ ---
    print("\n" + "=" * 110)
    print(f"{'Zápas':<40} | {'Voting':^20} | {'XGBoost':^20} | {'Shoda':^6} | {'xG':^9} | {'Tip'}")
    print(f"{'':40} | {'1    X    2':^20} | {'1    X    2':^20} | {'':6} | {'H  : A':^9} |")
    print("-" * 110)

    results = []

    for _, row in df_pred.iterrows():
        home = row.get('home_team', '?')
        away = row.get('away_team', '?')
        league = row.get('league', '?')
        match_name = f"[{league}] {home} vs {away}"
        fixture_id = row.get('fixture_id', f"{home}_{row.get('match_date', '')}")

        try:
            X = prepare_features(row, feature_cols, mv_scaler)

            # A) Voting pravděpodobnosti
            probs_v   = voting_clf.predict_proba(X)[0]
            pv_a, pv_x, pv_h = probs_v[0], probs_v[1], probs_v[2]

            # B) XGBoost pravděpodobnosti
            probs_xgb = xgb_clf.predict_proba(X)[0]
            px_a, px_x, px_h = probs_xgb[0], probs_xgb[1], probs_xgb[2]

            # C) xG (Poisson + XGBoost hybrid, clamp 0.1–8.0)
            gh = np.clip((poisson_h.predict(X)[0] + xgb_reg_h.predict(X)[0]) / 2, 0.1, 8.0)
            ga = np.clip((poisson_a.predict(X)[0] + xgb_reg_a.predict(X)[0]) / 2, 0.1, 8.0)

            # D) Poisson distribuce
            p1_poi, px_poi, p2_poi = 0.0, 0.0, 0.0
            for h in range(10):
                for a in range(10):
                    p = poisson.pmf(h, gh) * poisson.pmf(a, ga)
                    if   h > a: p1_poi += p
                    elif h == a: px_poi += p
                    else:        p2_poi += p

            # E) Blended (50% Voting + 50% Poisson) → finální tip
            p1 = 0.5 * pv_h + 0.5 * p1_poi
            px_b = 0.5 * pv_x + 0.5 * px_poi
            p2 = 0.5 * pv_a + 0.5 * p2_poi
            total = p1 + px_b + p2
            p1, px_b, p2 = p1 / total, px_b / total, p2 / total

            pred_class = predict_with_draw_boost(np.array([[p2, px_b, p1]]), draw_threshold)[0]
            if pred_class == 2:   tip = "1" if p1 > 0.50 else ("1X" if px_b > p2 else "1")
            elif pred_class == 0: tip = "2" if p2 > 0.50 else ("X2" if px_b > p1 else "2")
            else:                 tip = "X"

            # F) Shoda modelů
            v_pred   = np.argmax([pv_a, pv_x, pv_h])
            xgb_pred = np.argmax([px_a, px_x, px_h])
            shoda = "✅" if v_pred == xgb_pred else "❌"

            # G) Rozdíl v pravděpodobnostech (míra nejistoty)
            diff_h = abs(pv_h - px_h)
            diff_x = abs(pv_x - px_x)
            diff_a = abs(pv_a - px_a)
            max_diff = max(diff_h, diff_x, diff_a)

            # H) Confidence: P(vítěz) / baseline(0.333) — kolikrát silnější než náhoda
            max_blended = max(p1, px_b, p2)
            confidence = max_blended / (1/3)

            # Tisk řádku
            print(f"{match_name:<45} | {pv_h*100:>4.0f}% {pv_x*100:>4.0f}% {pv_a*100:>4.0f}% |"
                  f" {px_h*100:>4.0f}% {px_x*100:>4.0f}% {px_a*100:>4.0f}% |"
                  f"  {shoda}   | {gh:.2f}:{ga:.2f} | {tip:<4} {confidence:.2f}x")

            # H) SHAP analýza (jen XGBoost — má TreeExplainer)
            shap_vals, shap_file = generate_shap_plot(fixture_id, match_name, xgb_clf, X, feature_cols)
            if shap_vals is not None:
                all_shap_vals.append(shap_vals)

            results.append({
                'match': match_name, 'tip': tip, 'shoda': shoda,
                'p1': p1, 'px': px_b, 'p2': p2,
                'gh': gh, 'ga': ga, 'max_diff': max_diff,
                'confidence': confidence, 'shap_file': shap_file
            })

        except Exception as e:
            print(f"❌ Chyba: {match_name}: {e}")
            import traceback; traceback.print_exc()

    print("=" * 110)

    # --- SOUHRN ---
    print("\n📊 SOUHRN ANALÝZY")
    print("─" * 60)

    if results:
        shody   = sum(1 for r in results if r['shoda'] == "✅")
        neshody = len(results) - shody
        print(f"  Shoda modelů:   {shody}/{len(results)} zápasů  ({shody/len(results)*100:.0f}%)")
        print(f"  Neshoda modelů: {neshody} zápasů — vyšší nejistota predikce")

        # Nejistější predikce
        uncertain = sorted(results, key=lambda x: x['max_diff'], reverse=True)[:3]
        print(f"\n  ⚠️  Nejistější predikce (největší rozdíl Voting vs XGBoost):")
        for r in uncertain:
            print(f"     {r['match']:<45} diff={r['max_diff']:.1%}  tip={r['tip']}")

        # Nejjistější predikce (confidence ranking)
        confident = sorted(results, key=lambda x: x['confidence'], reverse=True)[:5]
        print(f"\n  🎯 Nejsilnější tipy (confidence nad baseline 1.0x):")
        for r in confident:
            tip_label = r['tip']
            print(f"     {r['match']:<45} tip={tip_label:<4}  conf={r['confidence']:.2f}x  "
                  f"p=({r['p1']*100:.0f}%/{r['px']*100:.0f}%/{r['p2']*100:.0f}%)")

        # SHAP soubory
        shap_files = [r['shap_file'] for r in results if r['shap_file']]
        if shap_files:
            print(f"\n  💡 SHAP grafy uloženy ({len(shap_files)} souborů):")
            for f in shap_files:
                print(f"     {os.path.join(SHAP_DIR, f)}")

    # --- VÝSLEDKY TRÉNINKU Z training_log.json ---
    print("\n" + "=" * 70)
    print("📈 OUT-OF-SAMPLE VÝKON — TimeSeriesSplit z posledního tréninku")
    print("=" * 70)
    print("  ℹ️  Tato čísla jsou skutečně out-of-sample (step3 TimeSeriesSplit).")
    print("     Evaluace na prepared_datasets = data leakage → viz poznámka níže.\n")
    try:
        import json
        log_path = os.path.join(MODEL_DIR, "training_log.json")
        if not os.path.exists(log_path):
            print(f"  ⚠️  training_log.json nenalezen: {log_path}")
        else:
            with open(log_path, "r", encoding="utf-8") as f:
                all_runs = json.load(f)

            # Vezmi poslední run
            run = all_runs[-1] if isinstance(all_runs, list) else all_runs
            ts = run.get("timestamp", "?")
            r  = run.get("results", run)  # fallback: results může být přímo na top-level

            print(f"  Poslední trénink: {ts}")
            print(f"  Dataset:         {r.get('n_samples','?')} zápasů  "
                  f"|  {r.get('n_features_all','?')} → {r.get('n_features_sel','?')} features")
            print(f"  Nejlepší model:  {r.get('best_classifier','?')}")
            print(f"  Draw threshold:  {r.get('draw_threshold','?')}")

            # Voting
            v = r.get("voting", {})
            x = r.get("xgboost", {})
            print(f"\n  {'Model':<22} {'Accuracy':>10} {'F1-macro':>10} {'Draw thr':>10}")
            print(f"  {'─'*55}")
            if v:
                print(f"  {'Voting (RF+GBM+LR)':<22} {v.get('accuracy','?'):>10}  "
                      f"{v.get('f1','?'):>9}  {v.get('draw_thr','?'):>9}")
            if x:
                print(f"  {'XGBoost':<22} {x.get('accuracy','?'):>10}  "
                      f"{x.get('f1','?'):>9}  {x.get('draw_thr','?'):>9}")

            # Fold detaily (pokud jsou uloženy)
            v_folds = v.get("folds", [])
            if v_folds:
                print(f"\n  Voting — fold detail:")
                print(f"  {'Fold':<6} {'Train':>6} {'Test':>6} {'Acc':>7} {'F1':>7}")
                print(f"  {'─'*38}")
                for fd in v_folds:
                    marker = " ← nejnovější" if fd['fold'] == len(v_folds) else ""
                    print(f"  {fd['fold']:<6} {fd['train']:>6} {fd['test']:>6} "
                          f"{fd['accuracy']:>6.3f}  {fd['f1']:>6.3f}{marker}")
            else:
                print(f"\n  ℹ️  Fold detail není v logu. Pro zobrazení foldů")
                print(f"     aktualizuj step3 aby ukládal 'folds' do results_log.")

            # Regressory
            reg = r.get("regressors", {})
            if reg:
                avg_poi = (reg.get('poisson_h', 0) + reg.get('poisson_a', 0)) / 2
                avg_xgb = (reg.get('xgb_h', 0) + reg.get('xgb_a', 0)) / 2
                print(f"\n  ⚽ xG regressory (MAE):")
                print(f"     Poisson:  {avg_poi:.3f} gólů/zápas")
                print(f"     XGBoost:  {avg_xgb:.3f} gólů/zápas")

            # Vybrané features
            selected = r.get("selected", [])
            if selected:
                print(f"\n  📌 Vybrané features ({len(selected)}):")
                neutral = [f for f in selected if not f.startswith(('home_avg_', 'away_avg_'))]
                pairs_h = [f for f in selected if f.startswith('home_avg_')]
                print(f"     Neutral ({len(neutral)}): {', '.join(neutral)}")
                print(f"     Párové features: {len(pairs_h)} párů × 2 = {len(pairs_h)*2}")

            print(f"\n  ⚠️  POZNÁMKA: Accuracy/F1 jsou průměry přes 5 TimeSeriesSplit foldů.")
            print(f"     Fold 5 (nejnovější data, největší train) je nejlepší proxy")
            print(f"     pro výkon na budoucích zápasech. Evaluace na celém")
            print(f"     prepared_datasets = data leakage (dává ~87% → zavádějící).")

    except Exception as e:
        print(f"  ⚠️  Backtesting selhal: {e}")
        import traceback; traceback.print_exc()

    # --- GLOBÁLNÍ SHAP SUMMARY ---
    if all_shap_vals:
        print("\n  📈 Generuji globální SHAP summary...")
        generate_shap_summary(all_shap_vals, feature_cols)

    print("\n✅ Analýza dokončena.")


if __name__ == "__main__":
    run_analysis()