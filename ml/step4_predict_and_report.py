# ml/step4_predict_and_report.py
# KOMPLETNÍ PROCES: Predikce -> SHAP -> Report
# VERZE: SNIPER MODE (Exact Match) 🎯
# Logika: 1:1 s backtestem (6% MAX / 2% STD)
# Model: Hybrid (70% RF + 30% Poisson)

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from scipy.stats import poisson

import matplotlib

matplotlib.use('Agg')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMG_DIR = os.path.join(DATA_DIR, "shap_images")
os.makedirs(IMG_DIR, exist_ok=True)


# --- Dixon-Coles (Hybridní část) ---
def calculate_dixon_coles_probs(avg_home_goals, avg_away_goals, rho=0, max_goals=10):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, avg_home_goals) * poisson.pmf(j, avg_away_goals)
    prob_matrix /= prob_matrix.sum()
    return np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)), np.sum(np.triu(prob_matrix, 1))


# --- SHAP Plot ---
def generate_shap_plot(explainer, X_row, feature_names, fixture_id, predicted_class):
    try:
        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            vals = shap_values[predicted_class][0]
        else:
            if len(shap_values.shape) == 2:
                vals = shap_values[0]
            else:
                vals = shap_values[..., predicted_class][0]

        df_shap = pd.DataFrame({'feature': feature_names, 'value': vals})
        df_shap['abs_value'] = df_shap['value'].abs()
        df_shap = df_shap.sort_values('abs_value', ascending=False).head(6)

        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in df_shap['value']]

        plt.figure(figsize=(5, 3.5))
        plt.barh(df_shap['feature'], df_shap['value'], color=colors)
        plt.axvline(0, color='black', linewidth=0.8)

        target_name = ['Domácí', 'Remízu', 'Hosty'][predicted_class]
        plt.title(f"Vliv na: {target_name}", fontsize=10, fontweight='bold')
        plt.gca().invert_yaxis()

        filename = f"shap_{fixture_id}.png"
        filepath = os.path.join(IMG_DIR, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    except Exception:
        return ""


def run_pipeline():
    print("🚀 Startuji Step 4 (SNIPER MODE - Exact Match)...")

    # 1. NAČTENÍ DAT
    query = """
    SELECT * FROM prepared_fixtures 
    WHERE match_date IS NOT NULL 
    ORDER BY match_date ASC
    """
    try:
        df_fixt = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Chyba DB: {e}");
        return

    if df_fixt.empty:
        print("⚠️ Žádné zápasy s nastaveným datem.");
        return

    # 2. NAČTENÍ MODELŮ
    voting_path = os.path.join(DATA_DIR, "model_voting_ensemble.pkl")
    rf_path = os.path.join(DATA_DIR, "model_randomforest.pkl")

    if os.path.exists(voting_path):
        artifact = joblib.load(voting_path)
        v_imputer, v_scaler, voting_model, clf_features = artifact
        X_clf = v_scaler.transform(v_imputer.transform(df_fixt[clf_features].replace([np.inf, -np.inf], np.nan)))
        voting_probs = voting_model.predict_proba(X_clf)
    else:
        print("⚠️ Voting model chybí!");
        return

    explainer = None
    if os.path.exists(rf_path):
        try:
            rf_artifact = joblib.load(rf_path)
            loaded_model = rf_artifact[2]
            base_rf = loaded_model.calibrated_classifiers_[0].estimator if hasattr(loaded_model,
                                                                                   "calibrated_classifiers_") else loaded_model
            explainer = shap.TreeExplainer(base_rf)
        except:
            pass

    # 3. CLEANUP DB
    inspector = inspect(engine)
    if inspector.has_table("predictions"):
        with engine.begin() as conn:
            ids = tuple(df_fixt["fixture_id"].tolist())
            if ids:
                sql = f"DELETE FROM predictions WHERE fixture_id IN {ids}" if len(
                    ids) > 1 else f"DELETE FROM predictions WHERE fixture_id = {ids[0]}"
                conn.execute(text(sql))

    # 4. PREDIKCE LOOP (SNIPER LOGIC)
    new_rows = []

    # --- KONFIGURACE STRATEGIE (1:1 se Step 7) ---
    THRESH_FAVORIT = 0.55
    THRESH_SAFE = 0.75
    THRESH_SUPER = 0.82
    THRESH_VALUE = 0.55

    # ZOBRAZENÍ VKLADU (Opraveno na 9% a 3%)
    PCT_MAX_DISPLAY = "9%"  # Sniper MAX
    PCT_STD_DISPLAY = "3%"  # Sniper STD

    for idx, row in df_fixt.iterrows():
        # A) Hybridní Výpočet (70% Voting + 30% Poisson)
        rf_h, rf_d, rf_a = voting_probs[idx]

        avg_h = row.get("home_avg_goals", 1.5)
        avg_a = row.get("away_avg_goals", 1.2)
        if pd.isna(avg_h): avg_h = 1.5
        if pd.isna(avg_a): avg_a = 1.2

        poi_h, poi_d, poi_a = calculate_dixon_coles_probs(avg_h, avg_a)

        ph = (rf_h * 0.7) + (poi_h * 0.3)
        pd_prob = (rf_d * 0.7) + (poi_d * 0.3)
        pa = (rf_a * 0.7) + (poi_a * 0.3)

        # Normalizace
        total = ph + pd_prob + pa
        ph, pd_prob, pa = ph / total, pd_prob / total, pa / total

        # B) Logika Signálů (Sniper)
        predicted_class = np.argmax([ph, pd_prob, pa])
        predicted_winner = "1" if predicted_class == 0 else ("X" if predicted_class == 1 else "2")

        signal_note = ""
        staking_rec = ""
        priority = 4

        # DOMÁCÍ
        if ph > pa:
            if ph > THRESH_FAVORIT:
                signal_note = "🔥 Favorit (1X)"
                staking_rec = f"MAX ({PCT_MAX_DISPLAY})"
                priority = 1
            elif (ph + pd_prob) > THRESH_SAFE:
                signal_note = "✅ Safe (1X)"
                staking_rec = f"STD ({PCT_STD_DISPLAY})"
                priority = 2
                # Super Safe Boost (Step 7 logic)
                if (ph + pd_prob) > THRESH_SUPER:
                    signal_note = "💎 Safe+ (1X)"
                    staking_rec = f"MAX ({PCT_MAX_DISPLAY})"
                    priority = 1

        # HOSTÉ
        elif pa > ph:
            if pa > THRESH_FAVORIT:
                signal_note = "🔥 Favorit (X2)"
                staking_rec = f"MAX ({PCT_MAX_DISPLAY})"
                priority = 1
            elif (pa + pd_prob) > THRESH_VALUE:
                signal_note = "✨ Value (X2)"
                staking_rec = f"STD ({PCT_STD_DISPLAY})"
                priority = 3
                # Super Safe Boost
                if (pa + pd_prob) > THRESH_SUPER:
                    signal_note = "💎 Safe+ (X2)"
                    staking_rec = f"MAX ({PCT_MAX_DISPLAY})"
                    priority = 1

        shap_img = ""
        if explainer:
            target_class = 0 if ph > pa else 2
            if predicted_winner == "X": target_class = 1
            shap_img = generate_shap_plot(explainer, X_clf[idx].reshape(1, -1), clf_features, row["fixture_id"],
                                          target_class)

        match_date_str = row["match_date"].strftime("%d.%m. %H:%M") if pd.notnull(row["match_date"]) else ""

        new_rows.append({
            "fixture_id": row["fixture_id"],
            "match_name": f"{row['home_team']} vs {row['away_team']}",
            "match_date_str": match_date_str,
            "match_date_obj": row["match_date"],
            "predicted_winner": predicted_winner,
            "signal_note": signal_note,
            "staking_rec": staking_rec,
            "priority": priority,
            "proba_home_win": ph,
            "proba_draw": pd_prob,
            "proba_away_win": pa,
            "expected_goals_home": avg_h,
            "expected_goals_away": avg_a,
            "fair_odd_home": round(1 / ph, 2) if ph > 0 else 0,
            "fair_odd_draw": round(1 / pd_prob, 2) if pd_prob > 0 else 0,
            "fair_odd_away": round(1 / pa, 2) if pa > 0 else 0,
            "shap_image": shap_img
        })

    # Uložení do DB
    df_out = pd.DataFrame(new_rows)
    db_cols = ["fixture_id", "predicted_winner", "proba_home_win", "proba_draw", "proba_away_win",
               "expected_goals_home", "expected_goals_away", "fair_odd_home", "fair_odd_draw", "fair_odd_away"]
    df_out["model_name"] = "hybrid_sniper"
    df_out[db_cols + ["model_name"]].to_sql("predictions", engine, if_exists="append", index=False)
    print("✅ Data uložena do DB.")

    generate_html_report(df_out)


def generate_html_report(df):
    df["Home %"] = (df["proba_home_win"] * 100).round(1)
    df["Draw %"] = (df["proba_draw"] * 100).round(1)
    df["Away %"] = (df["proba_away_win"] * 100).round(1)
    df["xG"] = df["expected_goals_home"].round(2).astype(str) + ":" + df["expected_goals_away"].round(2).astype(str)

    # Řazení: Datum -> Priorita
    df = df.sort_values(by=["match_date_obj", "priority"])

    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Sniper Predictions v5.1</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f0f2f5; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 30px;}
            .match-card {
                background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 25px; padding: 25px; display: flex; flex-wrap: wrap; align-items: flex-start;
                border-left: 6px solid transparent; transition: transform 0.2s;
            }
            .match-card:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }

            .card-priority-1 { border-left-color: #e74c3c; background: #fffdfd; } /* MAX */
            .card-priority-2 { border-left-color: #27ae60; } /* Safe */
            .card-priority-3 { border-left-color: #2980b9; } /* Value */
            .card-priority-4 { border-left-color: #bdc3c7; opacity: 0.85; }

            .match-info { flex: 1; min-width: 320px; padding-right: 30px; }
            .match-shap { flex: 0 0 400px; text-align: center; border-left: 1px solid #eee; padding-left: 30px; }

            .match-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;}
            .match-title { font-size: 1.4em; font-weight: 700; color: #2c3e50; }
            .match-date { font-size: 0.95em; color: #7f8c8d; font-weight: 600; background: #ecf0f1; padding: 4px 10px; border-radius: 20px; }

            .tags { display: flex; gap: 10px; margin-bottom: 15px; }
            .signal { font-weight: bold; padding: 6px 12px; border-radius: 6px; font-size: 0.9em; }
            .stake-tag { background: #34495e; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 0.9em; }
            .stake-max { background: #c0392b; box-shadow: 0 2px 4px rgba(192, 57, 43, 0.3); }

            .probs { display: flex; margin: 15px 0; border-radius: 8px; overflow: hidden; height: 30px; line-height: 30px; font-size: 0.9em; font-weight: 600;}
            .prob-box { text-align: center; color: white; }
            .prob-1 { background-color: #3498db; width: var(--p1); }
            .prob-x { background-color: #95a5a6; width: var(--px); }
            .prob-2 { background-color: #e74c3c; width: var(--p2); }

            table { width: 100%; margin-top: 15px; font-size: 0.95em; border-collapse: collapse; }
            td, th { padding: 8px 0; border-bottom: 1px solid #f1f1f1; text-align: left; }
            th { color: #7f8c8d; font-weight: 600; width: 40%; }
            .dc-row { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🎯 Sniper Predictions</h1>
    """

    current_date = None

    for _, row in df.iterrows():
        p1, px, p2 = row["Home %"], row["Draw %"], row["Away %"]

        date_only = row["match_date_str"].split(" ")[0]
        if date_only != current_date:
            html += f"<h2 style='color:#7f8c8d; margin: 40px 0 20px 0; border-bottom:2px solid #e0e0e0; padding-bottom:5px;'>📅 {date_only}</h2>"
            current_date = date_only

        signal_html = ""
        stake_html = ""
        priority_class = f"card-priority-{row['priority']}"

        if row['signal_note']:
            color = "#333";
            bg = "#eee"
            if "Favorit" in row['signal_note']: color = "#c0392b"; bg = "#fadbd8"
            if "Safe" in row['signal_note']: color = "#27ae60"; bg = "#d5f5e3"
            if "Value" in row['signal_note']: color = "#2980b9"; bg = "#d6eaf8"
            if "💎" in row['signal_note']: color = "#8e44ad"; bg = "#f5eef8"  # Safe+

            signal_html = f"<span class='signal' style='color:{color}; background:{bg}'>{row['signal_note']}</span>"
            stake_cls = "stake-max" if "MAX" in row['staking_rec'] else "stake-std"
            stake_html = f"<span class='stake-tag {stake_cls}'>💰 Vklad: {row['staking_rec']}</span>"

        dc_html_row = ""
        if "1X" in str(row['signal_note']):
            prob_dc = row["proba_home_win"] + row["proba_draw"]
            fair_dc = round(1 / prob_dc, 2)
            dc_html_row = f"<tr class='dc-row'><th>FÉR DC (1X):</th><td>{fair_dc}</td></tr>"
        elif "X2" in str(row['signal_note']):
            prob_dc = row["proba_away_win"] + row["proba_draw"]
            fair_dc = round(1 / prob_dc, 2)
            dc_html_row = f"<tr class='dc-row'><th>FÉR DC (X2):</th><td>{fair_dc}</td></tr>"

        img_tag = "<div style='color:#ccc; margin-top:50px;'>Bez grafu</div>"
        if row["shap_image"]:
            img_src = f"../data/shap_images/{row['shap_image']}"
            img_tag = f"<img src='{img_src}' alt='SHAP Analysis'><br><small style='color:#999'>Vliv faktorů</small>"

        time_only = row["match_date_str"].split(" ")[1]

        card = f"""
        <div class="match-card {priority_class}">
            <div class="match-info">
                <div class="match-header">
                    <div class="match-title">{row['match_name']}</div>
                    <div class="match-date">⏰ {time_only}</div>
                </div>
                <div class="tags">
                    {signal_html}
                    {stake_html}
                </div>
                <div class="probs" style="--p1: {p1}%; --px: {px}%; --p2: {p2}%;">
                    <div class="prob-box prob-1" style="width:{p1}%">1: {p1}%</div>
                    <div class="prob-box prob-x" style="width:{px}%">X: {px}%</div>
                    <div class="prob-box prob-2" style="width:{p2}%">2: {p2}%</div>
                </div>
                <table>
                    <tr><th>Očekávané skóre (xG):</th><td>{row['xG']}</td></tr>
                    <tr><th>Fér kurzy (1 | X | 2):</th><td>{row['fair_odd_home']} | {row['fair_odd_draw']} | {row['fair_odd_away']}</td></tr>
                    {dc_html_row}
                </table>
            </div>
            <div class="match-shap">
                {img_tag}
            </div>
        </div>
        """
        html += card

    html += "</body></html>"

    report_path = os.path.join(DATA_DIR, "predictions_report_final.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📄 Sniper Report (Exact): {report_path}")


if __name__ == "__main__":
    run_pipeline()