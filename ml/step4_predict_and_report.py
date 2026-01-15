# ml/step4_predict_and_report.py
# KOMPLETNÍ PROCES: Predikce -> SHAP -> Report
# VERZE 4.0: Řazení dle síly signálu + Datum zápasu

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from scipy.stats import poisson

# Nastavení matplotlibu (bez GUI)
import matplotlib

matplotlib.use('Agg')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMG_DIR = os.path.join(DATA_DIR, "shap_images")
os.makedirs(IMG_DIR, exist_ok=True)


# --- Dixon-Coles ---
def calculate_dixon_coles_probs(avg_home_goals, avg_away_goals, rho, max_goals=10):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, avg_home_goals) * poisson.pmf(j, avg_away_goals)

    def correction(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - (lam * mu * rho)
        elif x == 0 and y == 1:
            return 1 + (lam * rho)
        elif x == 1 and y == 0:
            return 1 + (mu * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    for i in range(2):
        for j in range(2):
            tau = correction(i, j, avg_home_goals, avg_away_goals, rho)
            prob_matrix[i, j] = max(0, prob_matrix[i, j] * tau)

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
        plt.title(f"Vliv na tip: {target_name}", fontsize=10, fontweight='bold')
        plt.gca().invert_yaxis()

        filename = f"shap_{fixture_id}.png"
        filepath = os.path.join(IMG_DIR, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as e:
        return ""


def run_pipeline():
    print("🚀 Startuji Step 4 (Predikce + SHAP + Report)...")

    # 1. NAČTENÍ DAT
    query = "SELECT * FROM prepared_fixtures ORDER BY match_date ASC LIMIT 10"
    try:
        df_fixt = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Chyba DB: {e}");
        return

    if df_fixt.empty: return

    # 2. NAČTENÍ MODELŮ
    voting_path = os.path.join(DATA_DIR, "model_voting_ensemble.pkl")
    rf_path = os.path.join(DATA_DIR, "model_randomforest.pkl")
    poisson_path = os.path.join(DATA_DIR, "model_poisson.pkl")

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

    dc_probs, goals_h, goals_a = [], [], []
    if os.path.exists(poisson_path):
        artifact = joblib.load(poisson_path)
        imputer, scaler, reg_home, reg_away, poi_features, rho = artifact
        X_poi = scaler.transform(imputer.transform(df_fixt[poi_features]))
        gh_pred, ga_pred = reg_home.predict(X_poi), reg_away.predict(X_poi)
        for gh, ga in zip(gh_pred, ga_pred):
            dc_probs.append(calculate_dixon_coles_probs(gh, ga, rho))
            goals_h.append(gh);
            goals_a.append(ga)
    else:
        dc_probs = [(0, 0, 0)] * len(df_fixt)

    # 3. CLEANUP DB
    inspector = inspect(engine)
    if inspector.has_table("predictions"):
        with engine.begin() as conn:
            ids = tuple(df_fixt["fixture_id"].tolist())
            if ids:
                sql = f"DELETE FROM predictions WHERE fixture_id IN {ids}" if len(
                    ids) > 1 else f"DELETE FROM predictions WHERE fixture_id = {ids[0]}"
                conn.execute(text(sql))

    # 4. PREDIKCE LOOP
    new_rows = []
    for idx, row in df_fixt.iterrows():
        v_p, p_p = voting_probs[idx], dc_probs[idx]

        ph = (v_p[0] * 0.7) + (p_p[0] * 0.3)
        pd_prob = (v_p[1] * 0.7) + (p_p[1] * 0.3)
        pa = (v_p[2] * 0.7) + (p_p[2] * 0.3)
        total = ph + pd_prob + pa
        ph, pd_prob, pa = ph / total, pd_prob / total, pa / total

        predicted_class = np.argmax([ph, pd_prob, pa])
        predicted_winner = "1" if predicted_class == 0 else ("X" if predicted_class == 1 else "2")

        signal_note = ""
        # Logika signálů
        if ph > pa:
            if ph > 0.60:
                signal_note = "🔥 1 (Favorit)"
            elif (ph + pd_prob) > 0.78:
                signal_note = "✅ 1X (Safe)"
        elif pa > ph:
            if pa > 0.60:
                signal_note = "🔥 2 (Favorit)"
            elif (pa + pd_prob) > 0.55:
                signal_note = "✨ X2 (Value)"

        shap_img = ""
        if explainer:
            target_class = 0 if ph > pa else 2
            if predicted_winner == "X": target_class = 1
            shap_img = generate_shap_plot(explainer, X_clf[idx].reshape(1, -1), clf_features, row["fixture_id"],
                                          target_class)

        # Formátování data
        match_date_str = row["match_date"].strftime("%d.%m. %H:%M") if pd.notnull(row["match_date"]) else ""

        new_rows.append({
            "fixture_id": row["fixture_id"],
            "match_name": f"{row['home_team']} vs {row['away_team']}",
            "match_date_str": match_date_str,  # Pro HTML
            "match_date_obj": row["match_date"],  # Pro třídění
            "predicted_winner": predicted_winner,
            "signal_note": signal_note,
            "proba_home_win": ph,
            "proba_draw": pd_prob,
            "proba_away_win": pa,
            "expected_goals_home": goals_h[idx] if goals_h else 0,
            "expected_goals_away": goals_a[idx] if goals_a else 0,
            "fair_odd_home": round(1 / ph, 2) if ph > 0 else 0,
            "fair_odd_draw": round(1 / pd_prob, 2) if pd_prob > 0 else 0,
            "fair_odd_away": round(1 / pa, 2) if pa > 0 else 0,
            "shap_image": shap_img
        })

    # Uložení do DB (bez pomocných sloupců pro HTML)
    df_out = pd.DataFrame(new_rows)
    db_cols = ["fixture_id", "predicted_winner", "proba_home_win", "proba_draw", "proba_away_win",
               "expected_goals_home", "expected_goals_away", "fair_odd_home", "fair_odd_draw", "fair_odd_away"]
    df_out["model_name"] = "hybrid_shap"
    # Ukládáme jen data, která patří do DB tabulky
    df_out[db_cols + ["model_name"]].to_sql("predictions", engine, if_exists="append", index=False)
    print("✅ Data uložena do DB.")

    generate_html_report(df_out)


def generate_html_report(df):
    # Příprava dat
    df["Home %"] = (df["proba_home_win"] * 100).round(1)
    df["Draw %"] = (df["proba_draw"] * 100).round(1)
    df["Away %"] = (df["proba_away_win"] * 100).round(1)
    df["xG"] = df["expected_goals_home"].round(2).astype(str) + ":" + df["expected_goals_away"].round(2).astype(str)

    # --- ŘAZENÍ (SORTING) ---
    # 1. Priorita signálu (Favorit > Safe > Value > Nic)
    # 2. Datum zápasu
    def get_priority(signal):
        if "Favorit" in str(signal): return 1
        if "Safe" in str(signal): return 2
        if "Value" in str(signal): return 3
        return 4  # Ostatní

    df["priority"] = df["signal_note"].apply(get_priority)
    df = df.sort_values(by=["priority", "match_date_obj"])

    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Football Predictions v4.0</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f4f4f9; }
            h1 { text-align: center; color: #333; }
            .match-card {
                background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px; padding: 20px; display: flex; flex-wrap: wrap; align-items: flex-start;
                border-left: 5px solid transparent;
            }
            .card-priority-1 { border-left-color: #e74c3c; } /* Favorit */
            .card-priority-2 { border-left-color: #27ae60; } /* Safe */
            .card-priority-3 { border-left-color: #2980b9; } /* Value */
            .card-priority-4 { border-left-color: #bdc3c7; opacity: 0.9; } /* Ostatní */

            .match-info { flex: 1; min-width: 300px; padding-right: 20px; }
            .match-shap { flex: 0 0 450px; text-align: center; border-left: 1px solid #eee; padding-left: 20px; }

            .match-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px;}
            .match-title { font-size: 1.3em; font-weight: bold; color: #2c3e50; }
            .match-date { font-size: 0.9em; color: #7f8c8d; font-weight: bold; }

            .probs { display: flex; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; height: 25px; line-height: 25px;}
            .prob-box { text-align: center; color: white; font-size: 0.85em; }
            .prob-1 { background-color: #3498db; width: var(--p1); }
            .prob-x { background-color: #95a5a6; width: var(--px); }
            .prob-2 { background-color: #e74c3c; width: var(--p2); }

            .signal { font-weight: bold; display: inline-block; padding: 6px 12px; border-radius: 4px; background: #eee; margin-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
            td, th { padding: 6px; border-bottom: 1px solid #eee; text-align: left; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; border-radius: 5px; }
            .dc-row { background-color: #e8f6f3; font-weight: bold; color: #27ae60; }
        </style>
    </head>
    <body>
        <h1>⚽ Predikce & Analýza</h1>
    """

    for _, row in df.iterrows():
        p1, px, p2 = row["Home %"], row["Draw %"], row["Away %"]

        # Barvy a Signál
        signal_html = ""
        priority_class = f"card-priority-{row['priority']}"

        if row['signal_note']:
            color = "black"
            if "Favorit" in row['signal_note']: color = "#e74c3c"
            if "Safe" in row['signal_note']: color = "#27ae60"
            if "Value" in row['signal_note']: color = "#2980b9"
            signal_html = f"<span class='signal' style='color:{color}; border: 1px solid {color}'>{row['signal_note']}</span>"

        # Double Chance řádek
        dc_html_row = ""
        if "1X" in row['signal_note']:
            prob_dc = row["proba_home_win"] + row["proba_draw"]
            fair_dc = round(1 / prob_dc, 2)
            dc_html_row = f"<tr class='dc-row'><th>FÉR DC (1X):</th><td>{fair_dc}</td></tr>"
        elif "X2" in row['signal_note']:
            prob_dc = row["proba_away_win"] + row["proba_draw"]
            fair_dc = round(1 / prob_dc, 2)
            dc_html_row = f"<tr class='dc-row'><th>FÉR DC (X2):</th><td>{fair_dc}</td></tr>"

        # Obrázek SHAP
        img_tag = "<div style='color:#ccc; margin-top:50px;'>Bez grafu</div>"
        if row["shap_image"]:
            img_src = f"../data/shap_images/{row['shap_image']}"
            img_tag = f"<img src='{img_src}' alt='SHAP Analysis'><br><small>Vliv faktorů</small>"

        card = f"""
        <div class="match-card {priority_class}">
            <div class="match-info">
                <div class="match-header">
                    <div class="match-title">{row['match_name']}</div>
                    <div class="match-date">📅 {row['match_date_str']}</div>
                </div>
                {signal_html}
                <div class="probs" style="--p1: {p1}%; --px: {px}%; --p2: {p2}%;">
                    <div class="prob-box prob-1" style="width:{p1}%">1: {p1}%</div>
                    <div class="prob-box prob-x" style="width:{px}%">X: {px}%</div>
                    <div class="prob-box prob-2" style="width:{p2}%">2: {p2}%</div>
                </div>
                <table>
                    <tr><th>Tip modelu:</th><td><b>{row['predicted_winner']}</b></td></tr>
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

    print(f"📄 Report (Seřazený + Datum): {report_path}")


if __name__ == "__main__":
    run_pipeline()