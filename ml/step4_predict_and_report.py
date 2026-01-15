# ml/step4_predict_and_report.py
# KOMPLETNÍ PROCES: Predikce -> Uložení do DB -> Generování HTML
# Vylepšení: Inteligentní volba mezi 1/1X a 2/X2, Legenda, Názvy týmů

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from scipy.stats import poisson

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# --- Dixon-Coles (stejné jako dříve) ---
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


def run_pipeline():
    print("🚀 Startuji Step 4+5: Predikce a Report...")

    # 1. NAČTENÍ DAT (Zápasy)
    query = "SELECT * FROM prepared_fixtures ORDER BY match_date ASC LIMIT 10"
    try:
        df_fixt = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Chyba: {e}")
        return

    if df_fixt.empty:
        print("⚠️ Žádné zápasy k predikci.")
        return

    print(f"🔎 Zpracovávám {len(df_fixt)} zápasů...")

    # 2. NAČTENÍ MODELŮ
    voting_path = os.path.join(DATA_DIR, "model_voting_ensemble.pkl")
    poisson_path = os.path.join(DATA_DIR, "model_poisson.pkl")

    # Voting Init
    voting_model = None
    clf_features = []
    v_imputer, v_scaler = None, None
    voting_probs = []

    if os.path.exists(voting_path):
        artifact = joblib.load(voting_path)
        v_imputer, v_scaler = artifact[0], artifact[1]
        # Smart Unpacking
        remaining = artifact[2:]
        for item in remaining:
            if hasattr(item, "predict"):
                voting_model = item
            elif isinstance(item, (list, np.ndarray, pd.Index)):
                clf_features = item

        if voting_model is None:  # Fallback
            if hasattr(artifact[2], "predict"):
                voting_model, clf_features = artifact[2], artifact[3]
            else:
                voting_model, clf_features = artifact[3], artifact[2]

        X_clf_raw = df_fixt[clf_features].replace([np.inf, -np.inf], np.nan)
        X_clf_imp = v_imputer.transform(X_clf_raw)
        X_clf_scaled = v_scaler.transform(X_clf_imp)
        voting_probs = voting_model.predict_proba(X_clf_scaled)
    else:
        print("⚠️ Voting model chybí!")
        return

    # Poisson Init
    dc_probs = []
    goals_h, goals_a = [], []
    if os.path.exists(poisson_path):
        artifact = joblib.load(poisson_path)
        if len(artifact) == 6:
            imputer, scaler, reg_home, reg_away, poi_features, rho = artifact
        else:
            imputer, scaler, reg_home, reg_away, poi_features = artifact; rho = 0

        X_poi_imp = imputer.transform(df_fixt[poi_features])
        X_poi_scaled = scaler.transform(X_poi_imp)
        goals_h = reg_home.predict(X_poi_scaled)
        goals_a = reg_away.predict(X_poi_scaled)

        for gh, ga in zip(goals_h, goals_a):
            dc_probs.append(calculate_dixon_coles_probs(gh, ga, rho))
    else:
        dc_probs = [(0, 0, 0)] * len(df_fixt)
        goals_h = [0] * len(df_fixt)
        goals_a = [0] * len(df_fixt)

    # 3. MAZÁNÍ STARÝCH PREDIKCÍ
    inspector = inspect(engine)
    if inspector.has_table("predictions"):
        with engine.begin() as conn:
            ids = tuple(df_fixt["fixture_id"].tolist())
            if ids:
                sql = f"DELETE FROM predictions WHERE fixture_id IN {ids}" if len(
                    ids) > 1 else f"DELETE FROM predictions WHERE fixture_id = {ids[0]}"
                conn.execute(text(sql))

    # 4. VÝPOČET PREDIKCÍ A SIGNÁLŮ
    new_rows = []
    for idx, row in df_fixt.iterrows():
        v_p = voting_probs[idx]
        p_p = dc_probs[idx]

        # Hybridní pravděpodobnost (70% Voting, 30% Poisson)
        ph = (v_p[0] * 0.7) + (p_p[0] * 0.3)
        pd_prob = (v_p[1] * 0.7) + (p_p[1] * 0.3)
        pa = (v_p[2] * 0.7) + (p_p[2] * 0.3)

        total = ph + pd_prob + pa
        ph, pd_prob, pa = ph / total, pd_prob / total, pa / total

        # --- NOVÁ LOGIKA SIGNÁLŮ (Smart Bet) ---
        predicted_winner = "X"
        signal_note = ""

        # DOMÁCÍ
        if ph > pa:
            predicted_winner = "1"
            if ph > 0.60:
                signal_note = "🔥 1 (Favorit)"  # Kurz cca < 1.66
            elif (ph + pd_prob) > 0.78:
                signal_note = "✅ 1X (Safe)"  # Kurz 1X je nízký, ale jistý

        # HOSTÉ
        elif pa > ph:
            predicted_winner = "2"
            if pa > 0.60:
                signal_note = "🔥 2 (Favorit)"
            elif (pa + pd_prob) > 0.55:
                signal_note = "✨ X2 (Value)"  # Naše X2 strategie

        # REMÍZA (Model tipuje remízu jako nejpravděpodobnější)
        else:
            predicted_winner = "X"
            if pd_prob > 0.35:
                signal_note = "⚖️ Risk Remíza"

        new_rows.append({
            "fixture_id": row["fixture_id"],
            "match_name": f"{row['home_team']} vs {row['away_team']}",  # Nový sloupec
            "model_name": "hybrid_v2",
            "predicted_winner": predicted_winner,
            "signal_note": signal_note,  # Ukládáme si poznámku
            "proba_home_win": round(ph, 4),
            "proba_draw": round(pd_prob, 4),
            "proba_away_win": round(pa, 4),
            "expected_goals_home": round(goals_h[idx], 2),
            "expected_goals_away": round(goals_a[idx], 2),
            "fair_odd_home": round(1 / ph, 2) if ph > 0 else 0,
            "fair_odd_draw": round(1 / pd_prob, 2) if pd_prob > 0 else 0,
            "fair_odd_away": round(1 / pa, 2) if pa > 0 else 0
        })

    if not new_rows: return

    # Uložení do DB
    df_out = pd.DataFrame(new_rows)
    # Odstraníme pomocné sloupce pro DB (pokud nechceme měnit schema tabulky,
    # signál a match_name tam zatím neukládáme, použijeme je jen pro report.
    # Pokud DB dovolí, uložíme vše.)
    # Pro jistotu uložíme jen standardní sloupce do DB, ale DF si necháme pro report
    db_cols = ["fixture_id", "model_name", "predicted_winner",
               "proba_home_win", "proba_draw", "proba_away_win",
               "expected_goals_home", "expected_goals_away",
               "fair_odd_home", "fair_odd_draw", "fair_odd_away"]

    df_out[db_cols].to_sql("predictions", engine, if_exists="append", index=False)
    print("✅ Data uložena do DB.")

    # 5. GENEROWÁNÍ HTML REPORTU
    generate_html_report(df_out)


def generate_html_report(df):
    # Příprava dat pro zobrazení
    df["Home %"] = (df["proba_home_win"] * 100).round(1)
    df["Draw %"] = (df["proba_draw"] * 100).round(1)
    df["Away %"] = (df["proba_away_win"] * 100).round(1)
    df["xG"] = df["expected_goals_home"].astype(str) + ":" + df["expected_goals_away"].astype(str)

    # Výběr sloupců
    display_cols = [
        "match_name", "signal_note",
        "Home %", "Draw %", "Away %",
        "fair_odd_home", "fair_odd_draw", "fair_odd_away", "xG"
    ]

    report_df = df[display_cols].copy()
    report_df.rename(columns={
        "match_name": "Zápas",
        "signal_note": "DOPORUČENÍ",
        "fair_odd_home": "Fair 1",
        "fair_odd_draw": "Fair 0",
        "fair_odd_away": "Fair 2"
    }, inplace=True)

    # HTML
    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Football Predictions</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
            h1 { text-align: center; color: #333; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            th { background: #2c3e50; color: white; padding: 12px; text-align: center; }
            td { border-bottom: 1px solid #ddd; padding: 10px; text-align: center; color: #333; }
            tr:hover { background: #f1f1f1; }

            /* Barvy pro signály */
            td:nth-child(2) { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>⚽ Predikce na další kolo</h1>
        """

    # Převod tabulky
    table_html = report_df.to_html(index=False, classes="table", border=0)

    # Podmíněné formátování (jednoduchý replace v HTML stringu)
    table_html = table_html.replace("🔥", "<span style='color:red'>🔥")
    table_html = table_html.replace("✅", "<span style='color:green'>✅")
    table_html = table_html.replace("✨", "<span style='color:blue'>✨")
    table_html = table_html.replace("</span>", "</span>")  # uzavření tagů

    html += table_html

    # LEGENDA
    html += """
        <div style="margin-top: 30px; background: white; padding: 15px; border-radius: 5px;">
            <h3>ℹ️ Legenda a Vysvětlivky</h3>
            <ul>
                <li><b>Fair 1 / 0 / 2:</b> Tzv. "Fér Kurz". Je to převrácená hodnota pravděpodobnosti (1 / %). 
                    <br><i>Příklad: Pokud je Fair 1 = 1.50 a sázkovka nabízí 1.70, je to výhodná sázka (Value Bet). Pokud nabízí 1.30, nebrat.</i>
                </li>
                <li><b>🔥 1 (Favorit):</b> Model věří domácím na více než 60 %. Doporučena čistá výhra (1).</li>
                <li><b>✅ 1X (Safe):</b> Domácí nejsou tak silní, ale prohra je nepravděpodobná (Součet 1+X > 80 %).</li>
                <li><b>🔥 2 (Favorit):</b> Model věří hostům na více než 60 %. Doporučena čistá výhra (2).</li>
                <li><b>✨ X2 (Value):</b> Naše speciální strategie. Hosté jsou podceňovaní, ale mají šanci neprohrát > 55 %.</li>
                <li><b>xG:</b> Očekávaný výsledek na góly (např. 1.45:0.90).</li>
            </ul>
        </div>
    </body>
    </html>
    """

    report_path = os.path.join(DATA_DIR, "predictions_report_final.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📄 Report vygenerován: {report_path}")


if __name__ == "__main__":
    run_pipeline()