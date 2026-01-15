# ml/step5_predictions_report.py
# Report predikcí – stejná struktura jako predictions_next_round.html, pro všechny modely

import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timezone

# --- načtení .env ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
HTML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "htmlFiles")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# přesné pořadí sloupců
COLS_ORDER = [
    "model","fixture_id","home_team","away_team","league","season","match_date",
    "prediction","proba_home_win","proba_draw","proba_away_win","max_val","diff",
    "interpretation","is_best_model"
]

# --- načtení predictions_next_round ---
df = pd.read_sql("SELECT * FROM predictions_next_round", engine)

# --- filtr jen na budoucí zápasy ---
df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
now_utc = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
df_future = df[df["match_date"].notna() & (df["match_date"] > now_utc)]

if df_future.empty:
    df_future = df.copy()

# --- reindex na správné sloupce ---
df_future = df_future.reindex(columns=COLS_ORDER).sort_values(
    ["model","match_date","league","home_team"]
)

# --- grafy ---
graph_links = ""
for model in sorted(df_future["model"].dropna().unique()):
    plt.figure(figsize=(6,4))
    df_future[df_future["model"] == model]["prediction"].value_counts().sort_index().plot(kind="bar")
    plt.title(f"Rozložení predikcí – {model}")
    plt.xlabel("Výsledek (0=Home win, 1=Draw, 2=Away win)")
    plt.ylabel("Počet zápasů")
    plt.tight_layout()
    fname = f"report_{model}.png"
    out_path = os.path.join(DATA_DIR, fname)
    plt.savefig(out_path)
    plt.close()
    graph_links += f"<h3>Model {model}</h3><img src='../data/{fname}' alt='Graf {model}'><br>"

# --- HTML report ---
table_html = df_future.to_html(index=False)

html_content = f"""
<html>
<head>
  <meta charset="utf-8">
  <title>Predictions Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background-color: #f2f2f2; }}
    h1 {{ color: #333; }}
    h3 {{ margin-top: 30px; }}
  </style>
</head>
<body>
<h1>Predictions Report</h1>
{table_html}
{graph_links}
</body>
</html>
"""

out_path = os.path.join(HTML_DIR, "predictions_report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"📄 HTML report uložen: {out_path}")
