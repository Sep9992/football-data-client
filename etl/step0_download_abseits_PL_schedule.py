# etl/step0_download_abseits_PL_schedule.py
from bs4 import BeautifulSoup
import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import datetime

# --- načtení .env ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

base_dir = os.path.dirname(os.path.dirname(__file__))
html_file = os.path.join(base_dir, "htmlFiles", "premier_league_2025-2026_abseits_at.html")

with open(html_file, encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

results = []
for a in soup.find_all("a", attrs={"aria-label": True, "href": True}):
    label = a["aria-label"]
    href = a["href"]

    # najdi datum v nejbližším předchozím <span class="match__date-formatted">
    date_tag = a.find_previous("span", class_="match__date-formatted")
    date = date_tag.get_text(strip=True) if date_tag else None

    # rozděl domácí/hosty
    if " - " in label:
        home, away = [t.strip() for t in label.split(" - ", 1)]
    else:
        home, away = label, None

    results.append({
        "date": date,
        "home_team": home,
        "away_team": away,
        "url": href
    })

df = pd.DataFrame(results)

# --- převod na datetime a odstranění neplatných řádků ---
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])

print(f"📊 Načteno {len(df)} zápasů z HTML")

# --- uložit do DB ---
engine = create_engine(DATABASE_URL)

fixtures_df = pd.DataFrame({
    "league": "PL",
    "season": "2025-26",
    "match_date": df["date"].dt.date,  # uložíme jen DATE
    "home_team": df["home_team"],
    "away_team": df["away_team"],
    "url": df["url"],
    "target": None,  # zatím neodehráno
    "created_at": datetime.datetime.now()
})

with engine.begin() as conn:
    for _, row in fixtures_df.iterrows():
        conn.execute(text("""
            INSERT INTO fixtures (league, season, match_date, home_team, away_team, url, target, created_at)
            VALUES (:league, :season, :match_date, :home_team, :away_team, :url, :target, :created_at)
            ON CONFLICT (league, season, match_date, home_team, away_team)
            DO NOTHING;
        """), row.to_dict())

print("✅ Data uložena do tabulky fixtures v DB (bez duplikací)")
