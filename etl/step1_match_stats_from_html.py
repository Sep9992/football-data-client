# etl/step1_match_stats_from_html.py
import os
import re
import datetime
import shutil
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

# --- načtení .env ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

base_dir = os.path.dirname(os.path.dirname(__file__))
html_dir = os.path.join(base_dir, "htmlFiles", "PL2025-26")
processed_dir = os.path.join(base_dir, "htmlFiles", "processed")
failed_dir = os.path.join(base_dir, "htmlFiles", "failed")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(failed_dir, exist_ok=True)

# --- pomocné funkce ---
def parse_number(val: str):
    if val is None:
        return None
    s = val.strip().replace("%", "").replace("\u00a0", " ")
    if not s or s in {"-", "—"}:
        return None
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None

def extract_values(block, selector=".team-stats__value"):
    vals = []
    for el in block.select(selector):
        num = parse_number(el.get_text(strip=True))
        if num is not None:
            vals.append(num)
    if len(vals) < 2:
        for el in block.find_all(["span", "div"]):
            num = parse_number(el.get_text(strip=True))
            if num is not None:
                vals.append(num)
    return vals[:2] if len(vals) >= 2 else [None, None]

def extract_pair(soup, class_names, caster=int):
    """
    Najde první existující blok podle zadaných CSS tříd a vrátí (home, away) převedené na čísla.
    Pokud se nic nenajde nebo čísla nejdou přečíst, vrací (0, 0).
    """
    for cls in class_names:
        block = soup.find("div", class_=cls)
        if block:
            h, a = extract_values(block)
            try:
                hv = caster(h) if h is not None else 0
            except Exception:
                hv = 0
            try:
                av = caster(a) if a is not None else 0
            except Exception:
                av = 0
            return hv, av
    return 0, 0

# --- hlavní parsovací funkce ---
def parse_match_statistics(html_content, log_prefix=""):
    soup = BeautifulSoup(html_content, "lxml")

    results = {}
    # základní metriky
    results["yellow_cards_home"], results["yellow_cards_away"] = extract_pair(
        soup, ["club-stats__yellowCards", "club-stats__yellow-cards"], int
    )
    results["red_cards_home"], results["red_cards_away"] = extract_pair(
        soup, ["club-stats__redCards", "club-stats__red-cards"], int
    )
    results["fouls_home"], results["fouls_away"] = extract_pair(
        soup, ["club-stats__fouls"], int
    )
    results["corners_home"], results["corners_away"] = extract_pair(
        soup, ["club-stats__corners"], int
    )
    results["shots_home"], results["shots_away"] = extract_pair(
        soup, ["club-stats__shots"], int
    )
    # saves: původní i nová varianta
    results["saves_home"], results["saves_away"] = extract_pair(
        soup, ["club-stats__saves", "club-stats__goalkeeper_saves"], int
    )
    # possession
    results["possession_home"], results["possession_away"] = extract_pair(
        soup, ["club-stats__possession", "club-stats__ball-possession"], float
    )
    # xG
    results["expected_goals_home"], results["expected_goals_away"] = extract_pair(
        soup, ["club-stats__expectedGoals", "club-stats__xg"], float
    )

    # nové varianty dle tvých ukázek
    results["shots_on_target_home"], results["shots_on_target_away"] = extract_pair(
        soup, ["club-stats__shotsOnTarget", "club-stats__shots-on-target", "club-stats__shotsOnGoals"], int
    )
    results["passes_total_home"], results["passes_total_away"] = extract_pair(
        soup, ["club-stats__passes", "club-stats__passes-total", "club-stats__total_passes"], int
    )
    results["passes_completed_home"], results["passes_completed_away"] = extract_pair(
        soup, ["club-stats__passes_accurate"], int
    )
    results["pass_accuracy_home"], results["pass_accuracy_away"] = extract_pair(
        soup, ["club-stats__passAccuracy", "club-stats__18"], float
    )
    results["shots_inside_box_home"], results["shots_inside_box_away"] = extract_pair(
        soup, ["club-stats__shots_insidebox"], int
    )
    results["shots_outside_box_home"], results["shots_outside_box_away"] = extract_pair(
        soup, ["club-stats__shots_outsidebox"], int
    )
    results["blocked_shots_home"], results["blocked_shots_away"] = extract_pair(
        soup, ["club-stats__blocked_shots"], int
    )

    # logování chybějících sekcí, které typicky nebývají nulové
    likely_nonzero = [
        "shots_home","shots_away",
        "shots_on_target_home","shots_on_target_away",
        "shots_inside_box_home","shots_inside_box_away",
        "shots_outside_box_home","shots_outside_box_away",
        "blocked_shots_home","blocked_shots_away",
        "passes_total_home","passes_total_away",
        "passes_completed_home","passes_completed_away",
        "pass_accuracy_home","pass_accuracy_away",
        "expected_goals_home","expected_goals_away",
        "possession_home","possession_away",
    ]
    missing = [k for k in likely_nonzero if results.get(k, 0) in (0, 0.0)]
    if missing:
        print(f"{log_prefix}⚠️ Chybějící/nerozpoznané: {', '.join(missing)}")

    return results

# --- datum ---
def extract_date(soup):
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and "content" in meta_desc.attrs:
        m = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})", meta_desc["content"])
        if m:
            return m.group(1)
    meta_og = soup.find("meta", attrs={"property": "og:title"})
    if meta_og and "content" in meta_og.attrs:
        m = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})", meta_og["content"])
        if m:
            return m.group(1)
    title_tag = soup.find("title")
    if title_tag:
        m = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})", title_tag.get_text())
        if m:
            return m.group(1)
    date_tag = soup.select_one(".match__date-formatted")
    if date_tag:
        return date_tag.get_text(strip=True)
    return None

# --- góly ---
def extract_goals_from_title(soup):
    title_tag = soup.find("title")
    if not title_tag:
        return None, None
    m = re.search(r"-\s*(\d+):(\d+)", title_tag.get_text())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# --- připojení k DB ---
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    processed = 0
    skipped_no_fixture = 0
    skipped_duplicate = 0

    for fname in os.listdir(html_dir):
        if not (fname.endswith(".html") or fname.endswith(".htm")):
            continue

        path = os.path.join(html_dir, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "lxml")

        # --- datum ---
        date_str = extract_date(soup)
        match_date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        if pd.isna(match_date):
            print(f"⚠️ Neplatné datum v {fname}")
            shutil.copy(path, os.path.join(failed_dir, fname))
            continue

        # --- týmy ---
        team_tags = soup.select(".match-scoreboard__club-title")
        home = team_tags[0].get_text(strip=True) if len(team_tags) > 0 else None
        away = team_tags[1].get_text(strip=True) if len(team_tags) > 1 else None
        if not home or not away:
            print(f"⚠️ Chybí tým(y) v {fname}")
            shutil.copy(path, os.path.join(failed_dir, fname))
            continue

        # --- fixture_id ---
        fixture_id = conn.execute(text("""
            SELECT id FROM fixtures
            WHERE league = 'PL'
              AND season = '2025-26'
              AND match_date = :match_date
              AND home_team = :home_team
              AND away_team = :away_team
        """), {
            "match_date": match_date.date(),
            "home_team": home,
            "away_team": away
        }).scalar()

        print(f"Soubor {fname}: date={date_str}, home={home}, away={away}, fixture_id={fixture_id}")

        if fixture_id is None:
            skipped_no_fixture += 1
            print(f"⚠️ Nenalezen fixture pro {match_date.date()} {home} vs {away} (soubor {fname})")
            shutil.copy(path, os.path.join(failed_dir, fname))
            continue

        # --- duplicitní kontrola ---
        existing = conn.execute(text("""
            SELECT 1 FROM match_statistics WHERE fixture_id = :fixture_id
        """), {"fixture_id": fixture_id}).scalar()
        if existing:
            skipped_duplicate += 1
            shutil.move(path, os.path.join(processed_dir, fname))
            continue

        # --- góly ---
        goals_home, goals_away = extract_goals_from_title(soup)

        row = {
            "fixture_id": fixture_id,
            "league": "PL",
            "season": "2025-26",
            "goals_home": goals_home if goals_home is not None else 0,
            "goals_away": goals_away if goals_away is not None else 0,
            "created_at": datetime.datetime.now(),
        }

        # --- metriky ze statistik (robustní) ---
        stats = parse_match_statistics(html, log_prefix=f"[{fname}] ")
        row.update(stats)

        # --- INSERT do DB (se všemi sloupci) ---
        conn.execute(text("""
            INSERT INTO match_statistics (
                fixture_id, league, season,
                goals_home, goals_away,
                possession_home, possession_away,
                passes_total_home, passes_total_away,
                passes_completed_home, passes_completed_away,
                pass_accuracy_home, pass_accuracy_away,
                expected_goals_home, expected_goals_away,
                shots_home, shots_away,
                shots_on_target_home, shots_on_target_away,
                shots_inside_box_home, shots_inside_box_away,
                shots_outside_box_home, shots_outside_box_away,
                blocked_shots_home, blocked_shots_away,
                saves_home, saves_away,
                yellow_cards_home, yellow_cards_away,
                red_cards_home, red_cards_away,
                fouls_home, fouls_away,
                corners_home, corners_away,
                created_at
            )
            VALUES (
                :fixture_id, :league, :season,
                :goals_home, :goals_away,
                :possession_home, :possession_away,
                :passes_total_home, :passes_total_away,
                :passes_completed_home, :passes_completed_away,
                :pass_accuracy_home, :pass_accuracy_away,
                :expected_goals_home, :expected_goals_away,
                :shots_home, :shots_away,
                :shots_on_target_home, :shots_on_target_away,
                :shots_inside_box_home, :shots_inside_box_away,
                :shots_outside_box_home, :shots_outside_box_away,
                :blocked_shots_home, :blocked_shots_away,
                :saves_home, :saves_away,
                :yellow_cards_home, :yellow_cards_away,
                :red_cards_home, :red_cards_away,
                :fouls_home, :fouls_away,
                :corners_home, :corners_away,
                :created_at
            )
            ON CONFLICT DO NOTHING;
        """), row)

        processed += 1
        shutil.move(path, os.path.join(processed_dir, fname))

    print(f"✅ Zpracováno {processed} souborů. Přeskočeno (bez fixture): {skipped_no_fixture}, duplicitní: {skipped_duplicate}")
