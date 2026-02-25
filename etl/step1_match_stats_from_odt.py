# etl/step1_match_stats_from_odt.py
# FIX: Přidána podpora pro "Brankářské zákroky" (Livesport varianta).

import os
import re
import datetime
from odf.opendocument import load
from odf import text, teletype
from sqlalchemy import create_engine, text as sql_text
from dotenv import load_dotenv

# --- KONFIGURACE ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ODT_DIR = os.path.join(BASE_DIR, "odtFiles", "PL2025-26")
DEFAULT_LEAGUE = "PL"
DEFAULT_SEASON = "2025-26"

os.makedirs(ODT_DIR, exist_ok=True)

# --- MAPOVÁNÍ STATISTIK ---
STATS_MAP = {
    "xGOT": "xgot",
    "Velké šance": "big_chances",
    "Doteky ve vápně soupeře": "box_touches",
    "Očekávané góly (xG)": "expected_goals",
    "Střely celkem": "shots",
    "Střely na branku": "shots_on_target",
    "Střely mimo branku": "shots_off_target",
    "Zblokované střely": "blocked_shots",
    "Střely uvnitř vápna": "shots_inside_box",
    "Střely mimo vápno": "shots_outside_box",
    "Střely do tyče/břevna": "woodwork",
    "Držení míče": "possession",
    "Ofsajdy": "offsides",
    "Rohové kopy": "corners",
    "Přesné průnikové přihrávky": "through_balls",
    "Fauly": "fouls",

    # OPRAVA: Přidány obě varianty názvosloví pro jistotu
    "Zákroky brankáře": "saves",
    "Brankářské zákroky": "saves",

    "Žluté karty": "yellow_cards",
    "Červené karty": "red_cards",
    "Přihrávky": "passes_total",
}


def parse_number(val_str):
    if not val_str: return None
    clean = val_str.strip().replace("%", "").replace(",", ".")
    if not clean or clean in ["-", "—"]: return None
    try:
        if "." in clean: return float(clean)
        return int(clean)
    except:
        return None


def parse_passes(val_str):
    if not val_str: return None, None, None
    match = re.search(r"(\d+)%\s*\((\d+)/(\d+)\)", val_str)
    if match:
        acc = float(match.group(1))
        comp = int(match.group(2))
        total = int(match.group(3))
        return comp, total, acc
    return None, None, None


def get_fixture_id_by_teams(conn, home_team, away_team):
    print(f"   🔍 Hledám v DB: '{home_team}' vs '{away_team}'")
    home_clean = home_team.strip()
    away_clean = away_team.strip()

    aliases = {
        "Sunderland": "AFC Sunderland",
        "Burnley": "Burnley FC",
        "Tottenham": "Tottenham Hotspur",
        "Manchester Utd": "Manchester United",
        "Fulham": "Fulham FC",
        "Nottingham": "Nottingham Forest",
        "Liverpool": "Liverpool FC",
        "Newcastle": "Newcastle United",
        "Chelsea": "Chelsea FC",
        "Leeds": "Leeds United",
        "West Ham": "West Ham United",
        "Brighton": "Brighton & Hove Albion",
        "Everton": "Everton FC",
        "Arsenal": "Arsenal FC",
        "Wolves": "Wolverhampton Wanderers",
        "Bournemouth": "AFC Bournemouth"
    }
    h = aliases.get(home_clean, home_clean)
    a = aliases.get(away_clean, away_clean)

    query = sql_text("""
        SELECT id, home_team, away_team FROM fixtures 
        WHERE LOWER(home_team) LIKE LOWER(:h) AND LOWER(away_team) LIKE LOWER(:a)
        ORDER BY match_date DESC LIMIT 1
    """)
    result = conn.execute(query, {"h": f"%{h}%", "a": f"%{a}%"}).fetchone()
    if result:
        print(f"      ✅ Nalezeno: ID {result[0]} ({result[1]} vs {result[2]})")
        return result[0]
    else:
        print(f"      ❌ Nenalezeno v DB (zkoušeno jako '{h}' vs '{a}').")
        return None


def process_odt_file(filepath, engine):
    print(f"📄 Zpracovávám: {os.path.basename(filepath)}")
    try:
        doc = load(filepath)
    except Exception as e:
        print(f"❌ Chyba při čtení ODT: {e}")
        return False

    all_text = []
    for p in doc.getElementsByType(text.P):
        txt = teletype.extractText(p).strip()
        if txt: all_text.append(txt)

    # 1. Týmy a Skóre
    score_idx = -1
    goals_home, goals_away = 0, 0
    home_team_name, away_team_name = None, None

    for i, line in enumerate(all_text[:30]):
        if re.match(r"^\d+-\d+$", line):
            score_idx = i
            parts = line.split("-")
            goals_home = int(parts[0])
            goals_away = int(parts[1])
            break

    if score_idx > 0:
        home_team_name = all_text[score_idx - 1]
        offset = 1
        while score_idx + offset < len(all_text):
            candidate = all_text[score_idx + offset]
            if candidate in ["Konec", "Po prodl.", "Na pen.", "Poločas", "Pen."] or ":" in candidate:
                offset += 1
                continue
            else:
                away_team_name = candidate
                break
    else:
        print("⚠️ Nepodařilo se najít skóre.")

    # 2. Fixture ID
    filename = os.path.basename(filepath)
    fixture_id = None
    match_id = re.match(r"^(\d+)_", filename)
    if match_id:
        fixture_id = int(match_id.group(1))

    if not fixture_id and home_team_name and away_team_name:
        with engine.connect() as conn:
            fixture_id = get_fixture_id_by_teams(conn, home_team_name, away_team_name)

    if not fixture_id:
        print(f"❌ SKIP: Nelze určit ID zápasu.")
        return False

    # 3. Statistiky
    stats_data = {
        "fixture_id": fixture_id,
        "league": DEFAULT_LEAGUE,
        "season": DEFAULT_SEASON,
        "goals_home": goals_home,
        "goals_away": goals_away,
        "created_at": datetime.datetime.now()
    }

    for i, line in enumerate(all_text):
        if line in STATS_MAP:
            db_col = STATS_MAP[line]
            if i > 0 and i < len(all_text) - 1:
                val_h_raw = all_text[i - 1]
                val_a_raw = all_text[i + 1]

                if line == "Přihrávky":
                    comp_h, total_h, acc_h = parse_passes(val_h_raw)
                    comp_a, total_a, acc_a = parse_passes(val_a_raw)
                    if total_h is not None:
                        stats_data["passes_completed_home"] = comp_h
                        stats_data["passes_total_home"] = total_h
                        stats_data["pass_accuracy_home"] = acc_h
                    if total_a is not None:
                        stats_data["passes_completed_away"] = comp_a
                        stats_data["passes_total_away"] = total_a
                        stats_data["pass_accuracy_away"] = acc_a
                else:
                    val_h = parse_number(val_h_raw)
                    val_a = parse_number(val_a_raw)
                    if val_h is not None: stats_data[f"{db_col}_home"] = val_h
                    if val_a is not None: stats_data[f"{db_col}_away"] = val_a

    # 4. Pojistka: Dopočet zákroků (jen pokud nebyly nalezeny v souboru)
    if "saves_home" not in stats_data:
        sot_away = stats_data.get("shots_on_target_away")
        g_away = stats_data.get("goals_away")
        if sot_away is not None and g_away is not None:
            stats_data["saves_home"] = max(0, sot_away - g_away)
            print("   🔧 Saves_home nebyly v souboru, dopočítány.")

    if "saves_away" not in stats_data:
        sot_home = stats_data.get("shots_on_target_home")
        g_home = stats_data.get("goals_home")
        if sot_home is not None and g_home is not None:
            stats_data["saves_away"] = max(0, sot_home - g_home)
            print("   🔧 Saves_away nebyly v souboru, dopočítány.")

    # 5. Uložení
    columns = list(stats_data.keys())
    placeholders = [f":{c}" for c in columns]
    update_clause = [f"{c} = EXCLUDED.{c}" for c in columns if c != "fixture_id"]

    sql = sql_text(f"""
        INSERT INTO match_statistics ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT (fixture_id) DO UPDATE SET
        {', '.join(update_clause)}
    """)

    try:
        with engine.begin() as conn:
            conn.execute(sql, stats_data)
        print("   ✅ Data uložena.")
        return True
    except Exception as e:
        print(f"   ❌ Chyba SQL: {e}")
        return False


def main():
    if not os.path.exists(ODT_DIR):
        print(f"Složka {ODT_DIR} neexistuje.")
        return
    engine = create_engine(DATABASE_URL)
    files = [f for f in os.listdir(ODT_DIR) if f.endswith(".odt")]
    if not files:
        print(f"Ve složce {ODT_DIR} nejsou žádné soubory.")
        return
    count = 0
    for f in files:
        if process_odt_file(os.path.join(ODT_DIR, f), engine):
            count += 1
    print(f"\nHotovo. Úspěšně zpracováno: {count}/{len(files)}")


if __name__ == "__main__":
    main()