"""
step1_match_stats_from_livesport.py  - Univerzální verze
=========================================================
Stahuje statistiky zápasů z livesport.cz a ukládá do DB.

POUŽITÍ:
  # Archiv PL 2024-25
  python step1_match_stats_from_livesport.py \\
      --links data/links_premier-league.txt \\
      --league PL --season 2024-25

  # Aktuální sezóna PL 2025-26
  python step1_match_stats_from_livesport.py \\
      --links data/links.txt \\
      --league PL --season 2025-26

  # Bundesliga archiv
  python step1_match_stats_from_livesport.py \\
      --links data/links_bundesliga.txt \\
      --league BL --season 2024-25

  # Přeskočit již stažené (doporučeno při opakovaném spuštění)
  python step1_match_stats_from_livesport.py \\
      --links data/links_premier-league.txt \\
      --league PL --season 2024-25 --skip-existing

  # Dry-run (jen zobraz co by se stalo)
  python step1_match_stats_from_livesport.py \\
      --links data/links_premier-league.txt \\
      --league PL --season 2024-25 --dry-run

PODPOROVANÉ LIGY:
  PL  - Premier League (Anglie) — s aliasy pro matching s existujícími fixtures
  BL  - Bundesliga (Německo) — bez aliasů, používá názvy z Livesport
  LL  - La Liga (Španělsko) — bez aliasů, používá názvy z Livesport
  SA  - Serie A (Itálie) — bez aliasů, používá názvy z Livesport
  FL  - Fortuna liga (Česká republika) — bez aliasů, používá názvy z Livesport

POZNÁMKA K ALIASŮM:
  Pouze PL má aliasy (např. "Tottenham" → "Tottenham Hotspur"), protože
  existující fixtures v DB byly importovány z původního API s plnými názvy.
  Ostatní ligy budou vytvářet fixtures s názvy přímo z Livesport.

KLÍČOVÁ VYLEPŠENÍ oproti původní verzi:
  ✅ CLI argumenty: --links, --league, --season, --skip-existing, --dry-run
  ✅ league a season se přebírají z CLI, NE hardcoded
  ✅ Auto-insert: pokud fixture chybí v DB, vytvoří se automaticky
  ✅ LEAGUE_CONFIG: aliasy per liga (PL, BL, LL, SA, FL)
  ✅ find_fixture_id filtruje i podle league + season
  ✅ --skip-existing: přeskočí URL které už mají match_statistics
  ✅ Statistiky spuštění: kolik staženo/přeskočeno/chyb
"""

import os
import re
import time
import random
import sys
import datetime
import argparse
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURACE
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ CHYBA: Nenalezeno DATABASE_URL v .env souboru!")
    sys.exit(1)

engine = create_engine(DATABASE_URL)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CACHE pro DB schéma a missing columns log
# ─────────────────────────────────────────────────────────────────────────────
_DB_COLUMNS_CACHE = None
_MISSING_COLUMNS_LOG = set()

# ─────────────────────────────────────────────────────────────────────────────
# 2. KONFIGURACE LING (aliasy jen pro PL)
# ─────────────────────────────────────────────────────────────────────────────
#
# POZNÁMKA: Pouze PL má aliasy pro matching s existujícími fixtures v DB,
#           které byly importovány z původního API s plnými názvy.
#           Ostatní ligy (BL, LL, SA, FL) nemají aliasy — používají názvy
#           přímo z Livesport a auto-insert vytvoří nové fixtures.
# ─────────────────────────────────────────────────────────────────────────────

LEAGUE_CONFIG = {
    "PL": {
        "name": "Premier League",
        "country": "Anglie",
        "aliases": {
            # Zkrácené → Plné jméno v DB
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
            "Bournemouth": "AFC Bournemouth",
        }
    },
    "BL": {
        "name": "Bundesliga",
        "country": "Německo",
        "aliases": {}  # Bundesliga: používají se názvy přímo z Livesport
    },
    "LL": {
        "name": "La Liga",
        "country": "Španělsko",
        "aliases": {}  # La Liga: používají se názvy přímo z Livesport
    },
    "SA": {
        "name": "Serie A",
        "country": "Itálie",
        "aliases": {}  # Serie A: používají se názvy přímo z Livesport
    },
    "FL": {
        "name": "Fortuna liga",
        "country": "Česká republika",
        "aliases": {}  # Fortuna liga: používají se názvy přímo z Livesport
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTIKY (mapování livesport → DB sloupce)
# ─────────────────────────────────────────────────────────────────────────────

STATS_TO_FIND = [
    "Očekávané góly (xG)", "xGOT", "Držení míče", "Střely celkem", "Střely na branku",
    "Střely mimo branku", "Zblokované střely", "Střely uvnitř vápna", "Střely mimo vápno",
    "Střely do tyče/břevna", "Velké šance", "Rohové kopy", "Doteky ve vápně soupeře",
    "Přesné průnikové přihrávky", "Ofsajdy", "Přímé kopy", "Přihrávky", "Dlouhé přihrávky",
    "Přihrávky v útočné třetině", "Centry", "Očekávané asistence (xA)", "Vhazování",
    "Fauly", "Žluté karty", "Červené karty", "Obranné zákroky", "Vyhrané souboje",
    "Obranné odkopy", "Přerušení přihrávek", "Chyby vedoucí ke střele",
    "Chyby vedoucí ke gólu", "Brankářské zákroky", "xGOT proti", "Zabráněné góly"
]

STATS_DB_MAP = {
    "Očekávané góly (xG)":         "expected_goals",
    "xGOT":                         "xgot",
    "Držení míče":                  "possession",
    "Střely celkem":                "shots",
    "Střely na branku":             "shots_on_target",
    "Střely mimo branku":           "shots_off_target",
    "Zblokované střely":            "blocked_shots",
    "Střely uvnitř vápna":          "shots_inside_box",
    "Střely mimo vápno":            "shots_outside_box",
    "Střely do tyče/břevna":        "woodwork",
    "Velké šance":                  "big_chances",
    "Rohové kopy":                  "corners",
    "Doteky ve vápně soupeře":      "box_touches",
    "Přesné průnikové přihrávky":   "through_balls",
    "Ofsajdy":                      "offsides",
    "Přímé kopy":                   "free_kicks",
    "Očekávané asistence (xA)":     "expected_assists",
    "Vhazování":                    "throw_ins",
    "Fauly":                        "fouls",
    "Žluté karty":                  "yellow_cards",
    "Červené karty":                "red_cards",
    "Vyhrané souboje":              "duels_won",
    "Obranné odkopy":               "clearances",
    "Přerušení přihrávek":          "interceptions",
    "Brankářské zákroky":           "saves",
    "Zabráněné góly":               "prevented_goals",
    "Přihrávky":                    "passes",
    "Dlouhé přihrávky":             "long_balls",
    "Přihrávky v útočné třetině":   "passes_final_third",
    "Centry":                       "crosses",
    "Obranné zákroky":              "tackles",
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. POMOCNÉ FUNKCE
# ─────────────────────────────────────────────────────────────────────────────

def parse_complex_stat(value_str):
    """Rozebere text jako '86% (447/519)' na (86, 447, 519)."""
    if not value_str: return None, None, None
    match = re.search(r'(\d+)%\s*\((\d+)/(\d+)\)', value_str)
    if match: return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match_frac = re.search(r'\((\d+)/(\d+)\)', value_str)
    if match_frac: return None, int(match_frac.group(1)), int(match_frac.group(2))
    match_pct = re.search(r'(\d+)%', value_str)
    if match_pct: return int(match_pct.group(1)), None, None
    try:
        clean_val = float(value_str.replace(',', '.').replace('%', ''))
        return None, clean_val, None
    except:
        return None, None, None


def get_match_metadata(driver):
    """Získá jména týmů, datum a skóre ze stránky Livesport."""
    try:
        home_team = driver.find_element(
            By.XPATH, "//div[contains(@class, 'home')]//div[contains(@class, 'participantName')]"
        ).text.strip()
        away_team = driver.find_element(
            By.XPATH, "//div[contains(@class, 'away')]//div[contains(@class, 'participantName')]"
        ).text.strip()
        date_str  = driver.find_element(
            By.XPATH, "//div[contains(@class, 'startTime')]"
        ).text.strip()
        match_date = datetime.datetime.strptime(date_str.split()[0], "%d.%m.%Y").date()

        goals_home, goals_away = None, None
        try:
            score_text = driver.find_element(By.CLASS_NAME, "detailScore__wrapper").text.strip()
            scores = re.findall(r'\d+', score_text)
            if len(scores) >= 2:
                goals_home = int(scores[0])
                goals_away = int(scores[1])
                print(f"    ⚽ Skóre: {goals_home}:{goals_away}")
        except:
            pass  # Zápas ještě neskončil nebo jiný formát

        print(f"    📋 Zápas: {home_team} vs {away_team} ({match_date})")
        return home_team, away_team, match_date, goals_home, goals_away

    except Exception as e:
        print(f"    ⚠️  Nepodařilo se načíst metadata: {e}")
        return None, None, None, None, None


def get_already_scraped_urls():
    """
    Vrátí sadu URL, které už mají záznam v match_statistics.
    Použito pro --skip-existing.
    """
    try:
        query = text("""
            SELECT f.url
            FROM fixtures f
            INNER JOIN match_statistics ms ON ms.fixture_id = f.id
            WHERE f.url IS NOT NULL
        """)
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        return {row[0] for row in rows}
    except Exception as e:
        print(f"  ⚠️  Nepodařilo se načíst již stažené URL: {e}")
        return set()


def find_or_create_fixture(home_raw, away_raw, match_date, url,
                            league, season, aliases, dry_run=False):
    """
    KLÍČOVÁ FUNKCE: Najde fixture v DB, nebo ho vytvoří.

    Postup:
    1. Aplikuje aliasy (livesport jméno → DB jméno)
    2. Hledá v DB podle home_team, away_team, match_date, league, season
    3. Pokud nenajde → INSERT nového fixture (auto-insert)
    4. Vrátí fixture_id

    PROČ auto-insert:
    - Archivní sezóny (2024-25) nejsou v DB jako fixtures
    - Živé sezóny jiných lig (BL, LL...) také nejsou v DB
    - Místo nutnosti spouštět step0 pro každou ligu,
      step1 si fixture vytvoří sám z dat na livesport stránce
    """
    # Aplikace aliasů pro HLEDÁNÍ (existující fixtures v DB)
    home_db = aliases.get(home_raw, home_raw)
    away_db = aliases.get(away_raw, away_raw)

    d_minus = match_date - datetime.timedelta(days=2)
    d_plus  = match_date + datetime.timedelta(days=2)

    try:
        # --- Krok 1: Hledej existující fixture ---
        query_find = text("""
            SELECT id FROM fixtures
            WHERE home_team ILIKE :h
              AND away_team ILIKE :a
              AND match_date BETWEEN :d1 AND :d2
              AND league = :league
            LIMIT 1
        """)
        with engine.connect() as conn:
            result = conn.execute(query_find, {
                "h": home_db, "a": away_db,
                "d1": d_minus, "d2": d_plus,
                "league": league
            }).fetchone()

        if result:
            print(f"    🔗 Nalezen existující fixture ID={result[0]}")
            return result[0]

        # --- Krok 2: Zkus bez league filtru (fallback pro přejmenované týmy) ---
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id FROM fixtures
                WHERE home_team ILIKE :h
                  AND away_team ILIKE :a
                  AND match_date BETWEEN :d1 AND :d2
                LIMIT 1
            """), {"h": home_db, "a": away_db, "d1": d_minus, "d2": d_plus}).fetchone()

        if result:
            print(f"    🔗 Nalezen fixture ID={result[0]} (bez league filtru)")
            return result[0]

        # --- Krok 3: Auto-insert nového fixture ---
        # Používáme RAW jméno z livesport (bez aliasů) pro nové záznamy
        # aby bylo konzistentní napříč sezónami
        if dry_run:
            print(f"    🔧 [DRY-RUN] Vytvořil bych fixture: "
                  f"{home_raw} vs {away_raw} ({match_date}) [{league} {season}]")
            return -1  # Dummy ID pro dry-run

        query_insert = text("""
            INSERT INTO fixtures (league, season, match_date, home_team, away_team, url)
            VALUES (:league, :season, :date, :home, :away, :url)
            ON CONFLICT DO NOTHING
            RETURNING id
        """)
        with engine.begin() as conn:
            result = conn.execute(query_insert, {
                "league":  league,
                "season":  season,
                "date":    match_date,
                "home":    home_raw,   # RAW jméno z livesport
                "away":    away_raw,   # RAW jméno z livesport
                "url":     url
            }).fetchone()

        if result:
            print(f"    ✨ Vytvořen nový fixture ID={result[0]} "
                  f"[{home_raw} vs {away_raw}, {league} {season}]")
            return result[0]
        else:
            # ON CONFLICT DO NOTHING → zkus najít znovu
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id FROM fixtures
                    WHERE home_team = :home AND away_team = :away
                      AND match_date = :date AND league = :league
                """), {"home": home_raw, "away": away_raw,
                       "date": match_date, "league": league}).fetchone()
            if result:
                return result[0]

        print(f"    ❌ Nepodařilo se vytvořit fixture pro {home_raw} vs {away_raw}")
        return None

    except Exception as e:
        print(f"    ❌ Chyba v find_or_create_fixture: {e}")
        return None


def get_db_columns():
    """Načte seznam sloupců v match_statistics tabulce (cachované)."""
    global _DB_COLUMNS_CACHE
    if _DB_COLUMNS_CACHE is not None:
        return _DB_COLUMNS_CACHE

    try:
        query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'match_statistics'
        """)
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        _DB_COLUMNS_CACHE = {row[0] for row in rows}
        print(f"\n  📋 DB schéma načteno: {len(_DB_COLUMNS_CACHE)} sloupců")
        return _DB_COLUMNS_CACHE
    except Exception as e:
        print(f"  ⚠️  Nelze načíst DB schéma: {e}")
        return set()


def save_to_db(fixture_id, extracted_data, goals_home, goals_away,
               league, season, dry_run=False):
    """
    Uloží (UPSERT) statistiky do match_statistics.

    ROBUSTNÍ VERZE:
    - Dynamicky zjistí které sloupce v DB existují
    - Přeskočí sloupce které chybí (místo crash)
    - Zaloguje chybějící sloupce do globálního setu
    """
    global _MISSING_COLUMNS_LOG

    if not fixture_id or not extracted_data:
        return

    if dry_run:
        print(f"    🔧 [DRY-RUN] Uložil bych {len(extracted_data)} statistik")
        return

    # Načti DB schéma (jen jednou, pak cache)
    db_columns = get_db_columns()
    if not db_columns:
        print(f"    ❌ Nelze uložit — DB schéma nenačteno")
        return

    params = {
        "fid":    fixture_id,
        "league": league,
        "season": season,
        "gh":     goals_home,
        "ga":     goals_away
    }

    columns             = []
    values_placeholders = []
    update_parts        = []

    # Helper: přidá sloupec jen pokud existuje v DB
    def add_col(col_name, param_name, value):
        if col_name in db_columns:
            columns.append(col_name)
            values_placeholders.append(f":{param_name}")
            update_parts.append(f"{col_name} = EXCLUDED.{col_name}")
            params[param_name] = value
            return True
        else:
            _MISSING_COLUMNS_LOG.add(col_name)
            return False

    # Základní sloupce
    add_col("fixture_id", "fid", fixture_id)
    add_col("league", "league", league)
    add_col("season", "season", season)

    if goals_home is not None:
        add_col("goals_home", "gh", goals_home)
    if goals_away is not None:
        add_col("goals_away", "ga", goals_away)

    # Statistiky
    for stat_name, values in extracted_data.items():
        base_col = STATS_DB_MAP.get(stat_name)
        if not base_col:
            continue

        (h_pct, h_val, h_tot), (a_pct, a_val, a_tot) = values
        COMPLEX = {"passes", "long_balls", "passes_final_third", "crosses", "tackles"}

        if base_col in COMPLEX:
            for suffix, val in [
                ("_accuracy_home", h_pct), ("_completed_home", h_val), ("_total_home", h_tot),
                ("_accuracy_away", a_pct), ("_completed_away", a_val), ("_total_away", a_tot)
            ]:
                if val is not None:
                    col = f"{base_col}{suffix}"
                    add_col(col, col, val)  # param = col
        else:
            for suffix, val in [("_home", h_val or h_pct), ("_away", a_val or a_pct)]:
                if val is not None:
                    col = f"{base_col}{suffix}"
                    add_col(col, col, val)

    if not update_parts:
        print(f"    ⚠️  Žádné sloupce k uložení")
        return

    query = text(f"""
        INSERT INTO match_statistics ({', '.join(columns)})
        VALUES ({', '.join(values_placeholders)})
        ON CONFLICT (fixture_id)
        DO UPDATE SET {', '.join(update_parts)}, created_at = NOW();
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, params)
        saved = len([c for c in columns if c not in ['fixture_id','league','season']])
        print(f"    💾 Uloženo {saved} statistik")
    except Exception as e:
        print(f"    ❌ Chyba: {str(e)[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SELENIUM POMOCNÉ FUNKCE
# ─────────────────────────────────────────────────────────────────────────────

def close_popups(driver):
    """Zavře cookies a popupy."""
    try:
        accept_btn = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        driver.execute_script("arguments[0].click();", accept_btn)
        print("    🍪 Cookies přijaty.")
        time.sleep(0.5)
    except:
        pass
    try:
        script = """
        var buttons = document.querySelectorAll('button, div[role="button"]');
        for (var b of buttons) {
            var t = b.innerText.toLowerCase();
            if (t.includes("rozumím") || t.includes("zavřít") || t.includes("close")) {
                b.click(); return true;
            }
        }
        return false;
        """
        driver.execute_script(script)
    except:
        pass


def click_stats_tab(driver):
    """Aktivuje záložku Statistiky."""
    for text_val in ["STATISTIKY", "Statistiky"]:
        try:
            script = f"""
            var elements = document.querySelectorAll('a, div, span, button');
            for (var el of elements) {{
                if (el.innerText && el.innerText.trim().toUpperCase() === "{text_val.upper()}") {{
                    el.click(); return true;
                }}
            }}
            return false;
            """
            if driver.execute_script(script):
                print(f"    ✅ Záložka Statistiky aktivována.")
                return True
        except:
            continue

    try:
        script_href = """
        var link = document.querySelector('a[href*="statistiky-zapasu"]');
        if (link) { link.click(); return true; }
        return false;
        """
        if driver.execute_script(script_href):
            print("    ✅ Záložka aktivována přes href.")
            return True
    except:
        pass

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 6. HLAVNÍ SCRAPING FUNKCE
# ─────────────────────────────────────────────────────────────────────────────

def scrape_match(driver, url, league, season, aliases, dry_run=False):
    """Stáhne statistiky jednoho zápasu."""
    print(f"\n{'='*70}")
    print(f"🌐 {url}")

    driver.get(url)
    time.sleep(random.uniform(3, 5))
    close_popups(driver)

    # Metadata
    home, away, match_date, g_home, g_away = get_match_metadata(driver)
    if not home or not match_date:
        print("    ❌ Nepodařilo se identifikovat zápas.")
        return False

    # Fixture (najdi nebo vytvoř)
    fixture_id = find_or_create_fixture(
        home, away, match_date, url,
        league, season, aliases, dry_run
    )
    if not fixture_id:
        print(f"    ⏭️  Přeskakuji {home} - {away} (fixture_id nenalezeno/nevytvořeno).")
        return False

    # Záložka statistiky
    if not click_stats_tab(driver):
        print("    ❌ Záložka Statistiky nenalezena.")
        return False
    time.sleep(2)

    # Extrakce statistik
    print(f"\n    {'DOMÁCÍ':<20} | {'STATISTIKA':^30} | {'HOSTÉ':>20}")
    print("    " + "─" * 76)

    extracted_data = {}
    found_stats    = set()

    for stat_name in STATS_TO_FIND:
        if stat_name in found_stats: continue
        try:
            elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{stat_name}')]")
            for el in elements:
                if not el.is_displayed(): continue
                parent = el.find_element(By.XPATH, "./..")
                if not any(c.isdigit() for c in parent.text):
                    parent = parent.find_element(By.XPATH, "./..")
                lines = [l.strip() for l in parent.text.split('\n') if l.strip()]

                if len(lines) >= 5:
                    raw_h = f"{lines[0]} {lines[1]}"; label = lines[2]; raw_a = f"{lines[3]} {lines[4]}"
                elif len(lines) == 3:
                    raw_h, label, raw_a = lines[0], lines[1], lines[2]
                else:
                    continue

                if stat_name in label:
                    res_h = parse_complex_stat(raw_h)
                    res_a = parse_complex_stat(raw_a)
                    print(f"    {raw_h:<20} | {stat_name:^30} | {raw_a:>20}")
                    extracted_data[stat_name] = (res_h, res_a)
                    found_stats.add(stat_name)
                    break
        except:
            continue

    # Uložení
    save_to_db(fixture_id, extracted_data, g_home, g_away, league, season, dry_run)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_missing_columns_summary():
    """Vypíše summary chybějících sloupců na konci."""
    if _MISSING_COLUMNS_LOG:
        print("\n" + "=" * 70)
        print("⚠️  CHYBĚJÍCÍ SLOUPCE V DB")
        print("=" * 70)
        for col in sorted(_MISSING_COLUMNS_LOG):
            print(f"  - {col}")
        print(f"\nCelkem: {len(_MISSING_COLUMNS_LOG)} sloupců chybí")
        print("Tip: Uprav DB schéma nebo ignoruj (data se uloží částečně).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stahuje statistiky zápasů z livesport.cz do DB."
    )
    parser.add_argument(
        "--links", required=True,
        help="Cesta k souboru s URL zápasů (např. data/links_premier-league.txt)"
    )
    parser.add_argument(
        "--league", required=True, choices=list(LEAGUE_CONFIG.keys()),
        help=f"Kód ligy: {', '.join(LEAGUE_CONFIG.keys())}"
    )
    parser.add_argument(
        "--season", required=True,
        help="Sezóna ve formátu RRRR-RR (např. 2024-25 nebo 2025-26)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Přeskoč URL které už mají statistiky v DB (doporučeno při opakovaném spuštění)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Jen zobraz co by se stalo, nic nezapisuj do DB"
    )
    parser.add_argument(
        "--chrome-version", type=int, default=None,
        help="Verze Chrome (výchozí: autodetekce)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Konfigurace ligy
    league_info = LEAGUE_CONFIG[args.league]
    aliases     = league_info["aliases"]

    # Načtení linků
    links_path = args.links
    if not os.path.isabs(links_path):
        links_path = os.path.join(BASE_DIR, links_path)

    if not os.path.exists(links_path):
        print(f"❌ Soubor s linky nenalezen: {links_path}")
        sys.exit(1)

    with open(links_path, "r") as f:
        all_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Filtr --skip-existing
    already_scraped = set()
    if args.skip_existing:
        already_scraped = get_already_scraped_urls()
        print(f"  ⏭️  Již staženo: {len(already_scraped)} URL")

    urls = [u for u in all_urls if u not in already_scraped]

    # ── HEADER ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"🚀 STEP1: STAHOVÁNÍ STATISTIK ZÁPASŮ")
    print("=" * 70)
    print(f"  Liga:      {args.league} — {league_info['name']} ({league_info['country']})")
    print(f"  Sezóna:    {args.season}")
    print(f"  Linky:     {links_path}")
    print(f"  Celkem URL: {len(all_urls)}")
    if args.skip_existing:
        print(f"  Přeskočeno (již staženo): {len(all_urls) - len(urls)}")
    print(f"  Ke zpracování: {len(urls)}")
    if args.dry_run:
        print("  ⚠️  DRY-RUN mód — nic se nezapíše do DB!")
    print("=" * 70)

    if not urls:
        print("✅ Vše již staženo. Nic k dělání.")
        return

    # ── CHROME ──────────────────────────────────────────────────────────────
    print("\n🌐 Spouštím Chrome...")
    options = uc.ChromeOptions()
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')

    chrome_kwargs = {"options": options}
    if args.chrome_version:
        chrome_kwargs["version_main"] = args.chrome_version

    driver = uc.Chrome(**chrome_kwargs)

    # ── SCRAPING ─────────────────────────────────────────────────────────────
    stats = {"ok": 0, "skip": 0, "error": 0}

    try:
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]", end="")
            try:
                ok = scrape_match(driver, url, args.league, args.season,
                                  aliases, args.dry_run)
                if ok:
                    stats["ok"] += 1
                else:
                    stats["skip"] += 1
            except Exception as e:
                print(f"\n    ❌ Chyba: {e}")
                stats["error"] += 1

            # Anti-bot pauza
            if i < len(urls):
                pause = random.uniform(2, 4)
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n\n⛔ Přerušeno uživatelem.")

    finally:
        print("\n🏁 Ukončuji prohlížeč...")
        try: driver.quit()
        except: pass

        # Vypíše které sloupce chyběly v DB
        print_missing_columns_summary()

    # ── ZÁVĚREČNÉ STATISTIKY ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ HOTOVO!")
    print("=" * 70)
    print(f"  Liga / Sezóna:   {args.league} {args.season}")
    print(f"  Celkem URL:      {len(urls)}")
    print(f"  ✅ Staženo:      {stats['ok']}")
    print(f"  ⏭️  Přeskočeno:  {stats['skip']}")
    print(f"  ❌ Chyb:        {stats['error']}")
    if args.dry_run:
        print("  (DRY-RUN — nic nebylo zapsáno do DB)")
    print("=" * 70)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        class DevNull:
            def write(self, msg): pass
        sys.stderr = DevNull()
    main()