"""
step0b_harvest_future_fixtures.py
==================================
Stahuje BUDOUCÍ zápasy (program) z Livesport a ukládá je do tabulky fixtures.
Spouštěj pravidelně (např. 1× týdně) aby byly predikce aktuální.

Zdroje:
  PL: https://www.livesport.cz/fotbal/anglie/premier-league/program/
  FL: https://www.livesport.cz/fotbal/cesko/chance-liga/program/

Logika:
  - Pokud zápas s daným (home_team, away_team, match_date) již v DB existuje → přeskočí
  - Pokud neexistuje → vloží nový řádek do fixtures (bez statistik, ty přijdou po odehrání)
  - Sezónu určí z data zápasu

Výstup:
  - Nové řádky v tabulce fixtures
  - Tabulka prepared_fixtures se aktualizuje až po spuštění step2
"""

import os
import re
import time
from datetime import datetime, date
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- KONFIGURACE ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

LEAGUES = {
    "premier-league": {
        "url":               "https://www.livesport.cz/fotbal/anglie/premier-league/program/",
        "league_code":       "PL",
        "season_start_month": 8,
    },
    "chance-liga": {
        "url":               "https://www.livesport.cz/fotbal/cesko/chance-liga/program/",
        "league_code":       "FL",
        "season_start_month": 7,
    },
}

# Kolik týdnů dopředu scrapovat (program stránka zobrazuje několik týdnů)
WEEKS_AHEAD = 4


# =============================================================================
# CHROME + HELPERS
# =============================================================================

def get_chrome_version():
    """Detekuje hlavní verzi Chrome z registru (Windows) nebo příkazové řádky."""
    try:
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for path in (
                r"Software\Google\Chrome\BLBeacon",
                r"Software\Wow6432Node\Google\Chrome\BLBeacon",
            ):
                try:
                    key = winreg.OpenKey(hive, path)
                    version, _ = winreg.QueryValueEx(key, "version")
                    major = int(version.split(".")[0])
                    print(f"   🔍 Chrome verze z registru: {version} → major={major}")
                    return major
                except Exception:
                    continue
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["google-chrome", "--version"],
            capture_output=True, text=True, timeout=5
        )
        m = re.search(r"(\d+)\.", result.stdout)
        if m:
            major = int(m.group(1))
            print(f"   🔍 Chrome verze z CLI: major={major}")
            return major
    except Exception:
        pass

    print("   ⚠️  Chrome verze nezjištěna, použije se auto-detect")
    return None


def handle_cookies(driver):
    """Automaticky akceptuje cookies banner."""
    try:
        accept_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        accept_btn.click()
        print("   🍪 Cookies akceptovány")
        time.sleep(1)
    except Exception:
        print("   ℹ️  Cookies banner se neobjevil")


def make_driver():
    """Inicializuje undetected Chrome driver."""
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--password-store=basic")
    options.add_argument("--disable-blink-features=AutomationControlled")

    chrome_version = get_chrome_version()
    chrome_kwargs = {"options": options, "use_subprocess": True}
    if chrome_version:
        chrome_kwargs["version_main"] = chrome_version

    driver = uc.Chrome(**chrome_kwargs)
    print("✅ Chrome driver inicializován")
    return driver


# =============================================================================
# PARSOVÁNÍ STRÁNKY
# =============================================================================

def determine_season(match_date, season_start_month):
    """Určí sezónu z data zápasu. Vrátí string jako '2025-26'."""
    y, m = match_date.year, match_date.month
    if m >= season_start_month:
        return f"{y}-{str(y+1)[-2:]}"
    else:
        return f"{y-1}-{str(y)[-2:]}"


def parse_date_from_element(el, year_hint=None):
    """
    Extrahuje datum z event__time elementu.
    Livesport na /program/ zobrazuje datum ve formátu 'DD.MM.' nebo 'DD.MM.YYYY'.
    year_hint: aktuální rok jako záchrana pokud rok chybí.
    """
    try:
        date_el = el.find_element(By.CLASS_NAME, "event__time")
        text = date_el.text.strip()

        # Formát s rokem: DD.MM.YYYY nebo DD.MM.YYYY HH:MM
        m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', text)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(year, month, day)

        # Formát bez roku: DD.MM. nebo DD.MM. HH:MM
        m = re.search(r'(\d{1,2})\.(\d{1,2})\.', text)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            today = date.today()
            # Zkus aktuální rok, pokud datum ještě nenastalo nebo je blízko
            for year_offset in (0, 1):
                year = today.year + year_offset
                try:
                    candidate = date(year, month, day)
                    if candidate >= today:
                        return candidate
                except ValueError:
                    continue
            # Poslední záchrana
            return date(year_hint or today.year, month, day)
    except Exception:
        pass
    return None


def parse_team_name(el, side):
    """
    Extrahuje název týmu z elementu zápasu.
    Na /program/ stránce je formát textu: 'DD.MM. HH:MM | HomeTeam | AwayTeam | ...'
    Primárně parsujeme z textu, jako záloha zkoušíme CSS třídy.
    """
    # Primární metoda: parsuj z textu elementu
    # Formát: 'DD.MM. HH:MM | HomeTeam | AwayTeam | PREVIEW'
    try:
        text = el.text.strip()
        # Rozděl na části podle '|' nebo nového řádku
        parts = [p.strip() for p in re.split(r'\||\n', text) if p.strip()]
        # parts[0] = 'DD.MM. HH:MM', parts[1] = HomeTeam, parts[2] = AwayTeam
        # Přeskoč první část (datum+čas) — ta obsahuje čísla a tečky
        team_parts = [p for p in parts if not re.match(r'^\d{1,2}\.\d{1,2}', p)
                      and p not in ('PREVIEW', 'FT', '-', 'AET', 'PEN')]
        if len(team_parts) >= 2:
            if side == "home":
                return team_parts[0]
            else:
                return team_parts[1]
    except Exception:
        pass

    # Záloha: CSS třídy
    for cls in (f"event__participant--{side}", f"event__team--{side}"):
        try:
            team_el = el.find_element(By.CLASS_NAME, cls)
            name = team_el.text.strip()
            if name:
                return name
        except Exception:
            pass

    # Druhá záloha: pořadí participant elementů
    try:
        participants = el.find_elements(
            By.XPATH, ".//*[contains(@class, 'event__participant')]"
        )
        idx = 0 if side == "home" else 1
        if len(participants) > idx:
            name = participants[idx].text.strip()
            if name:
                return name
    except Exception:
        pass

    return None


# Mapování zkrácených jmen z Livesport na jména v DB
TEAM_NAME_MAP = {
    # Premier League
    "Manchester Utd":    "Manchester United",
    "Brighton":          "Brighton & Hove Albion",
    "Wolves":            "Wolverhampton Wanderers",
    "Nottingham":        "Nottingham Forest",
    "Bournemouth":       "AFC Bournemouth",
    "Sunderland":        "AFC Sunderland",
    "Newcastle":         "Newcastle United",
    "Tottenham":         "Tottenham Hotspur",
    "West Ham":          "West Ham United",
    "Everton":           "Everton FC",
    "Fulham":            "Fulham FC",
    "Arsenal":           "Arsenal FC",
    "Chelsea":           "Chelsea FC",
    "Liverpool":         "Liverpool FC",
    "Brentford":         "Brentford",
    "Leeds":             "Leeds United",
    "Burnley":           "Burnley FC",
    "Crystal Palace":    "Crystal Palace",
    "Aston Villa":       "Aston Villa",
    "Manchester City":   "Manchester City",
    # Chance Liga — jména jsou typicky plná, ale pro jistotu
    "Sigma Olomouc":     "Sigma Olomouc",
    "Sparta Praha":      "Sparta Praha",
    "Slavia Praha":      "Slavia Praha",
    "Viktoria Plzeň":    "Viktoria Plzeň",
}


def normalize_team(name):
    """Normalizuje jméno týmu podle mapovací tabulky."""
    return TEAM_NAME_MAP.get(name, name)


def parse_date_from_match_text(el):
    """
    Extrahuje datum z textu match elementu.
    Na /program/ je datum přímo v textu elementu: '28.02. 21:00 | Wolves | ...'
    """
    try:
        text = el.text.strip()
        # Formát DD.MM.YYYY
        m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', text)
        if m:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

        # Formát DD.MM. (bez roku)
        m = re.search(r'(\d{1,2})\.(\d{1,2})\.', text)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            today = date.today()
            for year_offset in (0, 1):
                year = today.year + year_offset
                try:
                    candidate = date(year, month, day)
                    if candidate >= today:
                        return candidate
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def scrape_future_fixtures(driver, url, league_code, season_start_month):
    """
    Načte stránku /program/ a vrátí seznam budoucích zápasů jako list dict.

    Na /program/ stránce datum je přímo v textu každého match elementu
    (např. '28.02. 21:00 | Wolves | Aston Villa | PREVIEW'), ne v hlavičce.
    """
    print(f"\n🌐 Načítám: {url}")
    driver.get(url)
    time.sleep(4)
    handle_cookies(driver)
    time.sleep(3)

    for attempt in range(WEEKS_AHEAD):
        try:
            more_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH,
                    "//*[contains(@class, 'event__more') or contains(text(), 'více')]"
                ))
            )
            if more_btn.is_displayed():
                driver.execute_script("arguments[0].click();", more_btn)
                time.sleep(2)
                print(f"   🖱️  Klik 'více' #{attempt+1}")
        except Exception:
            break

    all_elements = driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]")
    print(f"   📝 Nalezeno {len(all_elements)} match elementů")

    today = date.today()
    fixtures = []
    skipped_past = 0
    skipped_no_date = 0
    skipped_no_team = 0

    for el in all_elements:
        try:
            match_id = el.get_attribute("id").replace("g_1_", "")

            # Datum přímo z textu elementu
            match_date = parse_date_from_match_text(el)
            if match_date is None:
                skipped_no_date += 1
                continue

            if match_date <= today:
                skipped_past += 1
                continue

            home_team = normalize_team(parse_team_name(el, "home") or "")
            away_team = normalize_team(parse_team_name(el, "away") or "")

            if not home_team or not away_team:
                skipped_no_team += 1
                continue

            season = determine_season(match_date, season_start_month)

            fixtures.append({
                "livesport_id": match_id,
                "match_date":   match_date,
                "home_team":    home_team,
                "away_team":    away_team,
                "league":       league_code,
                "season":       season,
            })

        except Exception as e:
            print(f"   ⚠️  Chyba: {e}")
            continue

    print(f"   ✅ Budoucích zápasů: {len(fixtures)}")
    if fixtures:
        print(f"   🔍 Prvních 5 zápasů:")
        for f in fixtures[:5]:
            print(f"      {f['match_date']}  {f['home_team']} vs {f['away_team']}  [{f['season']}]")
    if skipped_past:    print(f"   ⏭️  Přeskočeno (minulost): {skipped_past}")
    if skipped_no_date: print(f"   ⚠️  Přeskočeno (bez data): {skipped_no_date}")
    if skipped_no_team: print(f"   ⚠️  Přeskočeno (bez týmu): {skipped_no_team}")
    return fixtures


# =============================================================================
# ULOŽENÍ DO DB
# =============================================================================

def save_fixtures(fixtures, league_code):
    """
    Uloží budoucí zápasy do tabulky fixtures.
    Duplicity: stejný home_team + away_team (bez ohledu na datum) → aktualizuje datum.
    Filtruje neplatné záznamy (prázdná nebo podezřelá jména jako 'Odlož.').
    """
    if not fixtures:
        print(f"   ℹ️  Žádné nové zápasy pro {league_code}")
        return 0

    inserted = 0
    updated = 0
    skipped = 0

    with engine.begin() as conn:
        for f in fixtures:
            # Filtr neplatných jmen týmů
            if not f["home_team"] or not f["away_team"]:
                skipped += 1
                continue
            if any(x in f["home_team"] + f["away_team"]
                   for x in ("Odlož", "TBD", "???", "vs")):
                print(f"   ⚠️  Přeskočen neplatný záznam: {f['home_team']} vs {f['away_team']}")
                skipped += 1
                continue

            # Fuzzy duplicate check: stejné týmy bez ohledu na přesné datum
            # (abseits.at a livesport mohou mít datum o 1 den posunuté)
            existing = conn.execute(text("""
                SELECT id, match_date FROM fixtures
                WHERE home_team = :home AND away_team = :away
                  AND match_date BETWEEN :date_from AND :date_to
                  AND league = :league
            """), {
                "home":      f["home_team"],
                "away":      f["away_team"],
                "league":    f["league"],
                "date_from": str(pd.Timestamp(f["match_date"]) - pd.Timedelta(days=3)),
                "date_to":   str(pd.Timestamp(f["match_date"]) + pd.Timedelta(days=3)),
            }).fetchone()

            if existing:
                # Pokud datum nesedí → aktualizuj na livesport datum (přesnější)
                if str(existing[1])[:10] != str(f["match_date"]):
                    conn.execute(text("""
                        UPDATE fixtures SET match_date = :date WHERE id = :id
                    """), {"date": f["match_date"], "id": existing[0]})
                    updated += 1
                else:
                    skipped += 1
                continue

            # Vlož nový zápas
            conn.execute(text("""
                INSERT INTO fixtures (match_date, home_team, away_team, league, season)
                VALUES (:date, :home, :away, :league, :season)
            """), {
                "date":   f["match_date"],
                "home":   f["home_team"],
                "away":   f["away_team"],
                "league": f["league"],
                "season": f["season"],
            })
            inserted += 1

    print(f"   💾 {league_code}: vloženo {inserted} nových, "
          f"aktualizováno {updated} datumů, přeskočeno {skipped} duplicit")
    return inserted


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🚀 STEP0b: STAHOVÁNÍ BUDOUCÍCH ZÁPASŮ")
    print("=" * 60)

    driver = None
    try:
        driver = make_driver()

        total_inserted = 0
        for league_key, cfg in LEAGUES.items():
            print(f"\n{'='*60}")
            print(f"📅 Liga: {cfg['league_code']} ({league_key})")
            print(f"{'='*60}")

            fixtures = scrape_future_fixtures(
                driver,
                cfg["url"],
                cfg["league_code"],
                cfg["season_start_month"],
            )

            # Debug: ukáž prvních 5 nalezených zápasů
            if fixtures:
                print(f"\n   🔍 Prvních 5 zápasů:")
                for f in fixtures[:5]:
                    print(f"      {f['match_date']}  {f['home_team']} vs {f['away_team']}  [{f['season']}]")

            n = save_fixtures(fixtures, cfg["league_code"])
            total_inserted += n

    except Exception as e:
        print(f"\n❌ Kritická chyba: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if driver:
            driver.quit()
            print("\n🏁 Prohlížeč ukončen")

    print(f"\n{'='*60}")
    print(f"✅ HOTOVO — celkem vloženo {total_inserted} nových zápasů")
    print(f"   Spusť step2 + step3 pro aktualizaci prepared_fixtures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
