import os
import re
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURACE A DB PŘIPOJENÍ
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ Nenalezeno DATABASE_URL v .env souboru!")

engine = create_engine(DATABASE_URL)

LEAGUE_CODE = "L1"
SEASON = "2026/27"
RESULTS_URL = "https://www.livesport.cz/fotbal/francie/ligue-1/vysledky/"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINICE STATISTIK
# ─────────────────────────────────────────────────────────────────────────────
STATS_TO_FIND = [
    "Očekávané góly (xG)", "xGOT", "Držení míče", "Střely celkem", "Střely na branku",
    "Střely mimo branku", "Zblokované střely", "Střely uvnitř vápna", "Střely mimo vápno",
    "Střely do tyče/břevna", "Velké šance", "Rohové kopy", "Doteky ve vápně soupeře",
    "Přesné průnikové přihrávky", "Ofsajdy", "Přímé kopy", "Přihrávky", "Dlouhé přihrávky",
    "Přihrávky v útočné třetině", "Centry", "Očekávané asistence (xA)", "Vhazování",
    "Fauly", "Žluté karty", "Červené karty", "Obranné zákroky", "Vyhrané souboje",
    "Obranné odkopy", "Přerušení přihrávek", "Chyby vedoucí ke střele",
    "Chyby vedoucí ke gólu", "Brankářské zákroky", "xGOT proti", "Zabráněné góly",
    "Zmařené velké šance", "Získané míče", "Ztráty míče",
    "Kličky", "Úspěšné kličky"
]

STATS_DB_MAP = {
    "Očekávané góly (xG)": "expected_goals", "xGOT": "xgot", "Držení míče": "possession",
    "Střely celkem": "shots", "Střely na branku": "shots_on_target", "Střely mimo branku": "shots_off_target",
    "Zblokované střely": "blocked_shots", "Střely uvnitř vápna": "shots_inside_box",
    "Střely mimo vápno": "shots_outside_box",
    "Střely do tyče/břevna": "woodwork", "Velké šance": "big_chances", "Rohové kopy": "corners",
    "Doteky ve vápně soupeře": "box_touches", "Přesné průnikové přihrávky": "through_balls",
    "Ofsajdy": "offsides", "Přímé kopy": "free_kicks", "Očekávané asistence (xA)": "expected_assists",
    "Vhazování": "throw_ins", "Fauly": "fouls", "Žluté karty": "yellow_cards",
    "Červené karty": "red_cards", "Vyhrané souboje": "duels_won", "Obranné odkopy": "clearances",
    "Přerušení přihrávek": "interceptions", "Brankářské zákroky": "saves", "Zabráněné góly": "prevented_goals",
    "Přihrávky": "passes", "Dlouhé přihrávky": "long_balls", "Přihrávky v útočné třetině": "passes_final_third",
    "Centry": "crosses", "Obranné zákroky": "tackles", "Chyby vedoucí ke střele": "errors_shot",
    "Chyby vedoucí ke gólu": "errors_goal", "Zmařené velké šance": "big_chances_missed",
    "Získané míče": "ball_recoveries", "Ztráty míče": "possession_lost",
    "Kličky": "dribbles_total", "Úspěšné kličky": "dribbles_completed"
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. POMOCNÉ FUNKCE
# ─────────────────────────────────────────────────────────────────────────────
def get_db_columns():
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = 'match_statistics'")).fetchall()
    return {row[0] for row in rows}


def parse_complex_stat(value_str):
    if not value_str: return None, None, None
    match = re.search(r'(\d+)%\s*\((\d+)/(\d+)\)', value_str)
    if match: return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match_frac = re.search(r'\((\d+)/(\d+)\)', value_str)
    if match_frac: return None, int(match_frac.group(1)), int(match_frac.group(2))
    match_pct = re.search(r'(\d+)\s*%', value_str)
    if match_pct: return int(match_pct.group(1)), None, None
    clean_str = value_str.replace(',', '.')
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', clean_str)
    if numbers: return None, float(numbers[0]), None
    return None, None, None


def close_popups(driver):
    try:
        driver.execute_script("document.getElementById('onetrust-accept-btn-handler')?.click();")
        driver.execute_script(
            "document.querySelectorAll('button').forEach(b => { if(b.innerText.toLowerCase().includes('rozumím')) b.click(); });")
    except:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. HLAVNÍ LOGIKA - HARVESTER A SCRAPER
# ─────────────────────────────────────────────────────────────────────────────
def run_recovery():
    print("=" * 70)
    print("🚀 START: STAHOVÁNÍ ZÁPASŮ A STATISTIK PRO L1 2026/27")
    print("=" * 70)

    db_columns = get_db_columns()

    options = uc.ChromeOptions()
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')
    driver = uc.Chrome(options=options, use_subprocess=True)

    try:
        # FÁZE 1: Nalezení všech zápasů na stránce s výsledky
        print(f"🌍 Otevírám hlavní stránku ligy: {RESULTS_URL}")
        driver.get(RESULTS_URL)
        time.sleep(3)
        close_popups(driver)

        # ---------------------------------------------------------
        # RUČNÍ ROLOVÁNÍ
        # ---------------------------------------------------------
        print("\n" + "=" * 60)
        print("🛑 SKRIPT POZASTAVEN - ČEKÁM NA VÁS!")
        print("1. Jděte do otevřeného okna prohlížeče Chrome.")
        print("2. Ručně klikejte na 'Zobrazit další zápasy', dokud nedojedete k 1. kolu.")
        print("3. Až budete úplně dole, vraťte se sem do terminálu a stiskněte ENTER.")
        print("=" * 60 + "\n")

        input("👉 Jsem na úplném konci, stiskněte ENTER pro pokračování zpracování... ")

        print("✅ Pokračuji ve čtení rozbalené stránky...")
        time.sleep(2)
        # ---------------------------------------------------------

        # Sezbíráme všechny ID zápasů na stránce
        match_elements = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
        matches_to_scrape = []

        for el in match_elements:
            match_id = el.get_attribute("id")
            if match_id and match_id.startswith("g_1_"):
                livesport_id = match_id.replace("g_1_", "")
                url = f"https://www.livesport.cz/zapas/{livesport_id}/#/prehled-zapasu/statistiky-zapasu/0"
                matches_to_scrape.append({"id": livesport_id, "url": url})

        print(f"📦 Celkem nalezeno {len(matches_to_scrape)} zápasů k uložení.")

        # FÁZE 2: Průchod jednotlivými zápasy a ukládání Fixtures + Statistics
        for index, match in enumerate(matches_to_scrape, 1):
            url = match['url']
            ls_id = match['id']
            print(f"\n[{index}/{len(matches_to_scrape)}] 🌐 Hledám zápas ID: {ls_id}")

            driver.get(url)
            time.sleep(3)
            close_popups(driver)

            home_team, away_team, goals_home, goals_away = None, None, None, None

            # Extrakce metadat zápasu
            try:
                home_team = driver.find_element(By.CSS_SELECTOR,
                                                ".duelParticipant__home .participant__participantName").text.strip()
                away_team = driver.find_element(By.CSS_SELECTOR,
                                                ".duelParticipant__away .participant__participantName").text.strip()

                score_text = driver.find_element(By.CLASS_NAME, "detailScore__wrapper").text.strip()
                scores = re.findall(r'\d+', score_text)
                if len(scores) >= 2:
                    goals_home, goals_away = int(scores[0]), int(scores[1])
            except Exception as e:
                print(f"   ⚠️ Nepodařilo se načíst hlavičku zápasu: {e}")
                continue

            print(f"   ℹ️ {home_team} vs {away_team} (Skóre: {goals_home}:{goals_away})")

            # --- Vložení do tabulky Fixtures s INT ID ---
            try:
                with engine.begin() as conn:
                    # Zkusíme zápas najít podle unikátní URL
                    check_query = "SELECT id FROM fixtures WHERE url = :url LIMIT 1"
                    existing_match = conn.execute(text(check_query), {"url": url}).fetchone()

                    if existing_match:
                        db_fixture_id = existing_match[0]
                        update_query = """
                            UPDATE fixtures 
                            SET home_team = :home_team, away_team = :away_team 
                            WHERE id = :id
                        """
                        conn.execute(text(update_query), {
                            "home_team": home_team, "away_team": away_team, "id": db_fixture_id
                        })
                    else:
                        # Vložení nového zápasu (DB si vygeneruje číselné ID přes RETURNING)
                        insert_query = """
                            INSERT INTO fixtures (match_date, home_team, away_team, league, season, url)
                            VALUES (CURRENT_DATE, :home_team, :away_team, :league, :season, :url)
                            RETURNING id;
                        """
                        db_fixture_id = conn.execute(text(insert_query), {
                            "home_team": home_team,
                            "away_team": away_team,
                            "league": LEAGUE_CODE,
                            "season": SEASON,
                            "url": url
                        }).fetchone()[0]

            except Exception as e:
                print(f"   ❌ Chyba při ukládání fixture: {e}")
                continue

            # Extrakce statistik
            stats_opened = driver.execute_script("""
                let tabs = Array.from(document.querySelectorAll('a, div, span')).filter(el => el.innerText && el.innerText.trim().toUpperCase() === 'STATISTIKY');
                if(tabs.length > 0) { tabs[0].click(); return true; }
                return false;
            """)

            if not stats_opened:
                print("   ❌ Zápas nemá záložku Statistiky.")
                continue

            time.sleep(2)
            extracted_data = {}
            for stat_name in STATS_TO_FIND:
                try:
                    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{stat_name}')]")
                    for el in elements:
                        if not el.is_displayed(): continue

                        parent = el.find_element(By.XPATH, "./..")
                        for _ in range(4):
                            lines = [l.strip() for l in parent.text.split('\n') if l.strip()]
                            if len(lines) >= 3 and any(c.isdigit() for c in parent.text):
                                break
                            parent = parent.find_element(By.XPATH, "./..")

                        label_idx = -1
                        for i, line in enumerate(lines):
                            if stat_name in line:
                                label_idx = i
                                break

                        if label_idx != -1:
                            raw_h = " ".join(lines[:label_idx]).strip()
                            label = lines[label_idx]
                            raw_a = " ".join(lines[label_idx + 1:]).strip()

                            res_h = parse_complex_stat(raw_h)
                            res_a = parse_complex_stat(raw_a)

                            if res_h != (None, None, None) or res_a != (None, None, None):
                                extracted_data[stat_name] = (res_h, res_a)
                                break
                except:
                    continue

            if not extracted_data:
                print("   ⚠️ Žádné statistiky na stránce nenalezeny.")
                continue

            # Uložení statistik (Používáme db_fixture_id z databáze)
            params = {
                "fixture_id": db_fixture_id,
                "league": LEAGUE_CODE,
                "season": SEASON,
                "goals_home": goals_home,
                "goals_away": goals_away
            }
            update_parts = ["goals_home = EXCLUDED.goals_home", "goals_away = EXCLUDED.goals_away"]

            for stat_name, values in extracted_data.items():
                base_col = STATS_DB_MAP.get(stat_name)
                if not base_col: continue

                (h_pct, h_val, h_tot), (a_pct, a_val, a_tot) = values
                COMPLEX = {"passes", "long_balls", "passes_final_third", "crosses", "tackles"}

                if base_col in COMPLEX:
                    for suffix, val in [("_accuracy_home", h_pct), ("_completed_home", h_val), ("_total_home", h_tot),
                                        ("_accuracy_away", a_pct), ("_completed_away", a_val), ("_total_away", a_tot)]:
                        col = f"{base_col}{suffix}"
                        if val is not None and col in db_columns:
                            params[col] = val
                            update_parts.append(f"{col} = EXCLUDED.{col}")
                else:
                    for suffix, val in [("_home", h_val if h_val is not None else h_pct),
                                        ("_away", a_val if a_val is not None else a_pct)]:
                        col = f"{base_col}{suffix}"
                        if val is not None and col in db_columns:
                            params[col] = val
                            update_parts.append(f"{col} = EXCLUDED.{col}")

            columns = list(params.keys())
            query_str = f"""
                INSERT INTO match_statistics ({', '.join(columns)})
                VALUES ({', '.join(':' + c for c in columns)})
                ON CONFLICT (fixture_id) DO UPDATE SET {', '.join(update_parts)}, created_at = NOW();
            """

            try:
                with engine.begin() as conn:
                    conn.execute(text(query_str), params)
                print(f"   💾 ÚSPĚCH! Zapsáno {len(columns) - 5} metrik do tabulky statistik.")
            except Exception as e:
                print(f"   ❌ CHYBA ZÁPISU STATISTIK DO DB: {e}")

            time.sleep(random.uniform(1.5, 3.0))

    except KeyboardInterrupt:
        print("\n⛔ Přerušeno uživatelem.")
    finally:
        driver.quit()
        print("\n🎉 Zpracování dokončeno.")


if __name__ == "__main__":
    run_recovery()