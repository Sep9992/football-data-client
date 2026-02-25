import os
import time
import re
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- KONFIGURACE ---
LEAGUES = {
    "premier-league": "https://www.livesport.cz/fotbal/anglie/premier-league-2023-2024/#/I3O5jpB2/tabulka/celkem/"
    # "premier-league": "https://www.livesport.cz/fotbal/anglie/premier-league/vysledky/",
    # Další ligy:
    # "la-liga": "https://www.livesport.cz/fotbal/spanelsko/laliga/vysledky/",
    # "bundesliga": "https://www.livesport.cz/fotbal/nemecko/bundesliga/vysledky/",
}

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def ensure_data_dir():
    """Vytvoří data složku, pokud neexistuje."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Vytvořena složka: {DATA_DIR}")


def handle_cookies(driver):
    """Automaticky akceptuje cookies banner."""
    try:
        accept_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        accept_btn.click()
        print("    🍪 Cookies akceptovány")
        time.sleep(1)
    except:
        print("    ℹ️  Cookies banner se neobjevil")


def find_show_more_button(driver):
    """Hledá tlačítko 'Zobrazit více zápasů'."""
    selectors = [
        "//a[contains(text(), 'Zobrazit více zápasů')]",
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'zobrazit')]",
        "//a[contains(@class, 'event__more')]",
    ]

    for selector in selectors:
        try:
            elements = driver.find_elements(By.XPATH, selector)
            for elem in elements:
                if elem.is_displayed() and 'zobrazit' in elem.text.lower():
                    return elem
        except:
            continue

    return None


def load_all_matches(driver, max_clicks=100, wait_time=4):
    """Opakovaně kliká na 'Zobrazit více zápasů'."""
    print("    🔄 Začínám načítat starší zápasy...")
    print(f"    ⏱️  Čekací doba mezi kliky: {wait_time}s")
    click_count = 0
    consecutive_failures = 0
    max_failures = 3

    while click_count < max_clicks:
        try:
            time.sleep(1)
            more_btn = find_show_more_button(driver)

            if more_btn is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"    ✅ Tlačítko nenalezeno po {max_failures} pokusech. Konec.")
                    break
                print(f"    ⏳ Tlačítko nenalezeno, čekám... (pokus {consecutive_failures}/{max_failures})")
                time.sleep(2)
                continue

            consecutive_failures = 0
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_btn)
            time.sleep(0.5)

            print(f"    🖱️  Click #{click_count + 1}")
            driver.execute_script("arguments[0].click();", more_btn)
            click_count += 1

            if click_count % 5 == 0:
                print(f"    📊 Načteno {click_count} dávek...")

            time.sleep(wait_time)

        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break
            time.sleep(2)

    print(f"    ✅ Načítání dokončeno. Celkem kliknuto: {click_count}x")
    return click_count


def get_current_season():
    """
    Určí aktuální fotbalovou sezónu.
    Sezóna začíná v srpnu a končí v červenci.

    Returns:
        str: Sezóna ve formátu "YYYY/YY" (např. "2025/26")
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # Pokud jsme v měsících srpen-prosinec, sezóna začala tento rok
    if month >= 8:
        season_start = year
        season_end = year + 1
    else:
        # Pokud jsme v měsících leden-červenec, sezóna začala minulý rok
        season_start = year - 1
        season_end = year

    return f"{season_start}/{str(season_end)[-2:]}"


def extract_season_from_date(date_str):
    """
    Extrahuje sezónu z datumu.

    KLÍČOVÁ ZMĚNA: Pro dáta BEZ roku (např. "08.08.") vždy použijeme
    logiku, že hledáme nejbližší možný rok v minulosti vzhledem k aktuálnímu datu.

    Args:
        date_str: Datum jako "DD.MM.YYYY" nebo "DD.MM."

    Returns:
        str: Sezóna "YYYY/YY" nebo "Unknown"
    """
    now = datetime.now()

    try:
        # Varianta 1: DD.MM.YYYY (úplné datum)
        match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', date_str)
        if match:
            day, month, year = map(int, match.groups())
        else:
            # Varianta 2: DD.MM. (bez roku)
            match = re.search(r'(\d{1,2})\.(\d{1,2})\.', date_str)
            if match:
                day, month = map(int, match.groups())

                # OPRAVENÁ LOGIKA:
                # Určíme rok tak, že zápas byl nejpozději včera
                # Např. dnes je 16.2.2026:
                # - Datum "15.02." → únor 2026 (nedávno)
                # - Datum "08.08." → srpen 2025 (minulý rok, ne budoucnost!)

                year = now.year

                # Pokud by datum bylo v budoucnosti, musí být z minulého roku
                test_date = datetime(year, month, day)
                if test_date > now:
                    year -= 1
            else:
                return "Unknown"

        if year is None or month is None:
            return "Unknown"

        # Určení sezóny podle měsíce
        if month >= 8:  # Srpen-Prosinec
            season_start = year
            season_end = year + 1
        else:  # Leden-Červenec
            season_start = year - 1
            season_end = year

        return f"{season_start}/{str(season_end)[-2:]}"

    except Exception:
        return "Unknown"


def analyze_matches(driver):
    """
    Analyzuje načtené zápasy.
    """
    print("    🕵️  Analyzuji načtené zápasy...")

    all_elements = driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]")
    print(f"    📝 Nalezeno {len(all_elements)} zápasů")

    stats = {
        'total': len(all_elements),
        'by_season': {},
        'links': [],
        'date_extraction_success': 0,
        'sample_dates': []  # Pro debug
    }

    for idx, el in enumerate(all_elements):
        try:
            match_id = el.get_attribute("id").replace("g_1_", "")
            full_link = f"https://www.livesport.cz/zapas/{match_id}/"
            stats['links'].append(full_link)

            season = "Unknown"
            date_found = None

            # Pokus 1: Najdi datum v event__time
            try:
                date_el = el.find_element(By.CLASS_NAME, "event__time")
                date_str = date_el.text.strip()
                if date_str and re.search(r'\d{1,2}\.\d{1,2}\.', date_str):
                    date_found = date_str
                    season = extract_season_from_date(date_str)
                    if season != "Unknown":
                        stats['date_extraction_success'] += 1
            except:
                pass

            # Pokus 2: Hledej datum jinde
            if season == "Unknown":
                try:
                    text_elements = el.find_elements(By.XPATH, ".//*[contains(@class, 'event__')]")
                    for te in text_elements:
                        text = te.text.strip()
                        if re.search(r'\d{1,2}\.\d{1,2}\.', text):
                            date_found = text
                            season = extract_season_from_date(text)
                            if season != "Unknown":
                                stats['date_extraction_success'] += 1
                                break
                except:
                    pass

            stats['by_season'][season] = stats['by_season'].get(season, 0) + 1

            # Debug vzorky
            if idx < 3 or idx >= len(all_elements) - 3:
                stats['sample_dates'].append({
                    'index': idx,
                    'date': date_found,
                    'season': season
                })

        except Exception as e:
            print(f"    ⚠️  Chyba při zpracování zápasu {idx}: {e}")

    # Debug výpis
    print(f"    ✅ Datum extrahováno u {stats['date_extraction_success']}/{stats['total']} zápasů")
    if stats['sample_dates']:
        print(f"\n    🔍 DEBUG - Vzorky (první 3 + poslední 3):")
        for sample in stats['sample_dates']:
            print(f"       [{sample['index']:3d}] '{sample['date']}' → {sample['season']}")

    return stats


def harvest_links(league_key="premier-league", manual_check=True, wait_time=4):
    """
    Hlavní funkce pro sběr odkazů.
    """
    ensure_data_dir()

    if league_key not in LEAGUES:
        print(f"❌ Neznámá liga: {league_key}")
        return

    url = LEAGUES[league_key]
    output_file = os.path.join(DATA_DIR, f"links_{league_key}.txt")

    current_season = get_current_season()

    print(f"\n{'=' * 60}")
    print(f"🚀 SBĚR ODKAZŮ - {league_key.upper()}")
    print(f"   (Aktuální sezóna: {current_season})")
    print(f"{'=' * 60}")
    print(f"📍 URL: {url}")
    print(f"💾 Výstup: {output_file}\n")

    options = uc.ChromeOptions()
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver = None

    try:
        driver = uc.Chrome(options=options, use_subprocess=True)
        print("✅ Chrome driver inicializován")

    except Exception as e:
        print(f"❌ Chyba při spuštění Chrome: {e}")
        return

    try:
        print(f"🌐 Načítám stránku...")
        driver.get(url)
        time.sleep(4)
        handle_cookies(driver)

        print("    ⏳ Čekám na načtení zápasů...")
        time.sleep(3)

        clicks = load_all_matches(driver, wait_time=wait_time)

        if manual_check:
            print("\n" + "!" * 60)
            print("⏸️  PAUZA PRO KONTROLU")
            print("    Podívej se do prohlížeče.")
            print("    Pro ML model potřebuješ alespoň 3 sezóny (1000+ zápasů).")
            print("    Klikej na 'Zobrazit více zápasů' dokud nevidíš starší sezóny.")
            print(f"    (Automaticky bylo kliknuto {clicks}x)")
            input("    👉 Stiskni ENTER pro pokračování...")
            print("!" * 60 + "\n")

        stats = analyze_matches(driver)

        # Uložení do souboru
        with open(output_file, "w", encoding="utf-8") as f:
            for link in stats['links']:
                f.write(link + "\n")

        # Výpis statistik
        print(f"\n{'=' * 60}")
        print(f"✅ HOTOVO!")
        print(f"{'=' * 60}")
        print(f"📊 Celkem zápasů: {stats['total']}")
        print(f"💾 Uloženo do: {output_file}")

        if stats['by_season']:
            print(f"\n📅 Zápasy po sezónách:")
            sorted_seasons = sorted(stats['by_season'].keys(), reverse=True)

            if "Unknown" in sorted_seasons:
                sorted_seasons.remove("Unknown")
                sorted_seasons.append("Unknown")

            max_count = max(stats['by_season'].values()) if stats['by_season'] else 1
            for season in sorted_seasons:
                count = stats['by_season'][season]
                bar_length = int(count / max_count * 50) if max_count > 0 else 0
                bar = "█" * bar_length
                print(f"   {season:10s}: {count:3d} {bar}")

        print(f"\n🖱️  Automatických kliknutí: {clicks}")
        print(f"{'=' * 60}\n")

        # Doporučení
        if stats['total'] < 1000:
            print(f"💡 DOPORUČENÍ: Máš jen {stats['total']} zápasů.")
            print(f"   Pro dobrý ML model doporučuji alespoň 1000-1500 zápasů (3-4 sezóny).")
            print(f"   Zkus spustit znovu a během pauzy víc klikat.\n")

    except Exception as e:
        print(f"\n❌ CHYBA: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            try:
                driver.quit()
                print("🔒 Prohlížeč uzavřen")
            except Exception:
                pass


if __name__ == "__main__":
    # Základní použití - jedna liga
    # wait_time=3 nebo 4 pokud je Livesport pomalý
    harvest_links("premier-league", manual_check=True, wait_time=4)

    # Pro více lig najednou (odkomentuj):
    # harvest_multiple_leagues(["premier-league", "la-liga"], wait_time=4)