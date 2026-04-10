import os
import time
import re
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# --- KONFIGURACE ---
# Formát: "klíč": ("url", "cílová-sezóna", season_start_month, "league_code")
#
# season_start_month:
#   7 = Chance Liga (začíná v červenci)
#   8 = ostatní ligy (začínají v srpnu)
#
# league_code: kód ligy v DB — slouží k dotazu na nejnovější datum v DB

LEAGUES = {
    "premier-league": (
        "https://www.livesport.cz/fotbal/anglie/premier-league/vysledky/",
        "2025/26", 8, "PL"
    ),
    "chance-liga": (
        "https://www.livesport.cz/fotbal/cesko/chance-liga/vysledky/",
        "2025/26", 7, "FL"
    ),
    "bundesliga": (
        "https://www.livesport.cz/fotbal/nemecko/bundesliga/vysledky/",
        "2025/26", 8, "BL"
    ),
    "serie-a": (
        "https://www.livesport.cz/fotbal/italie/serie-a/vysledky/",
        "2025/26", 8, "SA"
    ),
    "laliga": (
        "https://www.livesport.cz/fotbal/spanelsko/laliga/vysledky/",
        "2025/26", 8, "LL"
    ),
}

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_last_date_in_db(league_code):
    """
    Vrátí nejnovější datum odehraného zápasu dané ligy z DB.
    Pokud DB není dostupná nebo liga nemá žádné zápasy, vrátí None.

    Příklad: pro PL vrátí datetime(2026, 3, 4) pokud poslední zápas byl 4.3.2026.
    Tento datum pak slouží jako cutoff — do txt souboru se uloží jen
    zápasy NOVĚJŠÍ než toto datum.
    """
    if engine is None:
        print("  ⚠️  DB nedostupná — ukládám všechny zápasy sezóny")
        return None
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MAX(f.match_date)
                FROM fixtures f
                INNER JOIN match_statistics ms ON ms.fixture_id = f.id
                WHERE f.league = :league
            """), {"league": league_code})
            row = result.fetchone()
            if row and row[0]:
                # match_date může být date nebo datetime
                last = row[0]
                if hasattr(last, 'date'):
                    last = last  # datetime → ponechat
                else:
                    last = datetime(last.year, last.month, last.day)
                return last
    except Exception as e:
        print(f"  ⚠️  Chyba při dotazu na DB: {e}")
    return None


def get_chrome_version():
    """
    Přečte verzi nainstalovaného Chrome z Windows registru nebo ze souboru.
    Vrátí hlavní číslo verze (např. 145) nebo None pokud se nepodaří zjistit.
    """
    # Metoda 1: Windows registr
    try:
        import winreg
        for key_path in [
            r"SOFTWARE\Google\Chrome\BLBeacon",
            r"SOFTWARE\Wow6432Node\Google\Chrome\BLBeacon",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
        ]:
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
                version, _ = winreg.QueryValueEx(key, "version")
                major = int(version.split(".")[0])
                print(f"  🔍 Chrome verze z registru: {version} → použiji verzi {major}")
                return major
            except:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                    version, _ = winreg.QueryValueEx(key, "version")
                    major = int(version.split(".")[0])
                    print(f"  🔍 Chrome verze z registru: {version} → použiji verzi {major}")
                    return major
                except:
                    continue
    except ImportError:
        pass

    # Metoda 2: Přímé spuštění chrome --version
    try:
        import subprocess
        for chrome_path in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe".format(
                os.environ.get("USERNAME", "")
            ),
        ]:
            if os.path.exists(chrome_path):
                result = subprocess.run(
                    [chrome_path, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                version_str = result.stdout.strip()  # "Google Chrome 145.0.7632.117"
                numbers = re.findall(r'\d+', version_str)
                if numbers:
                    major = int(numbers[0])
                    print(f"  🔍 Chrome verze ze spuštění: {version_str} → použiji verzi {major}")
                    return major
    except:
        pass

    print("  ⚠️  Nepodařilo se zjistit verzi Chrome — použiji autodetekci")
    return None


def ensure_data_dir():
    """Vytvoří data složku, pokud neexistuje."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Vytvořena složka: {DATA_DIR}")
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
    """
    Hledá tlačítko 'Zobrazit více zápasů'.
    Rozšířené selektory pro archivní sezóny.
    """
    selectors = [
        "//a[contains(text(), 'Zobrazit více zápasů')]",
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'zobrazit více')]",
        "//a[contains(@class, 'event__more')]",
        "//div[contains(@class, 'event__more')]",
        "//*[contains(@class, 'showMore')]",
        "//*[contains(@class, 'show-more')]",
        "//*[contains(text(), 'více zápasů')]",
    ]
    for selector in selectors:
        try:
            elements = driver.find_elements(By.XPATH, selector)
            for elem in elements:
                if elem.is_displayed():
                    return elem
        except:
            continue
    return None


def load_all_matches(driver, max_clicks=300, wait_time=4):
    """
    Opakovaně kliká na 'Zobrazit více zápasů'.

    Klíčové parametry pro archivní sezóny:
    - Po každém kliknutí čeká WAIT_AFTER_CLICK sekund (Livesport potřebuje čas)
    - Pak čeká na tlačítko pomocí WebDriverWait (aktivní čekání až 15s)
    - Pokud tlačítko zmizí, scrolluje dolů a zkouší znovu
    - max_no_new = počet kol bez nových zápasů → konec
    """
    WAIT_AFTER_CLICK  = max(wait_time, 6)   # min 6s po kliknutí
    WAIT_FOR_BUTTON   = 5                   # max čekání na tlačítko (bylo 12s)
    SCROLL_PAUSE      = 2                   # pauza po scrollu (bylo 3s)
    max_no_new        = 2                   # konec po 2 kolech bez nových (bylo 5)

    print("    🔄 Začínám načítat starší zápasy (archivní mód)...")
    print(f"    ⏱️  Čekání po kliknutí: {WAIT_AFTER_CLICK}s | Na tlačítko: {WAIT_FOR_BUTTON}s")

    click_count    = 0
    no_new_rounds  = 0
    last_count     = len(driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]"))

    while click_count < max_clicks:
        # 1. Scrolluj dolů aby se tlačítko zobrazilo
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

        # 2. Aktivně čekej na tlačítko (až WAIT_FOR_BUTTON sekund)
        more_btn = None
        try:
            for selector in [
                "//a[contains(text(), 'Zobrazit více zápasů')]",
                "//a[contains(@class, 'event__more')]",
                "//*[contains(text(), 'více zápasů')]",
            ]:
                try:
                    more_btn = WebDriverWait(driver, WAIT_FOR_BUTTON).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    if more_btn and more_btn.is_displayed():
                        break
                    more_btn = None
                except:
                    continue
        except:
            pass

        # 3. Záloha — zkus hledat bez WebDriverWait
        if not more_btn:
            more_btn = find_show_more_button(driver)

        if not more_btn:
            # Tlačítko není — zkontroluj zda přibyly zápasy
            current_count = len(driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]"))
            if current_count > last_count:
                # Přibyly zápasy ale tlačítko ještě není → počkej a zkus znovu
                print(f"    🔄 Přibylo {current_count - last_count} zápasů, hledám tlačítko...")
                last_count = current_count
                no_new_rounds = 0
                time.sleep(WAIT_AFTER_CLICK)
                continue
            else:
                no_new_rounds += 1
                print(f"    ⏳ Tlačítko nenalezeno, žádné nové zápasy "
                      f"({no_new_rounds}/{max_no_new})")
                if no_new_rounds >= max_no_new:
                    print(f"    ✅ Konec načítání — žádné nové zápasy.")
                    break
                time.sleep(WAIT_FOR_BUTTON)  # 5s mezi pokusy
                continue

        # 4. Klikni
        no_new_rounds = 0
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_btn)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", more_btn)
            click_count += 1
        except Exception as e:
            print(f"    ⚠️  Klik selhal: {e}")
            time.sleep(3)
            continue

        # 5. Čekej na načtení nové dávky
        time.sleep(WAIT_AFTER_CLICK)

        current_count = len(driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]"))
        added = current_count - last_count
        last_count = current_count
        print(f"    🖱️  Klik #{click_count} → +{added} zápasů (celkem: {current_count})")

    final_count = len(driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]"))
    print(f"    ✅ Hotovo. Kliknuto: {click_count}x | Zápasů na stránce: {final_count}")
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


def extract_season_from_date(date_str, season_start_month=8, target_season=None):
    """
    Extrahuje sezónu z datumu.

    KLÍČOVÁ OPRAVA: Pro data BEZ roku (např. "20.07.") se rok určuje tak,
    aby datum spadalo do target_season (pokud je zadáno).

    Příklad problému bez opravy:
      Dnes: 25.02.2026, target_season="2024/25", datum="20.07."
      Stará logika: červenec 2026 > dnes → rok=2025 → sezóna 2025/26 ← ŠPATNĚ
      Nová logika:  zkus rok 2025 → 2025/26 ≠ target → zkus rok 2024 → 2024/25 ✅

    Args:
        date_str: Datum jako "DD.MM.YYYY" nebo "DD.MM."
        season_start_month: Měsíc začátku sezóny (7=Chance Liga, 8=PL)
        target_season: Cílová sezóna "YYYY/YY" — pomůže vybrat správný rok
    """
    now = datetime.now()

    def month_to_season(month, year, ssm):
        """Vrátí sezónu pro daný měsíc/rok."""
        if month >= ssm:
            return f"{year}/{str(year+1)[-2:]}"
        else:
            return f"{year-1}/{str(year)[-2:]}"

    try:
        # Varianta 1: DD.MM.YYYY (úplné datum — rok je jasný)
        match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', date_str)
        if match:
            day, month, year = map(int, match.groups())
            return month_to_season(month, year, season_start_month)

        # Varianta 2: DD.MM. (bez roku — musíme odhadnout rok)
        match = re.search(r'(\d{1,2})\.(\d{1,2})\.', date_str)
        if not match:
            return "Unknown"

        day, month = map(int, match.groups())

        if target_season:
            # Víme jakou sezónu hledáme → zkus roky které by do ní seděly
            # Sezóna "2024/25" může obsahovat roky 2024 a 2025
            season_year = int(target_season.split("/")[0])
            candidate_years = [season_year, season_year + 1, season_year - 1]

            for yr in candidate_years:
                try:
                    datetime(yr, month, day)  # validace datumu
                    if month_to_season(month, yr, season_start_month) == target_season:
                        return target_season  # ✅ našli jsme rok který sedí
                except ValueError:
                    continue

            # Fallback: zkus standardní logiku
            year = now.year
            if datetime(year, month, day) > now:
                year -= 1
        else:
            # Bez target_season: standardní logika (nejbližší datum v minulosti)
            year = now.year
            try:
                if datetime(year, month, day) > now:
                    year -= 1
            except ValueError:
                return "Unknown"

        return month_to_season(month, year, season_start_month)

    except Exception:
        return "Unknown"


def parse_date_to_datetime(date_str, season_start_month=8, target_season=None):
    """
    Převede datumový řetězec z Livesportu na datetime objekt.
    Vrátí None pokud parsování selže.
    Používá stejnou logiku jako extract_season_from_date pro určení roku.
    """
    if not date_str:
        return None
    now = datetime.now()
    try:
        # DD.MM.YYYY
        m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', date_str)
        if m:
            d, mo, y = map(int, m.groups())
            return datetime(y, mo, d)

        # DD.MM. (bez roku) — odhadneme rok
        m = re.search(r'(\d{1,2})\.(\d{1,2})\.', date_str)
        if not m:
            return None
        d, mo = map(int, m.groups())

        if target_season:
            season_year = int(target_season.split("/")[0])
            for yr in [season_year, season_year + 1, season_year - 1]:
                try:
                    dt = datetime(yr, mo, d)
                    season_str = extract_season_from_date(
                        f"{d:02d}.{mo:02d}.{yr}", season_start_month)
                    if season_str == target_season:
                        return dt
                except ValueError:
                    continue

        # Fallback: nejbližší datum v minulosti
        yr = now.year
        try:
            dt = datetime(yr, mo, d)
            if dt > now:
                dt = datetime(yr - 1, mo, d)
            return dt
        except ValueError:
            return None
    except Exception:
        return None


def analyze_matches(driver, target_season=None, season_start_month=8, since_date=None):
    """
    Analyzuje načtené zápasy a filtruje pouze cílovou sezónu.

    Args:
        target_season:      Pokud zadáno (např. "2025/26"), uloží jen zápasy této sezóny.
        season_start_month: Měsíc začátku sezóny (7=Chance Liga, 8=PL).
        since_date:         datetime — uloží jen zápasy NOVĚJŠÍ než toto datum.
                            Pokud None, ukládá všechny zápasy cílové sezóny.
    """
    print("    🕵️  Analyzuji načtené zápasy...")
    if since_date:
        print(f"    📅 Cutoff z DB: zachovám pouze zápasy po {since_date.strftime('%d.%m.%Y')}")
    elif target_season:
        print(f"    🎯 Filtr sezóny: zachovám pouze {target_season}")

    all_elements = driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]")
    print(f"    📝 Nalezeno {len(all_elements)} zápasů celkem")

    stats = {
        'total': len(all_elements),
        'by_season': {},
        'links': [],            # pouze filtrované linky (cílová sezóna)
        'links_all': [],        # všechny linky (pro debug)
        'date_extraction_success': 0,
        'sample_dates': []
    }

    for idx, el in enumerate(all_elements):
        try:
            match_id = el.get_attribute("id").replace("g_1_", "")
            full_link = f"https://www.livesport.cz/zapas/{match_id}/"
            stats['links_all'].append(full_link)

            season = "Unknown"
            date_found = None

            # Pokus 1: event__time
            try:
                date_el = el.find_element(By.CLASS_NAME, "event__time")
                date_str = date_el.text.strip()
                if date_str and re.search(r'\d{1,2}\.\d{1,2}\.', date_str):
                    date_found = date_str
                    season = extract_season_from_date(date_str, season_start_month,
                                                        target_season)
                    if season != "Unknown":
                        stats['date_extraction_success'] += 1
            except:
                pass

            # Pokus 2: jiné elementy
            if season == "Unknown":
                try:
                    text_elements = el.find_elements(By.XPATH, ".//*[contains(@class, 'event__')]")
                    for te in text_elements:
                        text = te.text.strip()
                        if re.search(r'\d{1,2}\.\d{1,2}\.', text):
                            date_found = text
                            season = extract_season_from_date(text, season_start_month,
                                                              target_season)
                            if season != "Unknown":
                                stats['date_extraction_success'] += 1
                                break
                except:
                    pass

            stats['by_season'][season] = stats['by_season'].get(season, 0) + 1

            # Rozhodnutí zda zápas uložit:
            # 1. Musí být ve správné sezóně (nebo sezóna není filtrovaná)
            season_ok = (target_season is None or season == target_season)

            # 2. Musí být novější než cutoff z DB (pokud je zadán)
            date_ok = True
            match_datetime = None
            if since_date and date_found:
                match_datetime = parse_date_to_datetime(date_found, season_start_month,
                                                        target_season)
                if match_datetime is not None:
                    date_ok = match_datetime > since_date
                # Pokud datum nelze parsovat → ponecháme (raději stáhnout navíc)

            kept = season_ok and date_ok
            if kept:
                stats['links'].append(full_link)

            # Debug vzorky (první 3 + poslední 3)
            if idx < 3 or idx >= len(all_elements) - 3:
                stats['sample_dates'].append({
                    'index': idx,
                    'date': date_found,
                    'season': season,
                    'kept': kept
                })

        except Exception as e:
            print(f"    ⚠️  Chyba při zpracování zápasu {idx}: {e}")

    print(f"    ✅ Datum extrahováno u {stats['date_extraction_success']}/{stats['total']} zápasů")
    filtered = len(stats['links'])
    skipped = stats['total'] - filtered
    if since_date:
        print(f"    ✅ Nové zápasy po {since_date.strftime('%d.%m.%Y')}: {filtered} uloženo, {skipped} přeskočeno")
    elif target_season:
        print(f"    ✅ Filtrováno ({target_season}): {filtered} zachováno, {skipped} přeskočeno")

    if stats['sample_dates']:
        print(f"\n    🔍 DEBUG - Vzorky (první 3 + poslední 3):")
        for s in stats['sample_dates']:
            status = "✅" if s['kept'] else "❌ SKIP"
            print(f"       [{s['index']:3d}] '{s['date']}' → {s['season']}  {status}")

    return stats


def harvest_links(league_key="premier-league", manual_check=True, wait_time=4):
    """
    Hlavní funkce pro sběr odkazů.
    Automaticky zjistí nejnovější datum v DB a stáhne jen nové zápasy.
    """
    ensure_data_dir()

    if league_key not in LEAGUES:
        print(f"❌ Neznámá liga: {league_key}")
        print(f"   Dostupné klíče: {list(LEAGUES.keys())}")
        return

    url, target_season, season_start_month, league_code = LEAGUES[league_key]
    output_file = os.path.join(DATA_DIR, f"links_{league_key}.txt")
    current_season = get_current_season()

    # --- Zjisti cutoff datum z DB ---
    since_date = get_last_date_in_db(league_code)

    print(f"\n{'=' * 60}")
    print(f"🚀 SBĚR ODKAZŮ - {league_key.upper()} ({league_code})")
    print(f"   Cílová sezóna:   {target_season}")
    print(f"   Aktuální sezóna: {current_season}")
    if since_date:
        print(f"   📅 Poslední zápas v DB: {since_date.strftime('%d.%m.%Y')}")
        print(f"   🎯 Stahuji pouze zápasy PO tomto datu")
    else:
        print(f"   📅 DB prázdná → stahuji celou sezónu")
    print(f"{'=' * 60}")
    print(f"📍 URL: {url}")
    print(f"💾 Výstup: {output_file}\n")

    options = uc.ChromeOptions()
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')
    options.add_argument('--disable-blink-features=AutomationControlled')

    chrome_version = get_chrome_version()
    driver = None

    try:
        chrome_kwargs = {"options": options, "use_subprocess": True}
        if chrome_version:
            chrome_kwargs["version_main"] = chrome_version
        driver = uc.Chrome(**chrome_kwargs)
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
            print("⏸️  PAUZA — ZKONTROLUJ PROHLÍŽEČ")
            print("!" * 60)
            if since_date:
                cutoff_str = since_date.strftime('%d.%m.%Y')
                print(f"")
                print(f"  Poslední zápas v DB:  {cutoff_str}")
                print(f"  Hledáme zápasy:       novější než {cutoff_str}")
                print(f"")
                print(f"  CO UDĚLAT:")
                print(f"  1. Přepni se do okna prohlížeče")
                print(f"  2. Zkontroluj že vidíš zápasy těsně po {cutoff_str}")
                print(f"  3. Pokud ne — scrolluj dolů a klikej 'Zobrazit více zápasů'")
                print(f"  4. Vrať se sem a stiskni ENTER")
            else:
                season_year_start = int(target_season.split("/")[0])
                oldest_month_name = "červenci" if season_start_month == 7 else "srpnu"
                print(f"")
                print(f"  DB prázdná — hledáme celou sezónu {target_season}")
                print(f"  Scrolluj dolů dokud neuvidíš zápasy z {oldest_month_name} {season_year_start}")
                print(f"  Pak stiskni ENTER")
            print("!" * 60)
            input("\n  👉 Stiskni ENTER až je vše připraveno...\n")
            print("!" * 60 + "\n")

        stats = analyze_matches(driver, target_season=target_season,
                                season_start_month=season_start_month,
                                since_date=since_date)

        # Uložení do souboru
        with open(output_file, "w", encoding="utf-8") as f:
            for link in stats['links']:
                f.write(link + "\n")

        print(f"\n{'=' * 60}")
        print(f"✅ HOTOVO!")
        print(f"{'=' * 60}")
        print(f"📊 Celkem načteno na stránce: {stats['total']} zápasů")
        print(f"✅ Nových zápasů uloženo:     {len(stats['links'])}")
        print(f"💾 Soubor: {output_file}")

        if stats['by_season']:
            print(f"\n📅 Zápasy po sezónách (všechny načtené):")
            sorted_seasons = sorted(
                [s for s in stats['by_season'].keys() if s != "Unknown"],
                reverse=True
            )
            if "Unknown" in stats['by_season']:
                sorted_seasons.append("Unknown")
            max_count = max(stats['by_season'].values()) if stats['by_season'] else 1
            for season in sorted_seasons:
                count = stats['by_season'][season]
                bar = "█" * int(count / max_count * 50)
                marker = " ← CÍLOVÁ SEZÓNA" if season == target_season else ""
                print(f"   {season:10s}: {count:3d} {bar}{marker}")

        print(f"\n🖱️  Automatických kliknutí: {clicks}")
        print(f"{'=' * 60}\n")

        saved = len(stats['links'])
        if saved == 0:
            print(f"ℹ️  Žádné nové zápasy — DB je aktuální pro {league_code}.")
        elif saved < 5:
            print(f"✅ {saved} nový/nové zápas(y) — víkendové kolo.")

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

    return len(stats['links']) if 'stats' in dir() else 0


if __name__ == "__main__":
    """
    Spustí sběr nových odkazů pro VŠECH 5 lig.
    Pro každou ligu automaticky zjistí poslední datum v DB
    a stáhne pouze zápasy novější než toto datum.

    Použití:
      python step0_harvest_links.py           ← všech 5 lig
      python step0_harvest_links.py premier-league  ← jen PL (pro testování)
    """
    import sys

    # Volitelný argument: konkrétní liga pro testování
    if len(sys.argv) > 1:
        single_key = sys.argv[1]
        print(f"▶️  Spouštím jen ligu: {single_key}")
        harvest_links(single_key, manual_check=True, wait_time=4)
    else:
        # Všech 5 lig — každá v samostatném Chrome okně
        print("=" * 60)
        print("🚀 HROMADNÝ SBĚR — všech 5 lig")
        print("=" * 60)
        results = {}
        for key in LEAGUES:
            count = harvest_links(key, manual_check=True, wait_time=4)
            results[key] = count

        print("\n" + "=" * 60)
        print("📊 SOUHRN")
        print("=" * 60)
        for key, count in results.items():
            league_code = LEAGUES[key][3]
            print(f"  {league_code:3s}  {key:20s}: {count} nových zápasů")
        print("=" * 60)
        print("\n💡 Další krok: spusť step1 s --skip-existing pro každou ligu s novými linky.")