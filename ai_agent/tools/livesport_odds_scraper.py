"""
livesport_odds_scraper.py
Stahuje vždy NEJAKTUÁLNĚJŠÍ kurzy přes undetected_chromedriver z Livesportu.
UPGRADE (LINE SHOPPING):
  - Načte všechny sázkové kanceláře na stránce (Tipsport, Fortuna, Betano, Chance)
  - Pro každý jednotlivý trh (1, X, 2, 1X, 12, X2, Over 2.5, Under 2.5, BTTS) vybere NEJVYŠŠÍ kurz
  - Uloží nejvyšší kurz + přesný název sázkovky (book_1, book_x, book_2, ...) do DB
"""

import os
import time
import random
import re
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Potlačení chyby OSError: [WinError 6] Neplatný popisovač na Windows
uc.Chrome.__del__ = lambda self: None


class LivesportOddsScraper:
    KNOWN_BOOKMAKERS = [
        ("tipsport", "Tipsport"),
        ("fortuna", "Fortuna"),
        ("ifortuna", "Fortuna"),
        ("betano", "Betano"),
        ("chance", "Chance"),
        ("sazkabet", "Sazkabet"),
        ("kingsbet", "Kingsbet"),
        ("merkur", "MerkurXtip"),
        ("synottip", "SynotTip"),
    ]

    def __init__(self, headless=False):
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")
        self.engine = create_engine(db_url)
        self.headless = headless
        self.driver = None
        self._db_cols_cache = None

    def _get_bookmaker_odds_columns(self) -> set:
        if self._db_cols_cache is not None:
            return self._db_cols_cache
        with self.engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'bookmaker_odds'"
            )).fetchall()
        self._db_cols_cache = {r[0] for r in rows}
        return self._db_cols_cache

    def _get_chrome_version(self) -> int:
        """Automaticky zjistí hlavní verzi nainstalovaného Chrome (např. 155)."""
        env_ver = os.getenv("CHROME_VERSION")
        if env_ver and env_ver.isdigit():
            return int(env_ver)
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon")
            version_str, _ = winreg.QueryValueEx(key, "version")
            winreg.CloseKey(key)
            return int(version_str.split('.')[0])
        except Exception:
            return 155

    def start_browser(self):
        if self.driver is not None:
            return
        options = uc.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-first-run')
        options.add_argument('--password-store=basic')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')

        chrome_ver = self._get_chrome_version()
        print(f"  🌐 Spouštím Chrome (verze {chrome_ver})...")
        self.driver = uc.Chrome(options=options, use_subprocess=True, version_main=chrome_ver)
        self.driver.__class__.__del__ = lambda self: None

    def close_browser(self):
        if self.driver:
            try:
                self.driver.__class__.__del__ = lambda self: None
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def _extract_match_id(self, url_or_id):
        if not url_or_id:
            return ""
        if "mid=" in str(url_or_id):
            return str(url_or_id).split("mid=")[1].split("&")[0].split("#")[0][:8]
        clean = str(url_or_id).split('#')[0].split('?')[0]
        parts = [p for p in clean.split('/') if p]
        for part in reversed(parts):
            if len(part) == 8 and part.isalnum():
                return part
        return str(url_or_id)[:8]

    def _resolve_full_base_url(self, raw_url: str, match_id: str, fid: int) -> str:
        """
        Zajistí plnou adresu ve tvaru https://www.livesport.cz/zapas/fotbal/tym1/tym2/
        a uloží ji do tabulky fixtures.
        """
        clean_url = str(raw_url).split('#')[0].split('?')[0].rstrip('/') + '/'
        if not clean_url.startswith('http'):
            clean_url = f"https://www.livesport.cz{clean_url}"

        if "/fotbal/" in clean_url:
            return clean_url

        short_url = f"https://www.livesport.cz/zapas/{match_id}/"
        self.driver.get(short_url)
        time.sleep(2.0)

        try:
            cookie_btn = WebDriverWait(self.driver, 2).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_btn.click()
        except Exception:
            pass

        current = self.driver.current_url
        resolved_base = current.split('#')[0].split('?')[0].rstrip('/') + '/'

        if "/fotbal/" in resolved_base:
            full_canonical_url = f"{resolved_base}?mid={match_id}"
            try:
                with self.engine.begin() as conn:
                    conn.execute(
                        text("UPDATE fixtures SET url = :u WHERE id = :fid"),
                        {"u": full_canonical_url, "fid": fid}
                    )
            except Exception:
                pass

        return resolved_base

    def _detect_bookmaker_in_row(self, row_el) -> str:
        """Rozpozná název sázkové kanceláře (Tipsport, Fortuna, Betano, Chance) z HTML řádku."""
        attributes_to_check = []

        # 1. Prohledáme obrázky loga (alt, title) a odkazy (title, href) uvnitř řádku
        for tag in row_el.find_all(['img', 'a', 'span', 'div']):
            for attr in ('title', 'alt', 'href', 'aria-label'):
                val = tag.get(attr)
                if val:
                    attributes_to_check.append(str(val).lower())

        attributes_to_check.append(row_el.get_text(separator=' ').lower())
        combined_haystack = " ".join(attributes_to_check)

        for key, clean_name in self.KNOWN_BOOKMAKERS:
            if key in combined_haystack:
                return clean_name

        return ""

    def _parse_best_odds(self, page_source, expected_count, is_over_under=False):
        """
        Projde VŠECHNY řádky sázkových kanceláří na stránce a pro každý sloupec
        vrátí dvojici (nejvyšší_kurz, název_sázkovky).
        """
        soup = BeautifulSoup(page_source, 'html.parser')
        if not soup.body:
            return [(None, None)] * expected_count

        rows = soup.find_all('div', class_=lambda c: c and 'ui-table__row' in c)
        if not rows:
            rows = soup.find_all(
                ['div', 'a'],
                class_=lambda c: c and ('row' in c.lower() or 'bookmaker' in c.lower())
            )

        collected_rows = []

        for row in rows:
            if row.find('div', class_=lambda c: c and 'ui-table__row' in c):
                continue

            text_content = row.get_text(separator=' ').strip()

            if is_over_under:
                if '2.5' not in text_content and '2,5' not in text_content:
                    continue
                if ('1.5' in text_content or '1,5' in text_content) and ('3.5' in text_content or '3,5' in text_content):
                    continue

            matches = re.findall(r'\b\d{1,3}[.,]\d{2}\b', text_content)
            if len(matches) < expected_count:
                continue

            vals = [float(m.replace(',', '.')) for m in matches]
            if is_over_under:
                vals = [v for v in vals if v not in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5) and 1.12 <= v <= 50.0]
            else:
                vals = [v for v in vals if 1.01 < v <= 100.0]

            if len(vals) >= expected_count:
                bookie = self._detect_bookmaker_in_row(row) or "Livesport"
                collected_rows.append((bookie, vals[:expected_count]))

        if not collected_rows:
            return [(None, None)] * expected_count

        named_rows = [r for r in collected_rows if r[0] != "Livesport"]
        active_rows = named_rows if named_rows else collected_rows

        best_results = []
        for i in range(expected_count):
            max_odd = -1.0
            best_bookies = []

            for bookie, odds_list in active_rows:
                val = odds_list[i]
                if val > max_odd:
                    max_odd = val
                    best_bookies = [bookie]
                elif val == max_odd and bookie not in best_bookies:
                    best_bookies.append(bookie)

            if max_odd > 1.0:
                bookie_str = "/".join(best_bookies[:2])
                best_results.append((max_odd, bookie_str))
            else:
                best_results.append((None, None))

        return best_results

    def scrape_odds(self, days_ahead=7):
        days_ahead = min(days_ahead, 7)
        print(f"\n🌍 Stahuji NEJVYŠŠÍ kurzy (Tipsport, Fortuna, Betano, Chance) pro dalších {days_ahead} dnů...")

        with self.engine.begin() as conn:
            query = text(f"""
                SELECT f.id, f.url, f.home_team, f.away_team, f.league, f.match_date
                FROM fixtures f
                WHERE f.match_date >= CURRENT_DATE
                  AND f.match_date <= CURRENT_DATE + INTERVAL '{days_ahead} days'
                  AND f.url IS NOT NULL
                ORDER BY f.match_date ASC
            """)
            df_upcoming = pd.read_sql(query, conn)

        if df_upcoming.empty:
            print("  ℹ️ Žádné nadcházející zápasy v tomto období nenalezeny.")
            return {"status": "success", "matched": 0, "updated": 0, "count": 0, "new_odds": 0, "inserted": 0, "added": 0}

        print(f"  🔍 Nalezeno {len(df_upcoming)} zápasů pro kompletní aktualizaci kurzů.")
        db_cols = self._get_bookmaker_odds_columns()
        self.start_browser()
        saved_count = 0

        try:
            for idx, row in df_upcoming.iterrows():
                try:
                    fid = row['id']
                    raw_url = str(row['url'])
                    match_id = self._extract_match_id(raw_url)
                    print(f"    ⏳ Stahuji kurzy [{idx+1}/{len(df_upcoming)}]: {row.get('home_team')} vs {row.get('away_team')} ({match_id})")

                    full_base = self._resolve_full_base_url(raw_url, match_id, fid)

                    url_1x2 = f"{full_base}kurzy/kurzy-1x2/zakladni-doba/?mid={match_id}"
                    url_dc  = f"{full_base}kurzy/dvojita-sance/zakladni-doba/?mid={match_id}"
                    url_ou  = f"{full_base}kurzy/over-under/zakladni-doba/?mid={match_id}"
                    url_bts = f"{full_base}kurzy/oba-tymy-daji-gol/zakladni-doba/?mid={match_id}"

                    # 1. Kurzy 1X2
                    self.driver.get(url_1x2)
                    try:
                        cookie_btn = WebDriverWait(self.driver, 2).until(
                            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                        )
                        cookie_btn.click()
                    except Exception:
                        pass
                    time.sleep(2.5)

                    res_1x2 = self._parse_best_odds(self.driver.page_source, expected_count=3)
                    if not res_1x2 or res_1x2[0][0] is None:
                        print("      ❌ Kurzy nejsou aktuálně sázkovkami vypsány.")
                        continue

                    # 2. Dvojitá šance (1X, 12, X2)
                    self.driver.get(url_dc)
                    time.sleep(1.5)
                    res_dc = self._parse_best_odds(self.driver.page_source, expected_count=3)

                    # 3. Over / Under 2.5 gólu
                    self.driver.get(url_ou)
                    time.sleep(1.5)
                    res_ou = self._parse_best_odds(self.driver.page_source, expected_count=2, is_over_under=True)

                    # 4. Oba týmy dají gól (BTTS)
                    self.driver.get(url_bts)
                    time.sleep(1.5)
                    res_bts = self._parse_best_odds(self.driver.page_source, expected_count=2, is_over_under=False)

                    (odd_1, book_1), (odd_x, book_x), (odd_2, book_2) = res_1x2
                    (odd_1x, book_1x), (odd_12, book_12), (odd_x2, book_x2) = res_dc
                    (odd_o25, book_o25), (odd_u25, book_u25) = res_ou
                    (odd_bts_y, book_bts_y), (odd_bts_n, book_bts_n) = res_bts

                    if odd_1 is not None and odd_2 is not None:
                        record_data = {
                            "fixture_id": fid,
                            "home_team": row.get('home_team'),
                            "away_team": row.get('away_team'),
                            "league": row.get('league'),
                            "match_date": row.get('match_date'),
                            "odd_1": odd_1, "odd_x": odd_x, "odd_2": odd_2,
                            "odd_1x": odd_1x, "odd_12": odd_12, "odd_x2": odd_x2,
                            "odd_over_25": odd_o25, "odd_under_25": odd_u25,
                            "odd_btts_yes": odd_bts_y, "odd_btts_no": odd_bts_n,
                            "book_1": book_1 or "Livesport",
                            "book_x": book_x,
                            "book_2": book_2,
                            "book_1x": book_1x,
                            "book_12": book_12,
                            "book_x2": book_x2,
                            "book_over_25": book_o25,
                            "book_under_25": book_u25,
                            "book_btts_yes": book_bts_y,
                            "book_btts_no": book_bts_n,
                        }

                        valid_cols = [c for c in record_data.keys() if c in db_cols]
                        cols_sql = ", ".join(valid_cols + ["fetched_at"])
                        vals_sql = ", ".join([f":{c}" for c in valid_cols] + ["NOW()"])

                        with self.engine.begin() as conn:
                            conn.execute(text("DELETE FROM bookmaker_odds WHERE fixture_id = :fid"), {"fid": fid})
                            conn.execute(
                                text(f"INSERT INTO bookmaker_odds ({cols_sql}) VALUES ({vals_sql})"),
                                {k: record_data[k] for k in valid_cols}
                            )

                        saved_count += 1
                        saved_markets = [f"1X2 ({odd_1} {book_1} / {odd_x} {book_x} / {odd_2} {book_2})"]
                        if odd_1x:
                            saved_markets.append(f"DC ({odd_1x} {book_1x} / {odd_x2} {book_x2})")
                        if odd_o25:
                            saved_markets.append(f"O/U2.5 ({odd_o25} {book_o25} / {odd_u25} {book_u25})")
                        if odd_bts_y:
                            saved_markets.append(f"BTTS ({odd_bts_y} {book_bts_y} / {odd_bts_n} {book_bts_n})")
                        print(f"      ✅ MAX kurzy: {' | '.join(saved_markets)}")

                    time.sleep(random.uniform(1.0, 2.0))

                except Exception as e:
                    print(f"      ⚠️ Připojení přerušeno ({e}). Dávám 5s pauzu...")
                    time.sleep(5)
                    continue
        finally:
            self.close_browser()

        print(f"  ✅ Hotovo. Aktualizováno {saved_count} zápasů v databázi kurzů.")
        return {
            "status": "success",
            "matched": saved_count, "updated": saved_count, "count": saved_count,
            "new_odds": saved_count, "inserted": saved_count, "added": saved_count
        }