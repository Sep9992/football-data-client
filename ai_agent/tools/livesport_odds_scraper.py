"""
livesport_odds_scraper.py
Stahuje vždy NEJAKTUÁLNĚJŠÍ kurzy přes undetected_chromedriver z Livesportu.
OPRAVA:
  - Automatický překlad krátkých URL (/zapas/ID/) na plné SEO URL (/zapas/fotbal/tym1/tym2/)
  - Uložení plné URL zpět do tabulky fixtures pro okamžité příští načtení
  - Obnovena původní spolehlivá metoda _parse_odds
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
    def __init__(self, headless=False):
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")
        self.engine = create_engine(db_url)
        self.headless = headless
        self.driver = None

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
        Zajistí, že máme plnou adresu ve tvaru:
        https://www.livesport.cz/zapas/fotbal/tym1-ID1/tym2-ID2/
        Pokud je v DB jen krátká adresa (/zapas/ID/), otevře ji, nechá Chrome přesměrovat
        a plnou adresu uloží zpět do DB.
        """
        clean_url = str(raw_url).split('#')[0].split('?')[0].rstrip('/') + '/'
        if not clean_url.startswith('http'):
            clean_url = f"https://www.livesport.cz{clean_url}"

        # Pokud už URL obsahuje '/fotbal/' (plný tvar s týmy), můžeme ji rovnou použít
        if "/fotbal/" in clean_url:
            return clean_url

        # Jinak otevřeme krátkou URL a necháme Livesport přesměrovat na plnou adresu
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

        # Uložíme plnou URL do tabulky fixtures, abychom příště nemuseli přesměrovávat
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

    def _parse_odds(self, page_source, expected_count, is_over_under=False):
        """Původní osvědčený parser s ochranou proti sebrání hranice 0.5/1.5 místo 2.5."""
        soup = BeautifulSoup(page_source, 'html.parser')
        if not soup.body:
            return []

        rows = soup.find_all(
            ['div', 'a'],
            class_=lambda c: c and ('row' in c.lower() or 'cell' in c.lower() or 'odds' in c.lower() or 'bookmaker' in c.lower())
        )
        if not rows:
            rows = [soup.body]

        for row in rows:
            text_content = row.get_text(separator=' ')
            if is_over_under:
                if '2.5' not in text_content and '2,5' not in text_content:
                    continue
                # Přeskočíme obří nadřazený kontejner celé stránky, který obsahuje i 0.5, 1.5, 3.5
                if ('1.5' in text_content or '1,5' in text_content) and ('3.5' in text_content or '3,5' in text_content):
                    continue

            matches = re.findall(r'\b\d{1,3}[.,]\d{2}\b', text_content)
            if len(matches) >= expected_count:
                vals = [float(m.replace(',', '.')) for m in matches]
                if is_over_under:
                    vals = [v for v in vals if v not in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)]
                    if len(vals) >= 2:
                        return vals[:2]
                else:
                    if all(v > 1.0 for v in vals[:expected_count]):
                        return vals[:expected_count]
        return []

    def scrape_odds(self, days_ahead=7):
        days_ahead = min(days_ahead, 7)
        print(f"\n🌍 Stahuji NEJAKTUÁLNĚJŠÍ kurzy z Livesportu pro dalších {days_ahead} dnů...")

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
        self.start_browser()
        saved_count = 0

        try:
            for idx, row in df_upcoming.iterrows():
                try:
                    fid = row['id']
                    raw_url = str(row['url'])
                    match_id = self._extract_match_id(raw_url)
                    print(f"    ⏳ Stahuji kurzy [{idx+1}/{len(df_upcoming)}]: {row.get('home_team')} vs {row.get('away_team')} ({match_id})")

                    # Získáme plnou adresu včetně '/fotbal/tym1/tym2/'
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

                    o_1x2 = self._parse_odds(self.driver.page_source, expected_count=3)
                    if not o_1x2:
                        print("      ❌ Kurzy nejsou aktuálně sázkovkami vypsány.")
                        continue

                    # 2. Dvojitá šance (1X, 12, X2)
                    self.driver.get(url_dc)
                    time.sleep(1.5)
                    o_dc = self._parse_odds(self.driver.page_source, expected_count=3)

                    # 3. Over / Under 2.5 gólu
                    self.driver.get(url_ou)
                    time.sleep(1.5)
                    o_ou = self._parse_odds(self.driver.page_source, expected_count=2, is_over_under=True)

                    # 4. Oba týmy dají gól (BTTS)
                    self.driver.get(url_bts)
                    time.sleep(1.5)
                    o_bts = self._parse_odds(self.driver.page_source, expected_count=2, is_over_under=False)

                    odd_1 = o_1x2[0] if (isinstance(o_1x2, list) and len(o_1x2) >= 3) else None
                    odd_x = o_1x2[1] if (isinstance(o_1x2, list) and len(o_1x2) >= 3) else None
                    odd_2 = o_1x2[2] if (isinstance(o_1x2, list) and len(o_1x2) >= 3) else None

                    odd_1x = o_dc[0] if (isinstance(o_dc, list) and len(o_dc) >= 2) else None
                    odd_12 = o_dc[1] if (isinstance(o_dc, list) and len(o_dc) >= 2) else None
                    odd_x2 = o_dc[2] if (isinstance(o_dc, list) and len(o_dc) >= 3) else (o_dc[1] if (isinstance(o_dc, list) and len(o_dc) == 2) else None)

                    odd_o25 = o_ou[0] if (isinstance(o_ou, list) and len(o_ou) >= 2) else None
                    odd_u25 = o_ou[1] if (isinstance(o_ou, list) and len(o_ou) >= 2) else None

                    odd_bts_y = o_bts[0] if (isinstance(o_bts, list) and len(o_bts) >= 2) else None
                    odd_bts_n = o_bts[1] if (isinstance(o_bts, list) and len(o_bts) >= 2) else None

                    if odd_1 is not None and odd_2 is not None:
                        with self.engine.begin() as conn:
                            conn.execute(text("DELETE FROM bookmaker_odds WHERE fixture_id = :fid"), {"fid": fid})
                            conn.execute(text("""
                                INSERT INTO bookmaker_odds (
                                    fixture_id, home_team, away_team, league, match_date,
                                    odd_1, odd_x, odd_2, odd_1x, odd_12, odd_x2, 
                                    odd_over_25, odd_under_25, odd_btts_yes, odd_btts_no, book_1, fetched_at
                                ) VALUES (
                                    :fid, :home, :away, :league, :mdate,
                                    :o1, :ox, :o2, :o1x, :o12, :ox2, :oo25, :ou25, :obtsy, :obtsn, 'Livesport', NOW()
                                )
                            """), {
                                "fid": fid,
                                "home": row.get('home_team'), "away": row.get('away_team'),
                                "league": row.get('league'), "mdate": row.get('match_date'),
                                "o1": odd_1, "ox": odd_x, "o2": odd_2,
                                "o1x": odd_1x, "o12": odd_12, "ox2": odd_x2,
                                "oo25": odd_o25, "ou25": odd_u25,
                                "obtsy": odd_bts_y, "obtsn": odd_bts_n
                            })
                        saved_count += 1
                        saved_markets = [f"1X2 ({odd_1}/{odd_x}/{odd_2})"]
                        if odd_1x: saved_markets.append(f"DC ({odd_1x}/{odd_12}/{odd_x2})")
                        if odd_o25: saved_markets.append(f"O/U2.5 ({odd_o25}/{odd_u25})")
                        if odd_bts_y: saved_markets.append(f"BTTS ({odd_bts_y}/{odd_bts_n})")
                        print(f"      ✅ Uloženo: {' | '.join(saved_markets)}")

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