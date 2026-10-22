"""
livesport_harvester.py
Nástroj AI agenta pro získávání dat ze stránek Livesport.cz.
Spojuje funkčnost stahování odkazů na proběhlé zápasy a budoucí fixtures.
"""

import re
import time
from datetime import datetime, date, timedelta
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import os

# Dynamicky přidá kořenovou složku projektu do systémové cesty
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import našeho centrálního slovníku
from ai_agent.config.team_names import normalize_team


class LivesportHarvester:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None

        # Centrální konfigurace lig (9 evropských soutěží)
        self.leagues_config = {
            "PL": {"url_base": "https://www.livesport.cz/fotbal/anglie/premier-league", "start_month": 8},
            "FL": {"url_base": "https://www.livesport.cz/fotbal/cesko/chance-liga", "start_month": 7},
            "BL": {"url_base": "https://www.livesport.cz/fotbal/nemecko/bundesliga", "start_month": 8},
            "SA": {"url_base": "https://www.livesport.cz/fotbal/italie/serie-a", "start_month": 8},
            "LL": {"url_base": "https://www.livesport.cz/fotbal/spanelsko/laliga", "start_month": 8},
            "L1": {"url_base": "https://www.livesport.cz/fotbal/francie/ligue-1", "start_month": 8},
            "ED": {"url_base": "https://www.livesport.cz/fotbal/nizozemsko/eredivisie", "start_month": 8},
            "LP": {"url_base": "https://www.livesport.cz/fotbal/portugalsko/liga-portugal", "start_month": 8},
            "JL": {"url_base": "https://www.livesport.cz/fotbal/belgie/jupiler-pro-league", "start_month": 8},
        }

    def start_browser(self):
        """Spustí Chrome pro agenta."""
        if self.driver is not None:
            return

        options = uc.ChromeOptions()
        options.add_argument('--no-first-run')
        options.add_argument('--password-store=basic')
        options.add_argument('--disable-blink-features=AutomationControlled')

        if self.headless:
            options.add_argument('--headless')

        try:
            self.driver = uc.Chrome(options=options, use_subprocess=True)
        except Exception as e:
            raise RuntimeError(f"Nepodařilo se spustit Chrome: {e}")

    def close_browser(self):
        """Ukončí prohlížeč a vyčistí paměť."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def _accept_cookies(self):
        """Klikne na souhlas s cookies, pokud vyskočí banner."""
        try:
            accept_btn = WebDriverWait(self.driver, 4).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            self.driver.execute_script("arguments[0].click();", accept_btn)
            time.sleep(1)
        except Exception:
            pass

    def _determine_season(self, match_date: date, start_month: int) -> str:
        """Určí sezónu na základě data zápasu a měsíce, kdy liga začíná."""
        y, m = match_date.year, match_date.month
        if m >= start_month:
            return f"{y}/{str(y + 1)[-2:]}"
        else:
            return f"{y - 1}/{str(y)[-2:]}"

    # =========================================================
    # 1. ZÍSKÁNÍ ODKAZŮ (Minulé zápasy)
    # =========================================================
    def harvest_past_links(self, league_code: str, full_history: bool = False) -> dict:
        """
        Získá odkazy na nedávno odehrané zápasy.
        Pro denní běh agenta (full_history=False) neroluje celou historii, stáhne jen to, co je hned vidět.
        """
        if league_code not in self.leagues_config:
            return {"status": "error", "message": f"Neznámá liga: {league_code}"}

        config = self.leagues_config[league_code]
        url = f"{config['url_base']}/vysledky/"

        self.start_browser()
        self.driver.get(url)
        time.sleep(3)
        self._accept_cookies()
        time.sleep(2)

        if full_history:
            print(f"    ⏳ Roluji plnou historii pro ligu {league_code}...")
            clicks = 0
            for _ in range(30):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)

                clicked = self.driver.execute_script("""
                    var btn = document.querySelector('.event__more');
                    if(btn && btn.style.display !== 'none') {
                        btn.click();
                        return true;
                    }
                    return false;
                """)

                if clicked:
                    clicks += 1
                    time.sleep(2)
                else:
                    break
            print(f"    ✅ Provedeno {clicks} kliknutí. Nyní stahuji všechny zobrazené odkazy.")
        else:
            # Denní rutina - neroluje, vezme jen posledních cca 20-40 viditelných zápasů
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        links = []
        elements = self.driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]")

        for el in elements:
            match_id = el.get_attribute("id").replace("g_1_", "")
            full_link = f"https://www.livesport.cz/zapas/{match_id}/"
            links.append(full_link)

        return {
            "status": "success",
            "league": league_code,
            "count": len(links),
            "links": links
        }

    # =========================================================
    # 2. ZÍSKÁNÍ PROGRAMU (Budoucí zápasy)
    # =========================================================
    def harvest_future_fixtures(self, league_code: str) -> dict:
        """
        Získá program nadcházejících zápasů striktně na max 14 dní dopředu.
        """
        if league_code not in self.leagues_config:
            return {"status": "error", "message": f"Neznámá liga: {league_code}"}

        config = self.leagues_config[league_code]
        url = f"{config['url_base']}/program/"

        self.start_browser()
        self.driver.get(url)
        time.sleep(3)
        self._accept_cookies()
        time.sleep(2)

        fixtures = []
        elements = self.driver.find_elements(By.XPATH, "//div[starts-with(@id, 'g_1_')]")

        today = date.today()
        max_future_date = today + timedelta(days=14)

        for el in elements:
            try:
                match_id = el.get_attribute("id").replace("g_1_", "")
                text = el.text.strip()

                # Extrakce plného URL přímo z odkazu řádku (obsahuje /fotbal/tym1/tym2/?mid=...)
                try:
                    link_el = el.find_element(By.CSS_SELECTOR, "a.eventRowLink, a[href*='/zapas/']")
                    match_url = link_el.get_attribute("href").split('#')[0]
                except Exception:
                    match_url = f"https://www.livesport.cz/zapas/{match_id}/"

                # 1. Extrakce data
                match_date = None
                date_match = re.search(r'(\d{1,2})\.(\d{1,2})\.', text)
                if date_match:
                    day, month = int(date_match.group(1)), int(date_match.group(2))
                    year_match = re.search(r'\d{1,2}\.\d{1,2}\.(\d{4})', text)
                    if year_match:
                        year = int(year_match.group(1))
                    else:
                        # Ošetření přelomu roku (prosinec -> leden)
                        candidate = date(today.year, month, day)
                        if (candidate - today).days < -180:
                            year = today.year + 1
                        elif (candidate - today).days > 180:
                            year = today.year - 1
                        else:
                            year = today.year
                    match_date = date(year, month, day)
                elif re.search(r'\b\d{1,2}:\d{2}\b', text):
                    # Pokud u zápasu není datum, ale jen čas (např. "20:30"), hraje se dnes
                    match_date = today

                # FILTR: Přeskočíme historické zápasy A zápasy vzdálenější než 14 dní
                if not match_date or match_date < today or match_date > max_future_date:
                    continue

                # 2. Extrakce jmen týmů (primárně přes přesné DOM selektory, fallback přes vyčištěný text)
                raw_home, raw_away = None, None
                try:
                    home_el = el.find_element(By.CSS_SELECTOR, ".event__participant--home, [class*='homeParticipant']")
                    away_el = el.find_element(By.CSS_SELECTOR, ".event__participant--away, [class*='awayParticipant']")
                    raw_home = home_el.text.strip()
                    raw_away = away_el.text.strip()
                except Exception:
                    parts = [p.strip() for p in re.split(r'\||\n', text) if p.strip()]
                    ignore_tokens = {'PREVIEW', 'FT', '-', 'AET', 'PEN', 'ODLOŽENO', 'ZRUŠENO', 'FRO', 'PŘE'}
                    team_parts = [
                        p for p in parts
                        if not re.match(r'^\d{1,2}\.\d{1,2}', p)
                        and not re.match(r'^\d{1,2}:\d{2}$', p)
                        and p.upper() not in ignore_tokens
                    ]
                    if len(team_parts) >= 2:
                        raw_home, raw_away = team_parts[0], team_parts[1]

                if raw_home and raw_away:
                    home_team = normalize_team(raw_home)
                    away_team = normalize_team(raw_away)
                    season = self._determine_season(match_date, config["start_month"])

                    fixtures.append({
                        "livesport_id": match_id,
                        "match_date": match_date.isoformat(),
                        "home_team": home_team,
                        "away_team": away_team,
                        "league": league_code,
                        "season": season,
                        "url": match_url
                    })
            except Exception:
                continue

        return {
            "status": "success",
            "league": league_code,
            "count": len(fixtures),
            "fixtures": fixtures
        }