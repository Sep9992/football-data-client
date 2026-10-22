"""
match_statistics_scraper.py
Nástroj AI agenta pro stahování detailních statistik z konkrétních URL zápasů.
UPGRADE:
  - Opraven název 'Tyč/břevno' dle aktuálního Livesportu
  - Přidány 'Góly hlavou' (headed_goals) a 'Odkopy od brány' (goal_kicks)
  - Odfiltrovány duplicitní popisky střel ('Na branku', 'Mimo branku', 'Obdržené góly') z 2. sekce auditu
  - Ošetřen WinError 6 při ukončování prohlížeče Chrome na Windows
"""

import os
import re
import time
import random
import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Import centrálního konfigu
from ai_agent.config.team_names import normalize_team


class MatchStatisticsScraper:
    def __init__(self, db_url=None, headless=True):
        self.headless = headless
        self.driver = None

        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")
        self.engine = create_engine(db_url)

        self._db_columns_cache = None

        # Definice komplexních metrik pro ukládání (rozděluje se na _total, _completed, _accuracy)
        self.COMPLEX_STATS = {"passes", "long_balls", "passes_final_third", "crosses", "tackles", "dribbles"}

        # Plně aktualizovaný seznam statistik z Livesportu
        self.STATS_TO_FIND = [
            "Očekávané góly (xG)", "xGOT", "Držení míče", "Střely celkem", "Střely na branku",
            "Střely mimo branku", "Zblokované střely", "Střely uvnitř vápna", "Střely mimo vápno",
            "Tyč/břevno", "Góly hlavou", "Velké šance", "Proměněné velké šance", "Neproměněné velké šance",
            "Rohové kopy", "Doteky ve vápně soupeře", "Přesné průnikové přihrávky", "Ofsajdy",
            "Přímé kopy", "Přihrávky", "Přesné přihrávky", "Přesné dlouhé přihrávky", "Dlouhé přihrávky",
            "Přesné přihrávky v útočné třetině", "Přihrávky v útočné třetině",
            "Přesné centry", "Centry", "Očekávané asistence (xA)", "Přihrávky na střelu",
            "Vhazování", "Fauly", "Žluté karty", "Červené karty", "Úspěšné obranné zákroky",
            "Obranné zákroky", "Vyhrané souboje", "Obranné odkopy", "Zachycené přihrávky",
            "Chyby vedoucí ke střele", "Chyby vedoucí ke gólu", "Brankářské zákroky",
            "xGOT proti", "Zabráněné góly", "Odkopy od brány",
            "Kličky", "Úspěšné kličky", "Ztráty míče"
        ]

        # Duplicitní zkrácené popisky na stránce, které už ukládáme pod hlavním názvem
        # (Střely na branku, Střely mimo branku a samotné skóre zápasu)
        self.IGNORED_DUPLICATES = {
            "Na branku", "Mimo branku", "Obdržené góly"
        }

        # Mapování do DB
        self.STATS_DB_MAP = {
            "Očekávané góly (xG)":               "expected_goals",
            "xGOT":                              "xgot",
            "Držení míče":                       "possession",
            "Střely celkem":                     "shots",
            "Střely na branku":                  "shots_on_target",
            "Střely mimo branku":                "shots_off_target",
            "Zblokované střely":                 "blocked_shots",
            "Střely uvnitř vápna":               "shots_inside_box",
            "Střely mimo vápno":                 "shots_outside_box",
            "Tyč/břevno":                        "woodwork",
            "Góly hlavou":                       "headed_goals",
            "Velké šance":                       "big_chances",
            "Proměněné velké šance":             "big_chances_scored",
            "Neproměněné velké šance":           "big_chances_missed",
            "Rohové kopy":                       "corners",
            "Doteky ve vápně soupeře":           "box_touches",
            "Přesné průnikové přihrávky":        "through_balls",
            "Ofsajdy":                           "offsides",
            "Přímé kopy":                        "free_kicks",
            "Očekávané asistence (xA)":          "expected_assists",
            "Přihrávky na střelu":               "key_passes",
            "Vhazování":                         "throw_ins",
            "Fauly":                             "fouls",
            "Žluté karty":                       "yellow_cards",
            "Červené karty":                     "red_cards",
            "Vyhrané souboje":                   "duels_won",
            "Obranné odkopy":                    "clearances",
            "Zachycené přihrávky":               "interceptions",
            "Brankářské zákroky":                "saves",
            "xGOT proti":                        "xgot_faced",
            "Zabráněné góly":                    "prevented_goals",
            "Odkopy od brány":                   "goal_kicks",
            "Chyby vedoucí ke střele":           "errors_shot",
            "Chyby vedoucí ke gólu":             "errors_goal",
            "Ztráty míče":                       "possession_lost",

            # Komplexní statistiky (total/completed/accuracy)
            "Přihrávky":                         "passes",
            "Přesné přihrávky":                  "passes",
            "Přesné dlouhé přihrávky":           "long_balls",
            "Dlouhé přihrávky":                  "long_balls",
            "Přesné přihrávky v útočné třetině": "passes_final_third",
            "Přihrávky v útočné třetině":        "passes_final_third",
            "Přesné centry":                     "crosses",
            "Centry":                            "crosses",
            "Úspěšné obranné zákroky":           "tackles",
            "Obranné zákroky":                   "tackles",
            "Úspěšné kličky":                    "dribbles",
            "Kličky":                            "dribbles"
        }

    def start_browser(self):
        if self.driver is not None:
            return
        options = uc.ChromeOptions()
        options.add_argument('--no-first-run')
        options.add_argument('--password-store=basic')
        options.add_argument('--disable-blink-features=AutomationControlled')
        if self.headless:
            options.add_argument('--headless')
        self.driver = uc.Chrome(options=options, use_subprocess=True)

    def close_browser(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            try:
                # Zabrání chybě OSError: [WinError 6] Neplatný popisovač při ukončování Pythonu
                self.driver.quit = lambda: None
            except Exception:
                pass
            self.driver = None

    def _get_db_columns(self):
        if self._db_columns_cache is not None:
            return self._db_columns_cache
        query = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'match_statistics'")
        with self.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        self._db_columns_cache = {row[0] for row in rows}
        return self._db_columns_cache

    def scrape_urls(self, urls: list, league: str, season: str) -> dict:
        if not urls:
            return {"status": "success", "message": "Žádná URL k řešení.", "scraped": 0}

        existing_urls = set()
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT f.url FROM fixtures f 
                INNER JOIN match_statistics ms ON f.id = ms.fixture_id
                WHERE f.url IS NOT NULL
            """)).fetchall()
            existing_urls = {row[0] for row in result}

        urls_to_scrape = [u for u in urls if u not in existing_urls]
        skipped_count = len(urls) - len(urls_to_scrape)

        if not urls_to_scrape:
            return {"status": "success", "scraped": 0, "skipped": skipped_count, "errors": 0}

        print("\n======================================================================")
        print("🚀 STEP 2: STAHOVÁNÍ STATISTIK ZÁPASŮ")
        print("======================================================================")
        print(f"  Liga:      {league}")
        print(f"  Sezóna:    {season}")
        print(f"  Celkem URL: {len(urls)}")
        print(f"  Přeskočeno (již staženo): {skipped_count}")
        print(f"  Ke zpracování: {len(urls_to_scrape)}")
        print("======================================================================\n")

        print("🌐 Spouštím Chrome...")
        self.start_browser()
        print("  ✅ Chrome spuštěn\n")

        stats_log = {"ok": 0, "skip": skipped_count, "error": 0}

        try:
            for i, url in enumerate(urls_to_scrape, 1):
                try:
                    self.driver.get(url)
                    time.sleep(random.uniform(3, 5))
                    self._close_popups()

                    print(f"[{i}/{len(urls_to_scrape)}]")
                    print("======================================================================")
                    print(f"🌐 {url}")

                    home, away, match_date, g_home, g_away = self._get_match_metadata()

                    if not home or not match_date:
                        print("    ❌ Nepodařilo se identifikovat zápas.")
                        stats_log["error"] += 1
                        continue

                    fixture_id = self._find_or_create_fixture(home, away, match_date, url, league, season)

                    if not fixture_id:
                        stats_log["skip"] += 1
                        continue

                    with self.engine.connect() as conn:
                        has_stats = conn.execute(
                            text("SELECT 1 FROM match_statistics WHERE fixture_id = :fid"),
                            {"fid": fixture_id}
                        ).fetchone()

                    if has_stats:
                        print("    ⏭️ Přeskakuji: Statistiky už v DB jsou, URL opraveno.")
                        stats_log["skip"] += 1
                        continue

                    if not self._click_stats_tab():
                        print("    ❌ Záložka Statistiky nenalezena.")
                        stats_log["skip"] += 1
                        continue

                    time.sleep(2)  # Čekání na stažení React dat
                    extracted_data = self._extract_stats_from_page()

                    self._save_to_db(fixture_id, extracted_data, g_home, g_away, league, season)
                    stats_log["ok"] += 1

                except Exception as e:
                    print(f"    ❌ Chyba při zpracování {url}: {e}")
                    stats_log["error"] += 1
        finally:
            print("\n🏁 Ukončuji prohlížeč...\n")
            self.close_browser()

        print("======================================================================")
        print("✅ HOTOVO!")
        print("======================================================================")
        print(f"  Liga / Sezóna:   {league} {season}")
        print(f"  Celkem URL:      {len(urls)}")
        print(f"  ✅ Staženo:      {stats_log['ok']}")
        print(f"  ⏭️ Přeskočeno:   {stats_log['skip']}")
        print(f"  ❌ Chyb:         {stats_log['error']}")
        print("======================================================================")

        return {
            "status": "success",
            "scraped": stats_log["ok"],
            "skipped": stats_log["skip"],
            "errors": stats_log["error"]
        }

    # =====================================================================
    # INTERNÍ METODY
    # =====================================================================
    def _close_popups(self):
        try:
            accept_btn = self.driver.find_element(By.ID, "onetrust-accept-btn-handler")
            self.driver.execute_script("arguments[0].click();", accept_btn)
            print("    🍪 Cookies přijaty.")
            time.sleep(0.5)
        except Exception:
            pass

    def _get_match_metadata(self):
        try:
            home_raw = self.driver.find_element(By.XPATH, "//div[contains(@class, 'home')]//div[contains(@class, 'participantName')]").text.strip()
            away_raw = self.driver.find_element(By.XPATH, "//div[contains(@class, 'away')]//div[contains(@class, 'participantName')]").text.strip()
            date_str = self.driver.find_element(By.XPATH, "//div[contains(@class, 'startTime')]").text.strip()
            match_date = datetime.datetime.strptime(date_str.split()[0], "%d.%m.%Y").date()

            goals_home, goals_away = None, None
            try:
                score_text = self.driver.find_element(By.CLASS_NAME, "detailScore__wrapper").text.strip()
                scores = re.findall(r'\d+', score_text)
                if len(scores) >= 2:
                    goals_home = int(scores[0])
                    goals_away = int(scores[1])
                    print(f"    ⚽ Skóre: {goals_home}:{goals_away}")
            except Exception:
                pass

            home_team = normalize_team(home_raw)
            away_team = normalize_team(away_raw)
            print(f"    📋 Zápas: {home_team} vs {away_team} ({match_date})")
            return home_team, away_team, match_date, goals_home, goals_away
        except Exception:
            return None, None, None, None, None

    def _find_or_create_fixture(self, home, away, match_date, url, league, season):
        d_minus = match_date - datetime.timedelta(days=2)
        d_plus = match_date + datetime.timedelta(days=2)

        query_find = text("""
            SELECT id, url FROM fixtures
            WHERE home_team = :h AND away_team = :a
              AND match_date BETWEEN :d1 AND :d2 AND league = :league LIMIT 1
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query_find,
                                  {"h": home, "a": away, "d1": d_minus, "d2": d_plus, "league": league}).fetchone()

        if result:
            fixture_id = result[0]
            existing_url = result[1]
            if not existing_url:
                with self.engine.begin() as conn_upd:
                    conn_upd.execute(text("UPDATE fixtures SET url = :url WHERE id = :id"), {"url": url, "id": fixture_id})
            return fixture_id

        query_insert = text("""
            INSERT INTO fixtures (league, season, match_date, home_team, away_team, url)
            VALUES (:league, :season, :date, :home, :away, :url) RETURNING id
        """)
        with self.engine.begin() as conn:
            result = conn.execute(query_insert,
                                  {"league": league, "season": season, "date": match_date, "home": home, "away": away, "url": url}).fetchone()

        return result[0] if result else None

    def _parse_complex_stat(self, value_str, stat_name=""):
        """Umí přečíst '86% (447/519)', '26 (39%)', obrácené '(57%) 24' i čistá čísla."""
        if not value_str:
            return None, None, None
        s = value_str.strip()

        # 1. Klasický troj-formát: 86% (447/519)
        m = re.search(r'(\d+)%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)', s)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))

        # 2. Poloviční formát (bez %): (447/519)
        m = re.search(r'\(\s*(\d+)\s*/\s*(\d+)\s*\)', s)
        if m:
            return None, int(m.group(1)), int(m.group(2))

        # 3. NOVÉ DOMÁCÍ: 26 (39%) nebo 26(39%) -> Pct, Val, None
        m = re.search(r'(\d+)\s*\(\s*(\d+)%\s*\)', s)
        if m:
            return int(m.group(2)), int(m.group(1)), None

        # 4. NOVÉ HOSTÉ: (57%) 24 nebo (57%)24 -> Pct, Val, None
        m = re.search(r'\(\s*(\d+)%\s*\)\s*(\d+)', s)
        if m:
            return int(m.group(1)), int(m.group(2)), None

        # 5. Čistá procenta: 86%
        m = re.search(r'(\d+)%', s)
        if m:
            return int(m.group(1)), None, None

        # 6. Extrakce čistého čísla
        nums = re.findall(r'-?\d+\.?\d*', s.replace(',', '.'))
        if nums:
            val = float(nums[0])
            if any(x in stat_name.lower() for x in ["přesné", "úspěšn", "proměněn"]):
                return None, val, None  # Jde do _completed
            elif self.STATS_DB_MAP.get(stat_name) in self.COMPLEX_STATS:
                return None, None, val  # Jde do _total
            else:
                return None, val, None  # Normální statistika

        return None, None, None

    def _click_stats_tab(self):
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
                if self.driver.execute_script(script):
                    print("    ✅ Záložka Statistiky aktivována.\n")
                    return True
            except Exception:
                continue
        try:
            script_href = """
                var link = document.querySelector('a[href*="statistiky-zapasu"]');
                if (link) { link.click(); return true; }
                return false;
                """
            if self.driver.execute_script(script_href):
                print("    ✅ Záložka aktivována přes href.\n")
                return True
        except Exception:
            pass
        return False

    def _scan_all_page_stats(self) -> dict:
        """
        Přečte VŠECHNY statistické řádky, které jsou reálně přítomné na stránce Livesportu,
        abychom mohli porovnat: (1) co ukládáme, (2) co ignorujeme a (3) co v zápase chybí.
        """
        page_stats = {}

        js_scan = """
        var results = [];
        var allDivs = document.querySelectorAll('div');
        for (var d of allDivs) {
            if (!d.offsetParent) continue;
            var txt = (d.innerText || '').trim();
            if (!txt || txt.length > 140) continue;
            var lines = txt.split('\\n').map(s => s.trim()).filter(s => s.length > 0);
            if (lines.length >= 3 && lines.length <= 5) {
                results.push(lines);
            }
        }
        return results;
        """
        try:
            raw_blocks = self.driver.execute_script(js_scan) or []
        except Exception:
            raw_blocks = []

        ignored_headers = {
            "ZÁPAS", "1. POLOČAS", "2. POLOČAS", "STATISTIKY", "PŘEHLED", "KURZY",
            "SESTAVY", "H2H", "TABULKA", "ZPRÁVY", "SOUHRN", "TIMELINE"
        }

        def is_stat_value(s):
            return bool(re.match(r'^[-+]?\d+[.,]?\d*%?(?:\s*\(\d+/\d+\)|\s*\(\d+%\))?$|^\(\d+%?\)$', s.strip()))

        for lines in raw_blocks:
            label_candidates = [
                (i, l) for i, l in enumerate(lines)
                if 0 < i < len(lines) - 1
                and not is_stat_value(l)
                and any(c.isalpha() for c in l)
            ]
            if len(label_candidates) == 1:
                idx, label = label_candidates[0]
                if label.upper() in ignored_headers:
                    continue
                left_parts = lines[:idx]
                right_parts = lines[idx + 1:]
                if all(is_stat_value(p) for p in left_parts) and all(is_stat_value(p) for p in right_parts):
                    raw_h = " ".join(left_parts).strip()
                    raw_a = " ".join(right_parts).strip()
                    if label not in page_stats:
                        page_stats[label] = (raw_h, raw_a)

        return page_stats

    def _extract_stats_from_page(self) -> dict:
        """
        Extrahuje statistiky a vypíše přehledný 3-úrovňový audit:
          1. Co stahujeme do DB
          2. Co na stránce je, ale my to neukládáme (bez duplicitních popisků)
          3. Co ukládáme, ale v tomto zápase nebylo
        """
        all_on_page = self._scan_all_page_stats()

        for stat_name in self.STATS_TO_FIND:
            if stat_name in all_on_page:
                continue
            try:
                elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{stat_name}')]")
                for el in elements:
                    if not el.is_displayed():
                        continue
                    parent = el.find_element(By.XPATH, "./..")
                    if not any(c.isdigit() for c in parent.text):
                        parent = parent.find_element(By.XPATH, "./..")
                    lines = [l.strip() for l in parent.text.split('\n') if l.strip()]
                    if 3 <= len(lines) <= 5 and stat_name in lines:
                        idx = lines.index(stat_name)
                        if 0 < idx < len(lines) - 1:
                            raw_h = " ".join(lines[:idx]).strip()
                            raw_a = " ".join(lines[idx + 1:]).strip()
                            all_on_page[stat_name] = (raw_h, raw_a)
                            break
            except Exception:
                continue

        # ── SEKCE 1: STATISTIKY, KTERÉ STAHUJEME A UKLÁDÁME ────────────────────────
        print("    1️⃣  STAHOVANÉ STATISTIKY (Ukládáme do DB):")
        print("    DOMÁCÍ               |           STATISTIKA              |                HOSTÉ | DB SLOUPEC")
        print("    ────────────────────────────────────────────────────────────────────────────────────────────────────")

        extracted_data = {}
        found_stats = set()

        for stat_name in self.STATS_TO_FIND:
            if stat_name in all_on_page:
                raw_h, raw_a = all_on_page[stat_name]
                res_h = self._parse_complex_stat(raw_h, stat_name)
                res_a = self._parse_complex_stat(raw_a, stat_name)
                db_col = self.STATS_DB_MAP.get(stat_name, "⚠️ NENAMAPOVÁNO")

                print(f"    {raw_h:<20} | {stat_name:^33} | {raw_a:>20} | -> {db_col}")
                extracted_data[stat_name] = (res_h, res_a)
                found_stats.add(stat_name)

        # ── SEKCE 2: NA STRÁNCE JSOU, ALE MY JE NEUKLÁDÁME ─────────────────────────
        untracked_on_page = {
            k: v for k, v in all_on_page.items()
            if k not in self.STATS_TO_FIND and k not in self.IGNORED_DUPLICATES
        }
        print("\n    2️⃣  NA STRÁNCE NALEZENO, ALE NEUKLÁDÁME (Ignorované / Nové metriky na webu):")
        if untracked_on_page:
            for u_name, (u_h, u_a) in untracked_on_page.items():
                print(f"       👀 {u_h:<14} | {u_name:^33} | {u_a:>14}")
        else:
            print("       ✅ Žádné neznámé statistiky (stahujeme 100 % unikátních statistik ze stránky).")

        # ── SEKCE 3: UKLÁDÁME JE, ALE V TOMTO ZÁPASE NEJSOU ────────────────────────
        missing_in_match = [s for s in self.STATS_TO_FIND if s not in found_stats]
        print(f"\n    3️⃣  STATISTIKY, KTERÉ UKLÁDÁME, ALE V TOMTO ZÁPASE CHYBÍ ({len(missing_in_match)}):")
        if missing_in_match:
            for idx in range(0, len(missing_in_match), 3):
                chunk = missing_in_match[idx:idx + 3]
                print("       🕳️  " + "  •  ".join(chunk))
        else:
            print("       ✅ Všechny sledované statistiky byly v tomto zápase přítomny!")
        print()

        return extracted_data

    def _save_to_db(self, fixture_id, extracted_data, goals_home, goals_away, league, season):
        if not fixture_id or not extracted_data:
            print("    ⚠️  Žádné statistiky nenalezeny — do DB se nic neukládá!")
            return

        db_columns = self._get_db_columns()
        if not db_columns:
            print("    ❌ Nelze uložit — DB schéma nenačteno")
            return

        params = {"fid": fixture_id, "league": league, "season": season, "gh": goals_home, "ga": goals_away}
        columns, values_placeholders, update_parts = [], [], []
        missing_db_cols = []

        def add_col(col_name, param_name, value):
            if col_name in db_columns:
                if col_name not in columns:
                    columns.append(col_name)
                    values_placeholders.append(f":{param_name}")
                    update_parts.append(f"{col_name} = EXCLUDED.{col_name}")
                params[param_name] = value
                return True
            else:
                if col_name not in missing_db_cols:
                    missing_db_cols.append(col_name)
                return False

        add_col("fixture_id", "fid", fixture_id)
        add_col("league", "league", league)
        add_col("season", "season", season)

        if goals_home is not None:
            add_col("goals_home", "gh", goals_home)
        if goals_away is not None:
            add_col("goals_away", "ga", goals_away)

        for stat_name, values in extracted_data.items():
            base_col = self.STATS_DB_MAP.get(stat_name)
            if not base_col:
                continue

            (h_pct, h_val, h_tot), (a_pct, a_val, a_tot) = values

            if base_col in self.COMPLEX_STATS:
                for suffix, val in [
                    ("_accuracy_home", h_pct), ("_completed_home", h_val), ("_total_home", h_tot),
                    ("_accuracy_away", a_pct), ("_completed_away", a_val), ("_total_away", a_tot)
                ]:
                    if val is not None:
                        col = f"{base_col}{suffix}"
                        add_col(col, col, val)
            else:
                for suffix, val in [
                    ("_home", h_val if h_val is not None else h_pct),
                    ("_away", a_val if a_val is not None else a_pct)
                ]:
                    if val is not None:
                        col = f"{base_col}{suffix}"
                        add_col(col, col, val)

        if not update_parts:
            print("    ⚠️  Žádné sloupce k uložení")
            return

        query = text(f"""
                INSERT INTO match_statistics ({', '.join(columns)})
                VALUES ({', '.join(values_placeholders)})
                ON CONFLICT (fixture_id)
                DO UPDATE SET {', '.join(update_parts)}, created_at = NOW();
            """)

        try:
            with self.engine.begin() as conn:
                conn.execute(query, params)
            saved = len([c for c in columns if c not in ['fixture_id', 'league', 'season']])
            print(f"    💾 Uloženo {saved} sloupců do DB (z {len(extracted_data)} nalezených statistik)")
            if missing_db_cols:
                print(f"    ⚠️  Pozor, tyto sloupce zatím chybí v DB tabulce match_statistics: {', '.join(missing_db_cols)}")
        except Exception as e:
            print(f"    ❌ Chyba uložení do DB: {str(e)[:80]}")