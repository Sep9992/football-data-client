import undetected_chromedriver as uc


# Zde naimportujeme náš budoucí centrální konfig
# from ai_agent.config.team_names import normalize_team

class LivesportHarvester:
    """Nástroj pro AI agenta ke sběru dat z Livesportu."""

    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        # Centrální definice lig namísto rozesetí po skriptech
        self.leagues_config = {
            "PL": {"premier-league": "https://www.livesport.cz/fotbal/anglie/premier-league", "start_month": 8},
            "FL": {"chance-liga": "https://www.livesport.cz/fotbal/cesko/chance-liga", "start_month": 7},
            "BL": {"bundesliga": "https://www.livesport.cz/fotbal/nemecko/bundesliga/vysledky/", "start_month": 8},
            "SA": {"serie-a": "https://www.livesport.cz/fotbal/italie/serie-a/vysledky/", "start_month": 8},
            "LL": {"laliga": "https://www.livesport.cz/fotbal/spanelsko/laliga/vysledky/", "start_month": 8}
        }

    def start_browser(self):
        """Spustí Chrome (na pozadí, pokud je headless=True)."""
        options = uc.ChromeOptions()
        options.add_argument('--no-first-run')
        if self.headless:
            options.add_argument('--headless')

        # Zapouzdříme to do try-except, aby agent dostal smysluplnou chybu, když to spadne
        try:
            self.driver = uc.Chrome(options=options, use_subprocess=True)
        except Exception as e:
            raise RuntimeError(f"Nepodařilo se spustit Chrome: {e}")

    def close_browser(self):
        """Agent musí po sobě umět uklidit."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def _accept_cookies(self):
        """Interní metoda pro odkliknutí cookies (sdílená pro obě akce)."""
        # ... logika z vašeho původního handle_cookies() ...
        pass

    # =========================================================
    # DOVEDNOST 1: Získání odkazů na odehrané zápasy (bývalý step0)
    # =========================================================
    def harvest_past_links(self, league_code, since_date=None):
        """
        Agent zavolá toto, když potřebuje nové linky pro step1.
        NEUKLÁDÁ do TXT souboru, ale VRACÍ list odkazů do paměti agenta.
        """
        config = self.leagues_config.get(league_code)
        if not config:
            return {"error": f"Neznámá liga: {league_code}"}

        url = f"{config['url_base']}/vysledky/"
        self.driver.get(url)
        self._accept_cookies()

        # ... Zde bude vaše automatické klikání na "Zobrazit více" ...
        # Odstraníme manual_check a input().

        extracted_links = []  # Zde agent shromáždí linky z elementů

        # Místo uložení do txt vracíme data přímo agentovi
        return {
            "status": "success",
            "league": league_code,
            "links_found": extracted_links
        }

    # =========================================================
    # DOVEDNOST 2: Získání budoucích fixtures (bývalý step0b)
    # =========================================================
    def harvest_future_fixtures(self, league_code):
        """
        Agent zavolá toto, když potřebuje aktualizovat program zápasů.
        Vrací normalizovaná data (list dictů) pro uložení do DB.
        """
        config = self.leagues_config.get(league_code)
        if not config:
            return {"error": f"Neznámá liga: {league_code}"}

        url = f"{config['url_base']}/program/"
        self.driver.get(url)
        self._accept_cookies()

        # ... Zde bude vaše parsování event__time a team names ...

        fixtures = []
        # Zde využijeme náš nový centrální normalizátor jmen!
        # home_team = normalize_team(raw_name)

        return {
            "status": "success",
            "league": league_code,
            "fixtures_found": fixtures
        }