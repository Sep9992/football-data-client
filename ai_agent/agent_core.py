"""
agent_core.py
Mozek AI agenta. Tento skript diriguje jednotlivé nástroje (tools),
plánuje kroky a zajišťuje kompletní automatizaci aktualizace lig.
UPGRADE:
  - Integrovaný DatabaseAutoFixer přímo v agentovi (agent.auto_fix_database)
  - Bezpečná správa prohlížečů (close_all_browsers po stahování i v bloku finally)
  - Automatický reset cache modelů po prepare_datasets() a retrain_models()
  - Samooprava starých URL s '#' v _save_future_fixtures_to_db
  - Sjednocené odsazení v hlavním spouštěcím bloku
"""

import os
import sys
import traceback
from datetime import date
from sqlalchemy import text

# Zajištění správné cesty pro importy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from ai_agent.tools.livesport_harvester import LivesportHarvester
from ai_agent.tools.match_statistics_scraper import MatchStatisticsScraper
from ai_agent.tools.model_trainer import ModelTrainer
from ai_agent.tools.prediction_generator import PredictionGenerator
from ai_agent.tools.feature_analyzer import FeatureAnalyzer
from ai_agent.tools.livesport_odds_scraper import LivesportOddsScraper
from ai_agent.tools.dataset_builder import build_datasets
from ai_agent.tools.strategy_backtester import StrategyBacktester
from ai_agent.tools.data_debugger import DataDebugger
from ai_agent.tools.system_debugger import SystemDebugger
from ai_agent.tools.db_auto_fixer import DatabaseAutoFixer


class FootballAgent:
    def __init__(self, headless=True):
        self.headless = headless
        print("🤖 Inicializuji Football AI Agenta...")

        self.fixer = DatabaseAutoFixer()
        self.harvester = LivesportHarvester(headless=self.headless)
        self.scraper = MatchStatisticsScraper(headless=self.headless)
        self.trainer = ModelTrainer()
        self.predictor = PredictionGenerator()
        self.analyzer = FeatureAnalyzer()
        self.livesport_odds_scraper = LivesportOddsScraper(headless=self.headless)
        self.backtester = StrategyBacktester()
        self.data_debugger = DataDebugger()
        self.system_debugger = SystemDebugger()

    def close_all_browsers(self):
        """Bezpečně uzavře všechny otevřené instance prohlížeče Chrome a uvolní RAM."""
        for tool_name in ('harvester', 'scraper', 'livesport_odds_scraper'):
            tool = getattr(self, tool_name, None)
            if tool and hasattr(tool, 'close_browser'):
                try:
                    tool.close_browser()
                except Exception:
                    pass

    def _invalidate_caches(self):
        """Vymaže mezipaměť modelů a backtesteru po přepočtu dat nebo přetrénování."""
        if hasattr(self.predictor, 'models_loaded'):
            self.predictor.models_loaded = False
        if hasattr(self.analyzer, 'models_loaded'):
            self.analyzer.models_loaded = False
        if hasattr(self.backtester, '_cached_main_result'):
            self.backtester._cached_main_result = None
            self.backtester._cached_summer_result = None

    def get_current_season(self, start_month: int) -> str:
        """Dynamicky vypočítá aktuální sezónu (např. '2026/27') podle dnešního data."""
        today = date.today()
        year = today.year
        month = today.month

        if month >= start_month:
            return f"{year}/{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}/{str(year)[-2:]}"

    def auto_fix_database(self):
        """Spustí automatické čištění URL a sloučení případných duplicit v DB."""
        print(f"\n{'=' * 60}")
        print("🧹 AGENT TASK: Samooprava a čištění databáze (Auto-Janitor)")
        print(f"{'=' * 60}")
        try:
            self.fixer.resolve_duplicates()
        except Exception as e:
            print(f"⚠️ Chyba při auto-opravě databáze: {e}")

    def update_all_leagues(self):
        """
        Hlavní spouštěč. Projde všechny ligy definované v konfiguraci,
        sám určí jejich aktuální sezónu a provede kompletní aktualizaci.
        """
        leagues = self.harvester.leagues_config
        print(f"\n🚀 SPUŠTĚNÍ AGENTA: Nalezeno {len(leagues)} lig k aktualizaci.")

        try:
            for league_code, config in leagues.items():
                start_month = config["start_month"]
                current_season = self.get_current_season(start_month)
                self.update_league(league_code, current_season)
        finally:
            # Po stažení všech lig uvolníme prohlížeč harvesteru i scraperu z RAM
            self.harvester.close_browser()
            self.scraper.close_browser()

        print("\n✅ VŠECHNY LIGY BYLY ÚSPĚŠNĚ AKTUALIZOVÁNY.")

    def update_league(self, league_code: str, season: str):
        """Kompletní workflow pro aktualizaci jedné ligy."""
        print(f"\n{'=' * 70}")
        print(f"🎯 AGENT TASK: Aktualizace ligy {league_code} pro sezónu {season}")
        print(f"{'=' * 70}")

        try:
            # --- KROK 1: Získání nových odkazů ---
            print("\n" + "─" * 50)
            print(" 1. ZÍSKÁNÍ ODKAZŮ (Minulé zápasy)")
            print("─" * 50)
            harvest_result = self.harvester.harvest_past_links(league_code)

            if harvest_result["status"] == "error":
                print(f"❌ Chyba harvesteru: {harvest_result.get('message')}")
                return

            links_to_scrape = harvest_result.get("links", [])
            print(f"✅ Nalezeno odkazů na stránce: {len(links_to_scrape)}")

            # --- KROK 2: Stahování detailních statistik ---
            if links_to_scrape:
                # Zavřeme prohlížeč harvesteru před spuštěním scraperu, ať neběží 2× Chrome najednou
                self.harvester.close_browser()
                print("⏳ Stahuji detailní statistiky k nalezeným zápasům (Scraper)...")
                scrape_result = self.scraper.scrape_urls(links_to_scrape, league_code, season)
                print(f"✅ Statistiky vyřešeny: {scrape_result.get('scraped', 0)} uloženo, "
                      f"{scrape_result.get('skipped', 0)} přeskočeno, "
                      f"{scrape_result.get('errors', 0)} chyb.")
            else:
                print("⏭️ Přeskakuji stahování statistik (všechna data jsou aktuální).")

            # --- KROK 3: Aktualizace budoucího programu ---
            print("\n" + "─" * 50)
            print(" 2. ZÍSKÁNÍ PROGRAMU (Budoucí zápasy - horizont 14 dní)")
            print("─" * 50)
            future_result = self.harvester.harvest_future_fixtures(league_code)

            if future_result["status"] == "success":
                fixtures = future_result.get("fixtures", [])
                print(f"⏳ Zpracovávám a porovnávám {len(fixtures)} budoucích zápasů z Livesportu...")
                self._save_future_fixtures_to_db(fixtures, league_code)
            else:
                print(f"❌ Chyba při stahování programu: {future_result.get('message')}")

            print(f"\n🎉 HOTOVO: Liga {league_code} je plně aktuální.")

        except Exception as e:
            print(f"\n💥 Kritická chyba při zpracování ligy {league_code}: {e}")
            traceback.print_exc()

    def _save_future_fixtures_to_db(self, fixtures, league_code):
        """Pomocná metoda pro uložení budoucích zápasů do DB. Doplňuje i chybějící nebo poškozené URL."""
        if not fixtures:
            print("   ℹ️  Žádné budoucí zápasy k uložení.")
            return

        inserted, updated, skipped = 0, 0, 0

        with self.scraper.engine.begin() as conn:
            for f in fixtures:
                existing = conn.execute(text('''
                    SELECT id, match_date, url FROM fixtures
                    WHERE home_team = :home AND away_team = :away
                      AND match_date BETWEEN (DATE(:date) - INTERVAL '3 days') AND (DATE(:date) + INTERVAL '3 days')
                      AND league = :league
                '''), {
                    "home": f["home_team"], "away": f["away_team"],
                    "date": f["match_date"], "league": league_code
                }).fetchone()

                new_url = f.get("url")

                if existing:
                    fixture_id, existing_date, existing_url = existing[0], existing[1], existing[2]

                    needs_update = False
                    update_parts = []
                    params = {"id": fixture_id}

                    # Kontrola, zda se nezměnilo datum
                    if str(existing_date)[:10] != str(f["match_date"])[:10]:
                        update_parts.append("match_date = :date")
                        params["date"] = f["match_date"]
                        needs_update = True

                    # Kontrola, zda nechybí URL nebo zda neobsahuje starý přílepek '#'
                    if new_url and (not existing_url or "#" in str(existing_url)):
                        update_parts.append("url = :url")
                        params["url"] = new_url
                        needs_update = True

                    if needs_update:
                        query = f"UPDATE fixtures SET {', '.join(update_parts)} WHERE id = :id"
                        conn.execute(text(query), params)
                        updated += 1
                    else:
                        skipped += 1
                    continue

                # Zápas neexistuje -> Vlož nový VČETNĚ URL
                conn.execute(text('''
                    INSERT INTO fixtures (match_date, home_team, away_team, league, season, url)
                    VALUES (:date, :home, :away, :league, :season, :url)
                '''), {
                    "date": f["match_date"], "home": f["home_team"],
                    "away": f["away_team"], "league": league_code,
                    "season": f.get("season", ""), "url": new_url
                })
                inserted += 1

        print(f"   💾 Uloženo {inserted} nových, zaktualizováno {updated} záznamů (datum/URL), přeskočeno {skipped} duplicit.")

    def prepare_datasets(self):
        print("\n" + "=" * 60)
        print("⚙️ AGENT TASK: Výpočet formy týmů a Ligové DNA (Feature Engineering)")
        print("=" * 60)
        build_datasets()
        self._invalidate_caches()

    def retrain_models(self):
        """Spustí kompletní trénovací pipeline přes nástroj ModelTrainer."""
        print(f"\n{'=' * 60}")
        print("🧠 AGENT TASK: Přetrénování prediktivních modelů")
        print(f"{'=' * 60}")

        try:
            result = self.trainer.train_all()

            if result.get("status") == "success":
                self._invalidate_caches()
                print("\n🎉 MODELY ÚSPĚŠNĚ PŘETRÉNOVÁNY A ULOŽENY!")
                print(f"   Počet vzorků v DB: {result.get('n_samples')}")
                print(f"   Vybráno features:  {result.get('n_features_sel')}")
                print(f"   Nejlepší model:    {result.get('best_classifier')} (F1={result.get('best_f1_score')})")
                print(f"   Draw threshold:    {result.get('draw_threshold')}")
            else:
                print(f"\n❌ Chyba při tréninku: {result.get('message')}")

        except Exception as e:
            print(f"\n💥 Kritická chyba při tréninku: {e}")
            traceback.print_exc()

    def predict_upcoming_matches(self, days_ahead=7):
        """Vygeneruje a zobrazí predikce pro budoucí zápasy."""
        print(f"\n{'=' * 60}")
        print("🔮 AGENT TASK: Generování finálních predikcí")
        print(f"{'=' * 60}")

        try:
            result = self.predictor.generate_predictions(days_ahead=days_ahead)
            self.predictor.print_console_report(result)
        except Exception as e:
            print(f"\n💥 Kritická chyba při predikci: {e}")
            traceback.print_exc()

    def analyze_features(self, days_ahead=7):
        """Vygeneruje analýzu predikcí a SHAP grafy."""
        print(f"\n{'=' * 60}")
        print("🔬 AGENT TASK: Analýza predikcí (SHAP values)")
        print(f"{'=' * 60}")

        try:
            result = self.analyzer.analyze_upcoming_matches(days_ahead=days_ahead)
            self.analyzer.print_console_report(result)
        except Exception as e:
            print(f"\n💥 Kritická chyba při analýze: {e}")

    def fetch_live_odds(self, days_ahead=7):
        print(f"\n{'=' * 60}")
        print("💰 AGENT TASK: Aktualizace reálných kurzů (Livesport Scraper)")
        print(f"{'=' * 60}")
        try:
            self.livesport_odds_scraper.scrape_odds(days_ahead=days_ahead)
        except Exception as e:
            print(f"\n💥 Kritická chyba při aktualizaci kurzů: {e}")

    def evaluate_strategy(self):
        """Provede kompletní walk-forward testování aktuální strategie na historii."""
        print(f"\n{'=' * 60}")
        print("📊 AGENT TASK: Otestování profitabilní strategie (Backtest)")
        print(f"{'=' * 60}")

        try:
            # Hlavní produkční backtest (září - duben) + v 1. průchodu spočítá i léto do cache
            main_result = self.backtester.run_backtest()
            self.backtester.print_report(main_result, label="HLAVNÍ STRATEGIE (Září - Duben)")

            # Diagnostický backtest (pouze měsíce 5, 6, 7, 8 - okamžitě načteno z cache)
            print("\n" + "-" * 60)
            print("ℹ️  DIAGNOSTIKA: Sledování letních měsíců (Květen - Srpen)")
            print("-" * 60)
            diag_result = self.backtester.run_backtest(diagnostic_months=[5, 6, 7, 8])
            self.backtester.print_report(diag_result, label="LETNÍ MĚSÍCE")

        except Exception as e:
            print(f"\n💥 Kritická chyba při backtestingu: {e}")

    def debug_data(self):
        """Spustí kontrolu duplicit a integrity dat."""
        print(f"\n{'=' * 60}")
        print("🔍 AGENT TASK: Diagnostika databáze")
        print(f"{'=' * 60}")

        tables_to_check = ["fixtures", "prepared_datasets", "prepared_fixtures"]

        for table in tables_to_check:
            try:
                res = self.data_debugger.check_table(table)
                if res and "table" in res:
                    self.data_debugger.print_report(res)
                else:
                    print(f"\n======================================================================")
                    print(f"📋 TABULKA: {table}")
                    print(f"======================================================================")
                    print("  ℹ️ Tabulka je aktuálně prázdná nebo neobsahuje dostatek dat k diagnostice.\n")
            except Exception as e:
                print(f"💥 Chyba při diagnostice tabulky {table}: {e}")

    def debug_system(self):
        """Spustí komplexní test funkčnosti modelů a spojení."""
        try:
            self.system_debugger.run_diagnostics()
        except Exception as e:
            print(f"💥 Chyba při systémové diagnostice: {e}")


if __name__ == "__main__":
    agent = FootballAgent(headless=False)

    try:
        # --- NULTÁ FÁZE: SAMOOPRAVA DATABÁZE ---
        # agent.auto_fix_database()

        # --- FÁZE 1: SBĚR DAT A FEATURE ENGINEERING ---
        # 1. Stáhne odehrané výsledky, detailní statistiky a budoucí zápasy
        # agent.update_all_leagues()

        # 2. Vyčistí případné duplicity po stahování a stáhne čerstvé kurzy
        # agent.auto_fix_database()
        agent.fetch_live_odds()

        # 3. Přepočítá formu týmů, obranné metriky a Ligovou DNA
        # agent.prepare_datasets()

        # --- FÁZE 2: UČENÍ AI ---
        # 4. Přetrénuje modely na nejnovějším datasetu
        # agent.retrain_models()

        # --- FÁZE 3: DIAGNOSTIKA A KONTROLA ---
        # 5. Zkontroluje integritu tabulek a připravenost modelů
        # agent.debug_data()
        # agent.debug_system()

        # 6. Otestuje ROI a výkonnost strategie po jednotlivých ligách
        # agent.evaluate_strategy()

        # 7. Analýza vlivu faktorů na nadcházející zápasy (SHAP)
        # agent.analyze_features()

        # --- FÁZE 4: FINÁLNÍ VÝSTUP (Peníze) ---
        # 8. Výpočet a zobrazení finálních predikcí (Value Singles + AKO)
        # agent.predict_upcoming_matches()

    finally:
        print("\n[ÚKLID] Zavírám všechny prohlížeče...")
        agent.close_all_browsers()