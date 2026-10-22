"""
db_auto_fixer.py
Nástroj AI agenta pro automatické čištění a samoopravu databáze (Auto-Janitor).
UPGRADE:
  - Fáze 1: Očištění poškozených URL (odstranění '#/prehled-zapasu')
  - Fáze 2: Sloučení duplicit se stejným Livesport ID (MID) + záchrana kurzů i statistik
  - Fáze 3: Sloučení duplicit podle týmů a data (±3 dny), i když jednomu záznamu chybí URL
"""

import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


class DatabaseAutoFixer:
    def __init__(self, db_url=None):
        load_dotenv()
        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")
        self.engine = create_engine(db_url)

    def _extract_mid(self, url_or_id):
        """110% bezpečná extrakce Livesport ID. Pokud nenajde přesně 8 alfanumerických znaků, vrací None."""
        if not url_or_id or pd.isna(url_or_id):
            return None

        url_str = str(url_or_id).split('#')[0].split('?')[0]
        raw_str = str(url_or_id)

        # 1. Hledá parametr ?mid= (nový formát Livesportu)
        match = re.search(r'mid=([a-zA-Z0-9]{8})', raw_str)
        if match:
            return match.group(1)

        # 2. Hledá formát /zapas/XXXXXXXX/
        match = re.search(r'/zapas/([a-zA-Z0-9]{8})', url_str)
        if match:
            return match.group(1)

        # 3. Pojistka: Hledá jakýkoliv 8místný alfanumerický blok ohraničený lomítky
        parts = [p for p in url_str.split('/') if p]
        for part in reversed(parts):
            if len(part) == 8 and part.isalnum():
                return part

        return None

    def _merge_and_delete_loser(self, keeper_id: int, loser_id: int, home: str, away: str) -> bool:
        """
        Bezpečně v jedné transakci převede URL, kurzy a statistiky z loser_id na keeper_id
        (pokud je keeper_id ještě nemá) a následně loser_id kompletně vymaže ze všech tabulek.
        """
        try:
            with self.engine.begin() as conn:
                # 0. Záchrana URL v tabulce fixtures (pokud keeper nemá URL a loser ho má)
                conn.execute(text("""
                    UPDATE fixtures f_keep
                    SET url = f_lose.url
                    FROM fixtures f_lose
                    WHERE f_keep.id = :keeper AND f_lose.id = :loser
                      AND (f_keep.url IS NULL OR f_keep.url = '')
                      AND f_lose.url IS NOT NULL
                """), {"keeper": keeper_id, "loser": loser_id})

                # 1. Záchrana kurzů: přesuneme je na keeper_id (pokud je keeper ještě nemá)
                conn.execute(text("""
                    UPDATE bookmaker_odds SET fixture_id = :keeper WHERE fixture_id = :loser 
                    AND NOT EXISTS (SELECT 1 FROM bookmaker_odds WHERE fixture_id = :keeper)
                """), {"keeper": keeper_id, "loser": loser_id})

                # 2. Záchrana statistik: přesuneme je na keeper_id (pokud je keeper ještě nemá)
                conn.execute(text("""
                    UPDATE match_statistics SET fixture_id = :keeper WHERE fixture_id = :loser 
                    AND NOT EXISTS (SELECT 1 FROM match_statistics WHERE fixture_id = :keeper)
                """), {"keeper": keeper_id, "loser": loser_id})

                # 3. Bezpečné smazání zbytků, které patřily k loser_id (přes cizí klíče fixture_id)
                conn.execute(text("DELETE FROM bookmaker_odds WHERE fixture_id = :loser"), {"loser": loser_id})
                conn.execute(text("DELETE FROM match_statistics WHERE fixture_id = :loser"), {"loser": loser_id})

                for optional_table in ("prepared_fixtures", "prepared_datasets"):
                    try:
                        conn.execute(text(f"DELETE FROM {optional_table} WHERE fixture_id = :loser"), {"loser": loser_id})
                    except Exception:
                        pass

                # 4. Jako poslední smažeme samotný duplicitní zápas v tabulce fixtures
                conn.execute(text("DELETE FROM fixtures WHERE id = :loser"), {"loser": loser_id})
            return True
        except Exception as e:
            print(f"  ❌ Chyba při čištění duplicity {home} vs {away} [ID: {loser_id}]: {e}")
            return False

    def resolve_duplicates(self):
        print("\n🛠️ SPUŠTĚN AUTO-JANITOR (110% JISTOTA): Kontrola URL a odstraňování duplicit...")

        # ── KROK 0: Očištění URL od '#/prehled-zapasu' ──────────────────────────────
        try:
            with self.engine.begin() as conn:
                res = conn.execute(text("""
                    UPDATE fixtures
                    SET url = SPLIT_PART(url, '#', 1)
                    WHERE url LIKE '%#%'
                """))
                if res.rowcount > 0:
                    print(f"  🧹 Očištěno {res.rowcount} URL adres od přílepku '#...' v tabulce fixtures.")
        except Exception:
            pass

        # ── NAČTENÍ VŠECH ZÁPASŮ (včetně těch, kterým případně chybí URL) ───────────
        with self.engine.connect() as conn:
            query_all = text("""
                SELECT f.id, f.match_date, f.league, f.home_team, f.away_team, f.url,
                       CASE WHEN s.fixture_id IS NOT NULL THEN 1 ELSE 0 END as has_stats,
                       CASE WHEN f.url IS NOT NULL AND f.url != '' THEN 1 ELSE 0 END as has_url
                FROM fixtures f
                LEFT JOIN match_statistics s ON f.id = s.fixture_id
            """)
            df = pd.read_sql(query_all, conn)

        if df.empty:
            print("  ✅ Databáze je prázdná.")
            return

        resolved_count = 0
        deleted_ids = set()

        # ── FÁZE 1: Sloučení duplicit se stejným Livesport ID (MID) ─────────────────
        df['mid'] = df['url'].apply(self._extract_mid)
        df_with_mid = df.dropna(subset=['mid'])

        for mid, group in df_with_mid.groupby('mid'):
            if len(group) <= 1:
                continue

            home, away = group.iloc[0]['home_team'], group.iloc[0]['away_team']
            # Priorita pro přežití: 1. má statistiky, 2. má URL, 3. vyšší ID
            group = group.sort_values(by=['has_stats', 'has_url', 'id']).reset_index(drop=True)

            keeper_id = int(group.iloc[-1]['id'])
            loser_ids = [int(x) for x in group[group['id'] != keeper_id]['id'].tolist()]

            merged_losers = []
            for loser_id in loser_ids:
                if self._merge_and_delete_loser(keeper_id, loser_id, home, away):
                    deleted_ids.add(loser_id)
                    merged_losers.append(loser_id)
                    resolved_count += 1

            if merged_losers:
                print(f"  🔧 [MID: {mid}] Sloučena duplicita {home} vs {away} (Zachováno ID: {keeper_id}, Smazána ID: {merged_losers})")

        # ── FÁZE 2: Sloučení duplicit podle týmů a data (±3 dny, i bez URL) ─────────
        df_rem = df[~df['id'].isin(deleted_ids)].copy()
        df_rem['match_date'] = pd.to_datetime(df_rem['match_date'])

        for (lg, home, away), group in df_rem.groupby(['league', 'home_team', 'away_team']):
            if len(group) <= 1:
                continue

            group = group.sort_values('match_date').reset_index(drop=True)
            i = 0
            while i < len(group) - 1:
                r1 = group.iloc[i]
                r2 = group.iloc[i + 1]
                gap_days = abs((r2['match_date'] - r1['match_date']).days)

                if gap_days <= 3:
                    # Vybereme lepší záznam jako keeper (má statistiky > má URL > vyšší ID)
                    pair_df = pd.DataFrame([r1, r2]).sort_values(by=['has_stats', 'has_url', 'id'])
                    keeper_id = int(pair_df.iloc[-1]['id'])
                    loser_id = int(pair_df.iloc[0]['id'])

                    if loser_id not in deleted_ids and keeper_id != loser_id:
                        if self._merge_and_delete_loser(keeper_id, loser_id, home, away):
                            deleted_ids.add(loser_id)
                            resolved_count += 1
                            print(f"  🔧 [Datum ±{gap_days}d] Sloučena duplicita {home} vs {away} (Zachováno ID: {keeper_id}, Smazáno ID: {loser_id})")
                i += 1

        if resolved_count > 0:
            print(f"✅ Hotovo. Celkem bezpečně vyřešeno a sloučeno {resolved_count} duplikátů.")
        else:
            print("  ✅ Žádné duplicity nenalezeny. Databáze je 100% čistá.")


if __name__ == "__main__":
    fixer = DatabaseAutoFixer()
    fixer.resolve_duplicates()