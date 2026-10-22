"""
data_debugger.py
Nástroj AI agenta pro kontrolu integrity dat v databázi (hledání duplicit).
UPGRADE:
  - Opravena detekce duplicit po sobě jdoucích zápasů (sousedi v okně <= 14 dnů místo max-min)
  - Sjednoceno hledání napříč sezónami (odhalí duplicity i při jiném formátu zápisu sezóny)
  - Přidán přehled zastoupených lig a kontrola ID pro tabulku fixtures i prepared_*
"""

import pandas as pd
import os
import unicodedata
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


class DataDebugger:
    def __init__(self, db_url=None):
        load_dotenv()
        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")

        self.engine = create_engine(db_url)
        self.dup_window_days = 14

    def _norm(self, s):
        """Odstraní diakritiku a převede na malá písmena pro porovnání."""
        return ''.join(c for c in unicodedata.normalize('NFD', str(s).lower())
                       if unicodedata.category(c) != 'Mn').strip()

    def check_table(self, name: str) -> dict:
        """Provede kontrolu konkrétní tabulky a vrátí report."""
        try:
            df = pd.read_sql(text(f"SELECT * FROM {name}"), self.engine)
        except Exception as e:
            return {"status": "error", "message": f"Nelze načíst tabulku {name}: {e}"}

        if df.empty:
            return {"status": "success", "table": name, "message": f"Tabulka {name} je prázdná."}

        df['match_date'] = pd.to_datetime(df['match_date'])

        leagues_summary = {}
        if 'league' in df.columns:
            leagues_summary = df['league'].value_counts().to_dict()

        result = {
            "status": "success",
            "table": name,
            "rows": len(df),
            "date_min": df['match_date'].min().date(),
            "date_max": df['match_date'].max().date(),
            "leagues": leagues_summary,
            "dupe_names": {},
            "legit_repeats": 0,
            "real_dupes": []
        }

        if 'home_team' not in df.columns:
            return result

        df['_h'] = df['home_team'].map(self._norm)
        df['_a'] = df['away_team'].map(self._norm)

        # 1. Duplicitní názvy týmů (např. Sparta vs Sparta Praha)
        teams = sorted(set(df['home_team'].dropna()) | set(df['away_team'].dropna()))
        groups = {}
        for t in teams:
            groups.setdefault(self._norm(t), []).append(t)

        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        if dupes:
            for k, variants in dupes.items():
                det = []
                for v in variants:
                    count = ((df['home_team'] == v) | (df['away_team'] == v)).sum()
                    det.append(f"'{v}' ({count}x)")
                result["dupe_names"][k] = det

        # 2. Skutečné duplicity zápasů v krátkém čase (kontrola sousedních dat v okně <= 14 dnů)
        for keys, g in df.groupby(['_h', '_a']):
            if len(g) < 2:
                continue
            g = g.sort_values('match_date').reset_index(drop=True)

            for i in range(len(g) - 1):
                m1 = g.iloc[i]
                m2 = g.iloc[i + 1]
                gap = (m2['match_date'] - m1['match_date']).days

                if gap <= self.dup_window_days:
                    dupe_info = []
                    for m in (m1, m2):
                        row_id = m.get('fixture_id', m.get('id', '?'))
                        lg = m.get('league', '?')
                        sn = m.get('season', '?')
                        dupe_info.append(
                            f"id={row_id} | {m['match_date'].date()} | [{lg} {sn}] {m['home_team']} vs {m['away_team']}"
                        )
                    result["real_dupes"].append({"keys": keys, "matches": dupe_info})
                else:
                    result["legit_repeats"] += 1

        return result

    def print_report(self, report: dict):
        if report.get("status") == "error":
            print(f"   ❌ {report.get('message')}")
            return

        print(f"\n{'=' * 70}")
        print(f"📋 TABULKA: {report['table']}")
        print('=' * 70)

        if report.get("message"):
            print(f"   ℹ️ {report['message']}")
            return

        print(f"   Řádků: {report['rows']}  |  Rozsah: {report['date_min']} → {report['date_max']}")

        if report.get("leagues"):
            lg_str = ", ".join(f"{k}:{v}" for k, v in sorted(report["leagues"].items()))
            print(f"   🌍 Ligy ({len(report['leagues'])}): {lg_str}")

        if report["dupe_names"]:
            print(f"   ⚠️  DUPLICITNÍ NÁZVY ({len(report['dupe_names'])}):")
            for k, det in report["dupe_names"].items():
                print(f"        {', '.join(det)}")
        else:
            print("   ✅ Žádné duplicitní názvy týmů")

        print(
            f"\n   🔁 Opakované dvojice s odstupem > {self.dup_window_days} dnů: {report['legit_repeats']}  (odveta/nadstavba — OK)")

        if not report["real_dupes"]:
            print(f"   ✅ Skutečné duplicity zápasů (do {self.dup_window_days} dnů): 0")
        else:
            print(f"   ⚠️  SKUTEČNÉ DUPLICITY: {len(report['real_dupes'])}")
            for d in report["real_dupes"][:25]:
                print(f"        {d['keys']}:")
                for m in d["matches"]:
                    print(f"           {m}")
            if len(report["real_dupes"]) > 25:
                print(f"        ... a dalších {len(report['real_dupes']) - 25}")