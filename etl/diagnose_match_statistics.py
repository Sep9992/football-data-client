"""
diagnose_match_statistics.py
============================
Zobrazí pokrytí statistik v tabulce match_statistics
rozdělené podle ligy a sezóny.

Spuštění:
    python diagnose_match_statistics.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# Statistické sloupce (bez meta-sloupců)
SKIP_COLS = {
    'id', 'fixture_id', 'league', 'season', 'created_at',
    'goals_home', 'goals_away'
}

def main():
    with engine.connect() as conn:

        # 1. Načti vše
        df = pd.read_sql(text("""
            SELECT ms.*
            FROM match_statistics ms
            
            ORDER BY ms.league, ms.season
        """), conn)

    if df.empty:
        print("❌ Žádná data v match_statistics")
        return

    stat_cols = [c for c in df.columns if c not in SKIP_COLS]

    print("=" * 70)
    print("📊 DIAGNOSTIKA match_statistics")
    print("=" * 70)
    print(f"  Celkem záznamů: {len(df)}")
    print(f"  Statistických sloupců: {len(stat_cols)}")

    # 2. Přehled záznamů podle ligy + sezóny
    print(f"\n{'─'*70}")
    print("📋 ZÁZNAMY PODLE LIGY A SEZÓNY")
    print(f"{'─'*70}")
    summary = df.groupby(['league', 'season']).size().reset_index(name='zápasů')
    for _, row in summary.iterrows():
        print(f"  {row['league']:6s}  {row['season']:8s}  →  {row['zápasů']:4d} zápasů")

    # 3. Pokrytí každé statistiky podle ligy
    print(f"\n{'─'*70}")
    print("📈 POKRYTÍ STATISTIK (% non-NULL) PODLE LIGY")
    print(f"{'─'*70}")

    leagues = sorted(df['league'].unique())

    # Záhlaví
    header = f"  {'Statistika':<40}"
    for lg in leagues:
        header += f"  {lg:>8}"
    print(header)
    print("  " + "─" * (40 + len(leagues) * 10))

    # Pokrytí pro každý sloupec
    coverage_data = {}
    for col in sorted(stat_cols):
        row_str = f"  {col:<40}"
        col_coverage = {}
        for lg in leagues:
            mask = df['league'] == lg
            total = mask.sum()
            if total == 0:
                pct = None
            else:
                non_null = df.loc[mask, col].notna().sum()
                pct = non_null / total * 100
            col_coverage[lg] = pct
            if pct is None:
                row_str += f"  {'N/A':>7} "
            else:
                row_str += f"  {pct:>6.0f}% "
        coverage_data[col] = col_coverage
        print(row_str)

    # 4. Problémové statistiky (< 50% pokrytí v jakékoli lize)
    print(f"\n{'─'*70}")
    print("⚠️  STATISTIKY S NÍZKÝM POKRYTÍM (< 50% v nějaké lize)")
    print(f"{'─'*70}")
    problems = []
    for col, cov in coverage_data.items():
        for lg, pct in cov.items():
            if pct is not None and pct < 50:
                problems.append((col, lg, pct))

    if problems:
        problems.sort(key=lambda x: x[2])
        for col, lg, pct in problems:
            print(f"  {col:<40}  {lg:6s}  {pct:5.0f}%")
    else:
        print("  ✅ Všechny statistiky mají pokrytí ≥ 50%")

    # 5. Srovnání FL vs PL
    if 'FL' in leagues and 'PL' in leagues:
        print(f"\n{'─'*70}")
        print("🆚 SROVNÁNÍ FL vs PL (sloupce kde rozdíl > 20%)")
        print(f"{'─'*70}")
        print(f"  {'Statistika':<40}  {'FL':>6}  {'PL':>6}  {'Rozdíl':>8}")
        print(f"  {'─'*62}")
        diffs = []
        for col, cov in coverage_data.items():
            fl_pct = cov.get('FL')
            pl_pct = cov.get('PL')
            if fl_pct is not None and pl_pct is not None:
                diff = pl_pct - fl_pct
                if abs(diff) > 20:
                    diffs.append((col, fl_pct, pl_pct, diff))
        diffs.sort(key=lambda x: x[3], reverse=True)
        for col, fl, pl, diff in diffs:
            marker = "← FL chybí" if diff > 0 else "← PL chybí"
            print(f"  {col:<40}  {fl:>5.0f}%  {pl:>5.0f}%  {diff:>+7.0f}%  {marker}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
