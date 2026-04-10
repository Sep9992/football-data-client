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

    # 5. Srovnání všech lig vs PL (referenční liga s nejlepším pokrytím)
    ref_league = 'PL' if 'PL' in leagues else leagues[0]
    other_leagues = [lg for lg in leagues if lg != ref_league]
    if other_leagues:
        print(f"\n{'─'*70}")
        print(f"🆚 SROVNÁNÍ LIG vs {ref_league} — sloupce kde rozdíl > 20%")
        print(f"{'─'*70}")
        for lg in other_leagues:
            diffs = []
            for col, cov in coverage_data.items():
                ref_pct = cov.get(ref_league)
                lg_pct  = cov.get(lg)
                if ref_pct is not None and lg_pct is not None:
                    diff = ref_pct - lg_pct   # kladné = lg má méně než PL
                    if abs(diff) > 20:
                        diffs.append((col, lg_pct, ref_pct, diff))
            if not diffs:
                print(f"\n  {lg} vs {ref_league}: ✅ žádný rozdíl > 20%")
                continue
            diffs.sort(key=lambda x: x[3], reverse=True)
            print(f"\n  {lg} vs {ref_league}:")
            print(f"  {'Statistika':<40}  {lg:>6}  {ref_league:>6}  {'Rozdíl':>8}")
            print(f"  {'─'*62}")
            for col, lg_pct, ref_pct, diff in diffs:
                marker = f"← {lg} chybí" if diff > 0 else f"← {ref_league} chybí"
                print(f"  {col:<40}  {lg_pct:>5.0f}%  {ref_pct:>5.0f}%  {diff:>+7.0f}%  {marker}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()