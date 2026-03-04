"""
step6b_manual_odds.py  v1
==========================
Manuální zadání kurzů pro FL (Chance Liga) + jakoukoliv ligu co není v The Odds API.
Kurzy opisz z Fortuny / Tipsportu / Chance.cz před kolem (2 minuty práce).

POUŽITÍ:
    1. Spusť skript: python step6b_manual_odds.py
    2. Zobrazí se seznam nadcházejících FL zápasů
    3. Pro každý zápas zadej kurzy 1 / X / 2 (nebo Enter pro přeskočení)
    4. Kurzy se uloží do bookmaker_odds (stejná tabulka jako The Odds API)
    5. Spusť step7_show_console_tips.py → uvidíš VALUE% i pro FL

WORKFLOW celého kola:
    python step6_fetch_odds.py        ← PL kurzy automaticky
    python step6b_manual_odds.py      ← FL kurzy ručně (volitelné)
    python step7_show_console_tips.py ← dashboard s VALUE%

TIPY:
    - Kurzy opisuj jako desetinné (1.85, ne 18/10)
    - Zadávej kurzy těsně před kolem (kurzy se mění)
    - Doporučený zdroj: ifortuna.cz nebo chance.cz (nejnižší marže v ČR)
"""

import os
import pandas as pd
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

BOOKMAKER_NAME = "Fortuna"   # Změň dle zdroje kurzů


def ensure_table(conn):
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS bookmaker_odds (
            id          SERIAL PRIMARY KEY,
            fixture_id  INTEGER,
            home_team   TEXT,
            away_team   TEXT,
            league      TEXT,
            match_date  DATE,
            odd_1       NUMERIC,
            odd_x       NUMERIC,
            odd_2       NUMERIC,
            book_1      TEXT,
            book_x      TEXT,
            book_2      TEXT,
            all_odds    TEXT,
            fetched_at  TIMESTAMP DEFAULT NOW()
        )
    """))


def get_fl_fixtures(conn) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT fixture_id, home_team, away_team, league, match_date
        FROM prepared_fixtures
        WHERE league = 'FL'
          AND match_date >= CURRENT_DATE
          AND match_date <= CURRENT_DATE + INTERVAL '14 days'
        ORDER BY match_date ASC, home_team ASC
    """), conn)


def get_existing_odds(conn, fixture_ids: list) -> dict:
    """Vrátí existující kurzy pro dané fixture_id (pro zobrazení aktuálních hodnot)."""
    if not fixture_ids:
        return {}
    placeholders = ','.join([f':id{i}' for i in range(len(fixture_ids))])
    params = {f'id{i}': fid for i, fid in enumerate(fixture_ids)}
    rows = conn.execute(text(
        f"SELECT fixture_id, odd_1, odd_x, odd_2, book_1 FROM bookmaker_odds "
        f"WHERE fixture_id IN ({placeholders})"
    ), params).fetchall()
    return {r[0]: r for r in rows}


def prompt_odds(match_str: str, existing=None) -> tuple:
    """
    Interaktivní zadání kurzů pro jeden zápas.
    Vrátí (odd_1, odd_x, odd_2) nebo None pro přeskočení.
    """
    if existing:
        print(f"  Aktuální: 1={existing[1]}  X={existing[2]}  2={existing[3]}  [{existing[4]}]")

    print(f"  Zadej kurzy pro: {match_str}")
    print(f"  (Enter = přeskoč, 's' = ukonči zadávání)")

    try:
        raw_1 = input(f"    Kurz 1 (domácí):  ").strip()
        if raw_1.lower() == 's':
            return 'STOP', None, None
        if not raw_1:
            return None, None, None

        raw_x = input(f"    Kurz X (remíza):  ").strip()
        if raw_x.lower() == 's':
            return 'STOP', None, None
        if not raw_x:
            return None, None, None

        raw_2 = input(f"    Kurz 2 (hosté):   ").strip()
        if raw_2.lower() == 's':
            return 'STOP', None, None
        if not raw_2:
            return None, None, None

        o1 = float(raw_1.replace(',', '.'))
        ox = float(raw_x.replace(',', '.'))
        o2 = float(raw_2.replace(',', '.'))

        # Základní validace
        if not (1.0 < o1 < 50.0 and 1.0 < ox < 50.0 and 1.0 < o2 < 50.0):
            print("  ⚠️  Neplatné kurzy (musí být mezi 1.01 a 50.0) — přeskočeno")
            return None, None, None

        # Kontrola marže (obvyklá bookmaker marže 5–15%)
        implied = 1/o1 + 1/ox + 1/o2
        margin  = (implied - 1.0) * 100
        print(f"  ✅  {o1} / {ox} / {o2}  (marže: {margin:.1f}%)")

        return o1, ox, o2

    except (ValueError, KeyboardInterrupt):
        print("  ⚠️  Neplatný vstup — přeskočeno")
        return None, None, None


def save_odds(conn, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df['all_odds'] = df['all_odds'].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
    )
    ids = df['fixture_id'].dropna().tolist()
    if ids:
        placeholders = ','.join([f':id{i}' for i in range(len(ids))])
        params = {f'id{i}': fid for i, fid in enumerate(ids)}
        conn.execute(text(
            f"DELETE FROM bookmaker_odds WHERE fixture_id IN ({placeholders})"
        ), params)
    df.to_sql('bookmaker_odds', conn, if_exists='append', index=False)


def main():
    print("=" * 65)
    print("✍️  STEP6b: MANUÁLNÍ ZADÁNÍ FL KURZŮ")
    print("=" * 65)
    print(f"  Bookmaker: {BOOKMAKER_NAME}")
    print(f"  Kurzy opisuj z: ifortuna.cz / chance.cz / tipsport.cz")
    print()

    with engine.begin() as conn:
        ensure_table(conn)
        fixtures = get_fl_fixtures(conn)

        if fixtures.empty:
            print("  📭 Žádné FL zápasy v nadcházejících 14 dnech.")
            return

        existing = get_existing_odds(conn, fixtures['fixture_id'].tolist())

    print(f"  📅 FL zápasy k zadání: {len(fixtures)}")
    print(f"  (Enter = přeskoč zápas  |  's' = ukonči)\n")

    rows    = []
    entered = 0
    skipped = 0

    for _, row in fixtures.iterrows():
        date_str  = pd.Timestamp(row['match_date']).strftime("%d.%m.")
        match_str = f"{date_str}  {row['home_team']} vs {row['away_team']}"
        print(f"\n  ── [{entered+skipped+1}/{len(fixtures)}] ─────────────────────────")
        ex = existing.get(row['fixture_id'])

        o1, ox, o2 = prompt_odds(match_str, existing=ex)

        if o1 == 'STOP':
            print("\n  ⛔ Zadávání ukončeno.")
            break
        if o1 is None:
            skipped += 1
            continue

        rows.append({
            'fixture_id': row['fixture_id'],
            'home_team':  row['home_team'],
            'away_team':  row['away_team'],
            'league':     row['league'],
            'match_date': row['match_date'],
            'odd_1':      o1,
            'odd_x':      ox,
            'odd_2':      o2,
            'book_1':     BOOKMAKER_NAME,
            'book_x':     BOOKMAKER_NAME,
            'book_2':     BOOKMAKER_NAME,
            'all_odds':   {BOOKMAKER_NAME: {'1': o1, 'X': ox, '2': o2}},
        })
        entered += 1

    if rows:
        with engine.begin() as conn:
            save_odds(conn, rows)
        print(f"\n  💾 Uloženo {entered} zápasů do bookmaker_odds")
    else:
        print(f"\n  ℹ️  Žádné kurzy nebyly zadány.")

    print(f"  📊 Zadáno: {entered}  |  Přeskočeno: {skipped}")
    if entered > 0:
        print(f"\n  ✅ Hotovo. Nyní spusť: python step7_show_console_tips.py")


if __name__ == "__main__":
    main()