"""
step6_fetch_odds.py  v2
========================
Stahuje reálné kurzy z The Odds API → tabulka bookmaker_odds v DB.
Spouštět PŘED step7_show_console_tips.py (jednou týdně před kolem).

REGISTRACE: https://the-odds-api.com (Free tier: 500 req/měsíc)
ENV: přidej do .env:
    ODDS_API_KEY=tvůj_api_klíč
    ODDS_BANKROLL=10000   (volitelné, pro Kelly výpočet v step7)

SPOTŘEBA requestů (Free tier 500/měsíc):
    1 request  = výpis dostupných lig  (--list-sports)
    2 requesty = kurzy PL + FL za kolo
    Rezerva:  ~240 kol/měsíc → bezpečné

POUŽITÍ:
    python step6_fetch_odds.py              # Stáhni kurzy pro aktuální kolo
    python step6_fetch_odds.py --list-sports  # Zobraz dostupné ligy v API
"""

import os
import sys
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
BANKROLL      = float(os.getenv("ODDS_BANKROLL", "10000"))

engine = create_engine(DATABASE_URL)

BASE_URL    = "https://api.the-odds-api.com/v4"
REGIONS     = "eu"
MARKETS     = "h2h"
ODDS_FORMAT = "decimal"

# ─────────────────────────────────────────────────────────────
# LIGY: sport_key → náš kód ligy v DB
# Pokud liga vrací 404, spusť: python step6_fetch_odds.py --list-sports
# ─────────────────────────────────────────────────────────────
LEAGUES = {
    "soccer_epl": "PL",
    # "soccer_czech_republic_chance_liga": "FL",  # ❌ Není v The Odds API
    # Další dostupné ligy (přidej dle potřeby):
    # "soccer_efl_champ":              "EFL"
    # "soccer_uefa_champs_league":     "UCL"
    # "soccer_germany_bundesliga":     "BUN"
}

# ─────────────────────────────────────────────────────────────
# MAPOVÁNÍ NÁZVŮ: API název → náš název v DB
# ─────────────────────────────────────────────────────────────
TEAM_NAME_MAP = {
    # --- Premier League ---
    # The Odds API vrací jména BEZ "FC" suffixu → mapujeme na DB názvy
    "Arsenal":                          "Arsenal FC",
    "Aston Villa":                      "Aston Villa",
    "Bournemouth":                      "AFC Bournemouth",
    "AFC Bournemouth":                  "AFC Bournemouth",
    "Brentford":                        "Brentford",
    "Brighton and Hove Albion":         "Brighton & Hove Albion",
    "Brighton & Hove Albion":           "Brighton & Hove Albion",
    "Burnley":                          "Burnley FC",
    "Chelsea":                          "Chelsea FC",
    "Crystal Palace":                   "Crystal Palace",
    "Everton":                          "Everton FC",
    "Fulham":                           "Fulham FC",
    "Leeds United":                     "Leeds United",
    "Liverpool":                        "Liverpool FC",
    "Manchester City":                  "Manchester City",
    "Manchester United":                "Manchester United",
    "Newcastle United":                 "Newcastle United",
    "Nottingham Forest":                "Nottingham Forest",
    "Sunderland":                       "AFC Sunderland",
    "AFC Sunderland":                   "AFC Sunderland",
    "Tottenham Hotspur":                "Tottenham Hotspur",
    "Tottenham":                        "Tottenham Hotspur",
    "West Ham United":                  "West Ham United",
    "Wolverhampton Wanderers":          "Wolverhampton Wanderers",
    # --- Chance Liga ---
    "AC Sparta Prague":                 "Sparta Praha",
    "Sparta Prague":                    "Sparta Praha",
    "SK Slavia Prague":                 "Slavia Praha",
    "Slavia Prague":                    "Slavia Praha",
    "Viktoria Plzen":                   "Viktoria Plzeň",
    "FC Viktoria Plzen":                "Viktoria Plzeň",
    "Mlada Boleslav":                   "Mladá Boleslav",
    "FK Mlada Boleslav":                "Mladá Boleslav",
    "Hradec Kralove":                   "Hradec Králové",
    "FK Hradec Kralove":                "Hradec Králové",
    "Sigma Olomouc":                    "Sigma Olomouc",
    "SK Sigma Olomouc":                 "Sigma Olomouc",
    "Banik Ostrava":                    "Baník Ostrava",
    "FC Banik Ostrava":                 "Baník Ostrava",
    "Slovan Liberec":                   "Slovan Liberec",
    "FC Slovan Liberec":                "Slovan Liberec",
    "FK Teplice":                       "Teplice",
    "Teplice":                          "Teplice",
    "FK Jablonec":                      "Jablonec",
    "Jablonec":                         "Jablonec",
    "FK Pardubice":                     "Pardubice",
    "Pardubice":                        "Pardubice",
    "FC Karvina":                       "Karviná",
    "Karvina":                          "Karviná",
    "Dukla Prague":                     "Dukla Praha",
    "FK Dukla Prague":                  "Dukla Praha",
    "FC Zlin":                          "Zlín",
    "Zlin":                             "Zlín",
    "Bohemians 1905":                   "Bohemians",
    "Bohemians Prague 1905":            "Bohemians",
    "FK Slovacko":                      "Slovácko",
    "Slovacko":                         "Slovácko",
    "FC Zbrojovka Brno":                "Zbrojovka Brno",
    "SK Dynamo Ceske Budejovice":       "České Budějovice",
}


def normalize_team(api_name: str) -> str:
    """Přeloží API název týmu na náš interní název.
    Fallback: zkusí přidat/odebrat ' FC' pro automatické párování.
    """
    if api_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[api_name]
    # Fallback 1: přidej FC
    with_fc = api_name + " FC"
    if with_fc in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[with_fc]
    # Fallback 2: odeber FC
    without_fc = api_name.replace(" FC", "").strip()
    if without_fc in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[without_fc]
    return api_name


def list_available_sports():
    """
    Zobrazí všechny sporty/ligy dostupné pro tvůj API klíč.
    Spusť: python step6_fetch_odds.py --list-sports
    """
    print("\n📋 Dostupné sporty/ligy v The Odds API:")
    url    = f"{BASE_URL}/sports"
    params = {"apiKey": ODDS_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        if resp.status_code != 200:
            print(f"  ❌ API chyba {resp.status_code}: {resp.text[:200]}")
            return
        sports = resp.json()
        soccer = [s for s in sports if 'soccer' in s.get('key','').lower()
                  or 'football' in s.get('key','').lower()]
        print(f"  (Zobrazuji jen fotbal, celkem {len(sports)} sportů dostupných)")
        print(f"  Zbývající requesty: {remaining}\n")
        print(f"  {'sport_key':<50} {'Název':<40} {'Aktivní'}")
        print(f"  {'─'*100}")
        for s in soccer:
            active = "✅ Live" if s.get('active') else "  Mimo sezónu"
            print(f"  {s['key']:<50} {s.get('title',''):<40} {active}")
        print(f"\n  💡 Zkopíruj správné sport_key do LEAGUES dict v konfiguraci.")
    except requests.RequestException as e:
        print(f"  ❌ Síťová chyba: {e}")


def fetch_odds_for_league(sport_key: str, league_code: str) -> list:
    url    = f"{BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": "iso",
    }
    try:
        resp      = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")

        if resp.status_code == 401:
            print(f"  ❌ Neplatný API klíč (401). Zkontroluj ODDS_API_KEY v .env")
            return []
        if resp.status_code == 404:
            print(f"  ⚠️  [{league_code}] Liga '{sport_key}' nenalezena (404).")
            print(f"       Spusť: python step6_fetch_odds.py --list-sports")
            print(f"       a zkopíruj správný sport_key do LEAGUES v konfiguraci.")
            return []
        if resp.status_code == 422:
            print(f"  ⚠️  [{league_code}] Liga není dostupná mimo sezónu (422).")
            return []
        if resp.status_code != 200:
            print(f"  ⚠️  API chyba {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        print(f"  ✅ [{league_code}] {len(data)} zápasů  "
              f"| Zbývající req: {remaining} / použito: {used}")
        return data

    except requests.RequestException as e:
        print(f"  ❌ Síťová chyba: {e}")
        return []


def parse_best_odds(bookmakers: list, home_name: str, away_name: str) -> dict:
    """Vybere BEST odds pro 1, X, 2 přes všechny bookmakers."""
    best = {
        '1': {'odd': 0.0, 'bookmaker': '-'},
        'X': {'odd': 0.0, 'bookmaker': '-'},
        '2': {'odd': 0.0, 'bookmaker': '-'},
    }
    all_books = {}

    for bm in bookmakers:
        bm_key  = bm.get('key', bm.get('title', '?'))
        bm_name = bm.get('title', bm_key)
        book_odds = {}

        for market in bm.get('markets', []):
            if market.get('key') != 'h2h':
                continue
            for outcome in market.get('outcomes', []):
                name = outcome.get('name', '')
                odd  = float(outcome.get('price', 0))
                if name == home_name:
                    book_odds['1'] = odd
                    if odd > best['1']['odd']:
                        best['1'] = {'odd': odd, 'bookmaker': bm_name}
                elif name == away_name:
                    book_odds['2'] = odd
                    if odd > best['2']['odd']:
                        best['2'] = {'odd': odd, 'bookmaker': bm_name}
                elif name == 'Draw':
                    book_odds['X'] = odd
                    if odd > best['X']['odd']:
                        best['X'] = {'odd': odd, 'bookmaker': bm_name}
        if book_odds:
            all_books[bm_name] = book_odds

    best['all'] = all_books
    return best


def match_to_fixture(api_home: str, api_away: str, api_date: str,
                     fixtures_df: pd.DataFrame):
    """Páruje API zápas s řádkem v prepared_fixtures (±1 den tolerance)."""
    our_home = normalize_team(api_home)
    our_away = normalize_team(api_away)
    try:
        # Odstraň timezone info aby bylo srovnatelné s DB daty (tz-naive)
        api_dt = pd.Timestamp(api_date).tz_localize(None).normalize()
    except Exception:
        return None, None

    for _, row in fixtures_df.iterrows():
        row_date = pd.Timestamp(row['match_date']).normalize()
        if abs((row_date - api_dt).days) > 1:
            continue
        h_ok = str(row['home_team']).strip().lower() == our_home.strip().lower()
        a_ok = str(row['away_team']).strip().lower() == our_away.strip().lower()
        if h_ok and a_ok:
            return row['fixture_id'], row
    return None, None


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


def save_odds(conn, rows: list):
    if not rows:
        return
    import json
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
    print(f"  💾 Uloženo {len(df)} zápasů do bookmaker_odds")


def main():
    print("=" * 70)
    print("📡 STEP6: STAHOVÁNÍ REÁLNÝCH KURZŮ (The Odds API)")
    print("=" * 70)

    if not ODDS_API_KEY:
        print("\n  ❌ ODDS_API_KEY chybí v .env!")
        print("     1. Zaregistruj se na https://the-odds-api.com (zdarma)")
        print("     2. Přidej do .env: ODDS_API_KEY=tvůj_klíč")
        return

    print(f"\n  API klíč: {ODDS_API_KEY[:8]}...")
    print(f"  Bankroll: {BANKROLL:,.0f} Kč  |  Ligу: {list(LEAGUES.values())}")

    # --list-sports přepínač
    if "--list-sports" in sys.argv:
        list_available_sports()
        return

    # Načti zápasy z DB
    with engine.begin() as conn:
        ensure_table(conn)
        fixtures_df = pd.read_sql(text("""
            SELECT fixture_id, home_team, away_team, league, match_date
            FROM prepared_fixtures
            WHERE match_date >= CURRENT_DATE
              AND match_date <= CURRENT_DATE + INTERVAL '14 days'
            ORDER BY match_date ASC
        """), conn)

    if fixtures_df.empty:
        print("  📭 Žádné nadcházející zápasy v DB.")
        return

    print(f"\n  📅 Zápasy v DB: {len(fixtures_df)}")
    print(f"\n  📡 Stahuji kurzy...")

    all_rows        = []
    total_matched   = 0
    total_unmatched = 0

    for sport_key, league_code in LEAGUES.items():
        api_data = fetch_odds_for_league(sport_key, league_code)
        if not api_data:
            continue

        matched   = 0
        unmatched = []

        for game in api_data:
            api_home = game.get('home_team', '')
            api_away = game.get('away_team', '')
            api_date = game.get('commence_time', '')

            fixture_id, fixture_row = match_to_fixture(
                api_home, api_away, api_date, fixtures_df
            )
            best = parse_best_odds(
                game.get('bookmakers', []), api_home, api_away
            )

            if fixture_id is None:
                unmatched.append(f"{api_home} vs {api_away} ({api_date[:10]})")
                total_unmatched += 1
                continue

            all_rows.append({
                'fixture_id': fixture_id,
                'home_team':  fixture_row['home_team'],
                'away_team':  fixture_row['away_team'],
                'league':     league_code,
                'match_date': fixture_row['match_date'],
                'odd_1':      best['1']['odd'] or None,
                'odd_x':      best['X']['odd'] or None,
                'odd_2':      best['2']['odd'] or None,
                'book_1':     best['1']['bookmaker'],
                'book_x':     best['X']['bookmaker'],
                'book_2':     best['2']['bookmaker'],
                'all_odds':   best['all'],
            })
            matched       += 1
            total_matched += 1

        if unmatched:
            print(f"     Nespárováno v [{league_code}]:")
            for u in unmatched[:8]:
                print(f"       ⚠️  {u}")
            if len(unmatched) > 8:
                print(f"       ... a {len(unmatched)-8} dalších → přidej do TEAM_NAME_MAP")

    if all_rows:
        with engine.begin() as conn:
            save_odds(conn, all_rows)

    print(f"\n  📊 Souhrn:  spárováno {total_matched}  |  nespárováno {total_unmatched}")
    if total_matched > 0:
        print(f"  ✅ Hotovo. Nyní spusť: python step7_show_console_tips.py")
    else:
        print(f"\n  💡 Tipy při 0 spárovaných zápasech:")
        print(f"     1. Spusť: python step6_fetch_odds.py --list-sports")
        print(f"     2. Najdi správné sport_key pro PL a FL")
        print(f"     3. Aktualizuj LEAGUES dict na začátku tohoto souboru")


if __name__ == "__main__":
    main()