import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# --- KONFIGURACE ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Cesta pro uložení scaleru (stejná složka jako modely)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Sjednoceno se step3: ../models relativně od umístění skriptu
# step2 je v etl/, step3/step4 v ml/ → obě dávají stejný projekt/models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

MARKET_VALUES = {
    # --- PREMIER LEAGUE (mil. EUR) ---
    "Manchester City": 1290.0, "Arsenal FC": 1270.0, "Chelsea FC": 1160.0,
    "Liverpool FC": 1040.0, "Manchester United": 719.0, "Tottenham Hotspur": 877.0,
    "Newcastle United": 710.0, "Aston Villa": 532.0, "Brighton & Hove Albion": 510.0,
    "West Ham United": 339.0, "Nottingham Forest": 592.0, "Brentford": 434.0,
    "Crystal Palace": 536.0, "Wolverhampton Wanderers": 278.0, "Everton FC": 424.0,
    "Fulham FC": 373.0, "AFC Bournemouth": 447.0,
    "Leeds United": 321.0, "Burnley FC": 252.0, "AFC Sunderland": 327.0,

    # --- CHANCE LIGA (přibližné hodnoty dle Transfermarkt, mil. EUR) ---
    "Sparta Praha": 82.0, "Slavia Praha": 75.0, "Viktoria Plzeň": 38.0,
    "Baník Ostrava": 22.0, "Mladá Boleslav": 18.0, "Bohemians": 14.0,
    "Slovácko": 13.0, "Sigma Olomouc": 14.0, "Hradec Králové": 12.0,
    "Slovan Liberec": 16.0, "Teplice": 11.0, "Jablonec": 12.0,
    "Pardubice": 15.0, "Zlín": 9.0, "Karviná": 10.0,
    "České Budějovice": 10.0, "Dukla Praha": 8.0,
    "Zbrojovka Brno": 9.0,

    # --- BUNDESLIGA (české názvy z Livesportu, mil. EUR Transfermarkt 2025) ---
    "Bayern":           937.0,
    "Leverkusen":       639.0,
    "Dortmund":         489.0,
    "RB Lipsko":        467.0,
    "Stuttgart":        347.0,
    "Frankfurt":        358.0,
    "Brémy":            197.0,
    "Wolfsburg":        195.0,
    "Hoffenheim":       155.0,
    "Mohuč":            148.0,
    "Union Berlín":     136.0,
    "Augsburg":         134.0,
    "Kolín n. R.":      120.0,
    "Freiburg":         179.0,
    "Mönchengladbach":  180.0,
    "Heidenheim":        91.0,
    "St. Pauli":         75.0,
    "Hamburk":           85.0,

    # --- LA LIGA (české názvy z Livesportu, mil. EUR Transfermarkt 2025) ---
    "Real Madrid":     1540.0,
    "Barcelona":       1180.0,
    "Atl. Madrid":      820.0,
    "Ath. Bilbao":      390.0,
    "Real Sociedad":    320.0,
    "Villarreal":       280.0,
    "Betis":            270.0,
    "Sevilla":          245.0,
    "Girona":           210.0,
    "Valencia":         180.0,
    "Celta Vigo":       165.0,
    "Osasuna":          120.0,
    "Mallorca":         115.0,
    "Vallecano":        105.0,
    "Getafe":            95.0,
    "Espanyol":         100.0,
    "Alavés":            85.0,
    "Levante":           55.0,
    "Elche":             45.0,
    "Oviedo":            40.0,

    # --- SERIE A (české názvy z Livesportu, mil. EUR Transfermarkt 2025) ---
    "Inter":            720.0,
    "Juventus":         580.0,
    "Neapol":           570.0,
    "AC Milán":         540.0,
    "Atalanta":         510.0,
    "AS Řím":           390.0,
    "Lazio":            330.0,
    "Fiorentina":       290.0,
    "Bologna":          270.0,
    "Turín FC":         175.0,
    "Udinese":          120.0,
    "FC Janov":         115.0,
    "Cagliari":         100.0,
    "Parma":             95.0,
    "Como":              90.0,
    "Verona":            88.0,
    "Lecce":             85.0,
    "Sassuolo":          80.0,
    "Cremonese":         65.0,
    "Pisa":              55.0,
}

# CORE STATS (robustní, 0-5% NULL) - VŽDY použít
CORE_STATS = [
    'goals', 'possession',
    'shots', 'shots_on_target', 'shots_off_target', 'blocked_shots',
    'passes_total', 'passes_completed',
    'expected_goals', 'corners', 'free_kicks', 'throw_ins',
    'fouls', 'yellow_cards', 'red_cards',
    'saves', 'offsides', 'interceptions'
]

# OPTIONAL STATS (25-50% NULL v archivech) - použít když dostupné
OPTIONAL_STATS = [
    'xgot',                # 77% NULL v archivech, 0% v PL 2025-26 ← DŮLEŽITÉ!
    'big_chances',         # 39% NULL
    'box_touches',         # 29% NULL
    'shots_inside_box',    # 29% NULL
    'shots_outside_box',   # 32% NULL
    'passes_accuracy',     # pouze nové sezóny, v archivech NULL
    'errors_shot',         # pouze nové sezóny (přidáno 2026-03)
    'errors_goal',         # pouze nové sezóny (přidáno 2026-03)
]

# STATS_TO_ROLL: CORE stats pro hlavní rolling pipeline
# OPTIONAL stats jsou zpracovány separátně v compute_optional_stats (NULL-aware)
STATS_TO_ROLL = CORE_STATS

# Conceded stats
CORE_CONCEDED = ['expected_goals', 'shots', 'shots_on_target', 'goals']
OPTIONAL_CONCEDED = ['xgot', 'big_chances', 'box_touches']

# ❌ STÁLE VYŘAZENO (>75% NULL, moc málo dat):
# expected_assists (76%), duels_won (76%), long_balls_* (76%),
# prevented_goals (77%), woodwork (78%), through_balls (90%),
# pass_accuracy (100%), clearances (29%), tackles_* (29%),
# crosses_* (29%), passes_final_third_* (29%)

# Sezóny vyloučené z tréninku modelu (zachovány v DB pro ELO výpočet)
# Důvod: příliš málo statistik → fillna(0) kazí rolling průměry
# FL 2022-23: jen ~24 stat/zápas, chybí xG, passes, blocked_shots atd.
EXCLUDE_FROM_TRAINING = [
    ("FL", "2022-23"),
]


def load_data(conn):
    """Načte VŠECHNY fixtures a propojí je se statistikami.

    Načítá CORE i OPTIONAL stats — optional sloupce budou mít NULL
    kde data nebyla dostupná (starší sezóny). compute_optional_stats
    pak tyto NULL správně ignoruje při rolling průměru.
    """
    stats_columns = []
    # Všechny unikátní statistiky (CORE + OPTIONAL, bez duplikátů)
    all_stats = list(dict.fromkeys(CORE_STATS + OPTIONAL_STATS))
    for stat in all_stats:
        if stat == 'goals':
            continue  # goals jsou již načteny jako goals_home/goals_away zvlášť
        stats_columns.append(f"s.{stat}_home")
        stats_columns.append(f"s.{stat}_away")

    cols_sql = ",\n            ".join(stats_columns)

    query = f"""
        SELECT 
            f.id as fixture_id, f.match_date, f.home_team, f.away_team, f.league, f.season,
            s.goals_home, s.goals_away,
            {cols_sql}
        FROM fixtures f
        LEFT JOIN match_statistics s ON f.id = s.fixture_id
        ORDER BY f.match_date ASC
    """
    return pd.read_sql(text(query), conn)


def normalize_numeric(df):
    """Převede sloupce na čísla a normalizuje procenta."""
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in df.columns:
        if col in ['fixture_id', 'match_date', 'home_team', 'away_team', 'league', 'season']:
            continue

        df[col] = pd.to_numeric(df[col], errors='coerce')

        if "accuracy" in col or "possession" in col:
            mask = df[col] > 1.0
            df.loc[mask, col] = df.loc[mask, col] / 100.0

    return df


def compute_target(df):
    """Vytvoří target proměnné pro odehrané zápasy."""
    played_mask = df['goals_home'].notna() & df['goals_away'].notna()

    df["home_win"] = np.nan
    df["points_home"] = np.nan
    df["points_away"] = np.nan

    df.loc[played_mask, "home_win"] = (
            df.loc[played_mask, "goals_home"] > df.loc[played_mask, "goals_away"]
    ).astype(int)

    df.loc[played_mask, "points_home"] = np.where(
        df.loc[played_mask, "goals_home"] > df.loc[played_mask, "goals_away"], 3,
        np.where(df.loc[played_mask, "goals_home"] == df.loc[played_mask, "goals_away"], 1, 0)
    )

    df.loc[played_mask, "points_away"] = np.where(
        df.loc[played_mask, "goals_away"] > df.loc[played_mask, "goals_home"], 3,
        np.where(df.loc[played_mask, "goals_home"] == df.loc[played_mask, "goals_away"], 1, 0)
    )

    return df


def compute_elo(df):
    """Vypočítá dynamické ELO pro každý zápas."""
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo = {team: 1500.0 for team in teams}
    K = 30
    h_elos, a_elos = [], []

    df = df.sort_values("match_date").reset_index(drop=True)

    for idx, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        r_h, r_a = elo[h], elo[a]
        h_elos.append(r_h)
        a_elos.append(r_a)

        if pd.notna(row['goals_home']) and pd.notna(row['goals_away']):
            exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))

            if row['goals_home'] > row['goals_away']:
                actual_h = 1.0
            elif row['goals_home'] == row['goals_away']:
                actual_h = 0.5
            else:
                actual_h = 0.0

            shift = K * (actual_h - exp_h)
            elo[h] += shift
            elo[a] -= shift

    df["home_elo"] = h_elos
    df["away_elo"] = a_elos

    print(f"  ✅ ELO vypočítáno: rozsah {min(h_elos):.0f} - {max(h_elos):.0f}")

    return df


def compute_optional_stats(df, unique_teams):
    """
    Počítá rolling averages pro OPTIONAL stats (xgot, big_chances, atd.)

    KLÍČOVÝ ROZDÍL:
    - NULL hodnoty se NENAHRAZUJÍ nulou
    - Průměr se počítá JEN z non-NULL hodnot
    - Přidává se "coverage" = % non-NULL dat v okně

    Výhoda:
    - xgot z 2 zápasů (1.2, 1.4) = avg 1.3, coverage 0.4 (40% dat)
    - Model se naučí důvěřovat jen features s vysokou coverage
    """
    new_cols = {}

    # Inicializace sloupců
    for col in OPTIONAL_STATS:
        for prefix in ['home', 'away']:
            new_cols[f"{prefix}_avg_{col}_last5"] = pd.Series(dtype=float, index=df.index)
            new_cols[f"{prefix}_{col}_coverage"] = pd.Series(dtype=float, index=df.index)

        # Conceded (jen pro některé)
        if col in OPTIONAL_CONCEDED:
            for prefix in ['home', 'away']:
                new_cols[f"{prefix}_avg_{col}_conceded_last5"] = pd.Series(dtype=float, index=df.index)

    print("  📊 Počítám optional stats (xgot, big_chances, atd.) s coverage...")

    for team_idx, team in enumerate(unique_teams, 1):
        if team_idx % 10 == 0:
            print(f"    Optional: {team_idx}/{len(unique_teams)} týmů...")

        mask_h = df['home_team'] == team
        mask_a = df['away_team'] == team
        team_matches = df[mask_h | mask_a].copy()

        for col in OPTIONAL_STATS:
            col_h, col_a = f"{col}_home", f"{col}_away"

            # Check pokud sloupec existuje v DB
            if col_h not in df.columns:
                continue

            # Vlastní statistiky
            team_stats = pd.Series(index=team_matches.index, dtype=float)
            team_stats.loc[mask_h] = team_matches.loc[mask_h, col_h]
            team_stats.loc[mask_a] = team_matches.loc[mask_a, col_a]

            # ⭐ KLÍČOVÝ KROK: NENAHRAZUJ NULL!
            # pandas .mean() automaticky ignoruje NaN
            shifted = team_stats.shift(1)  # Nepoužij aktuální zápas

            # Rolling average (ignoruje NULL)
            roll_avg = shifted.rolling(window=5, min_periods=1).mean()

            # Coverage: kolik % hodnot v okně není NULL
            def calc_coverage(x):
                non_null = x.notna().sum()
                total = len(x)
                return non_null / total if total > 0 else 0.0

            roll_coverage = shifted.rolling(window=5, min_periods=1).apply(
                calc_coverage, raw=False
            )

            # Ulož
            new_cols[f"home_avg_{col}_last5"].update(roll_avg[mask_h])
            new_cols[f"away_avg_{col}_last5"].update(roll_avg[mask_a])
            new_cols[f"home_{col}_coverage"].update(roll_coverage[mask_h])
            new_cols[f"away_{col}_coverage"].update(roll_coverage[mask_a])

            # Conceded stats
            if col in OPTIONAL_CONCEDED:
                conceded = pd.Series(index=team_matches.index, dtype=float)
                conceded.loc[mask_h] = team_matches.loc[mask_h, col_a]
                conceded.loc[mask_a] = team_matches.loc[mask_a, col_h]

                shifted_conc = conceded.shift(1)
                roll_conc = shifted_conc.rolling(window=5, min_periods=1).mean()

                new_cols[f"home_avg_{col}_conceded_last5"].update(roll_conc[mask_h])
                new_cols[f"away_avg_{col}_conceded_last5"].update(roll_conc[mask_a])

    print("  ✅ Optional stats dokončeny")

    return pd.DataFrame(new_cols)

def compute_features(df):
    """
    Vypočítá klouzavé průměry s weighted averages.
    OPTIMALIZOVÁNO: Pouze last5 (last10 odstraněno).
    """
    df = df.sort_values("match_date").reset_index(drop=True)

    cols_to_avg = ['points'] + STATS_TO_ROLL

    new_cols = {}
    unique_teams = pd.concat([df['home_team'], df['away_team']]).unique()

    # Pouze last5 (last10 odstraněno)
    for col in cols_to_avg:
        new_cols[f"home_avg_{col}_last5"] = pd.Series(dtype=float, index=df.index)
        new_cols[f"away_avg_{col}_last5"] = pd.Series(dtype=float, index=df.index)

        if col in ['expected_goals', 'shots', 'shots_on_target', 'goals']:
            new_cols[f"home_avg_{col}_conceded_last5"] = pd.Series(dtype=float, index=df.index)
            new_cols[f"away_avg_{col}_conceded_last5"] = pd.Series(dtype=float, index=df.index)

    print("\n  🔄 Počítám rolling averages (weighted by opponent ELO)...")

    for team_idx, team in enumerate(unique_teams, 1):
        if team_idx % 5 == 0:
            print(f"    Zpracováno {team_idx}/{len(unique_teams)} týmů...")

        mask_h = df['home_team'] == team
        mask_a = df['away_team'] == team
        team_matches = df[mask_h | mask_a].copy()

        for col in cols_to_avg:
            col_h, col_a = f"{col}_home", f"{col}_away"
            if col_h not in df.columns:
                continue

            team_stats = pd.Series(index=team_matches.index, dtype=float)
            team_stats.loc[mask_h] = team_matches.loc[mask_h, col_h]
            team_stats.loc[mask_a] = team_matches.loc[mask_a, col_a]

            # Weighted by opponent ELO
            opponent_elo = pd.Series(index=team_matches.index, dtype=float)
            opponent_elo.loc[mask_h] = team_matches.loc[mask_h, 'away_elo']
            opponent_elo.loc[mask_a] = team_matches.loc[mask_a, 'home_elo']

            elo_weights = (opponent_elo / 1500).clip(0.8, 1.2)

            # OPRAVA: NULL nahrazuj až PO weighted, ne před.
            # Původní fillna(0) PŘED weightingem způsoboval podhodnocení týmů
            # s chybějícími historickými stats (FL archivní sezóny).
            # Příklad: [NULL, NULL, NULL, 1.5, 1.8] → old: 0.66, new: 1.65
            weighted_stats = team_stats * elo_weights  # NULL * weight = NULL → ignorováno
            weighted_stats_filled = weighted_stats.fillna(0)  # fillna AŽ pro rolling

            roll_5 = weighted_stats_filled.shift(1).rolling(window=5, min_periods=1).mean()

            new_cols[f"home_avg_{col}_last5"].update(roll_5[mask_h])
            new_cols[f"away_avg_{col}_last5"].update(roll_5[mask_a])

            # Conceded stats (jen pro klíčové útočné metriky)
            # xgot/big_chances/box_touches conceded jsou v compute_optional_stats
            if col in ['expected_goals', 'shots', 'shots_on_target', 'goals']:
                conceded_stats = pd.Series(index=team_matches.index, dtype=float)
                conceded_stats.loc[mask_h] = team_matches.loc[mask_h, col_a]
                conceded_stats.loc[mask_a] = team_matches.loc[mask_a, col_h]

                conceded_stats_filled = conceded_stats.fillna(0)
                roll_against = conceded_stats_filled.shift(1).rolling(window=5, min_periods=1).mean()

                new_cols[f"home_avg_{col}_conceded_last5"].update(roll_against[mask_h])
                new_cols[f"away_avg_{col}_conceded_last5"].update(roll_against[mask_a])

    print("  ✅ Rolling averages dokončeny")

    # Statické sloupce
    new_cols["market_value_home"] = df["home_team"].map(MARKET_VALUES).fillna(200)
    new_cols["market_value_away"] = df["away_team"].map(MARKET_VALUES).fillna(200)
    new_cols["market_value_diff"] = new_cols["market_value_home"] - new_cols["market_value_away"]

    # Fatigue
    home_rest = pd.Series(7.0, index=df.index)
    away_rest = pd.Series(7.0, index=df.index)
    for team in unique_teams:
        team_indices = df[(df['home_team'] == team) | (df['away_team'] == team)].index
        diff = df.loc[team_indices, 'match_date'].diff().dt.days.fillna(7)
        for idx in team_indices:
            if df.at[idx, 'home_team'] == team:
                home_rest[idx] = diff.at[idx]
            else:
                away_rest[idx] = diff.at[idx]

    new_cols["home_rest_days"] = home_rest
    new_cols["away_rest_days"] = away_rest

    # --- OPTIONAL STATS ---
    # (Přidej PŘED řádek: df_new_features = pd.DataFrame(new_cols))
    df_optional = compute_optional_stats(df, unique_teams)

    # Sloučit s ostatními features
    df_new_features = pd.DataFrame(new_cols)
    df_new_features = pd.concat([df_new_features, df_optional], axis=1)
    df = pd.concat([df, df_new_features], axis=1)

    return df


def create_enhanced_features(df):
    """
    OPTIMALIZOVANÉ Enhanced features.

    ODSTRANĚNO:
    - attacking_quality (horší než xgot)
    - form_momentum (nízká korelace)

    PŘIDÁNO:
    - Goal difference average
    - Interaction features
    """
    print("\n  🚀 Vytvářím optimalizované enhanced features...")

    # 1. SHOT CONVERSION (vylepšeno)
    for side in ['home', 'away']:
        goals = f'{side}_avg_goals_last5'
        sot = f'{side}_avg_shots_on_target_last5'

        if goals in df.columns and sot in df.columns:
            # Lepší handling: min 1 střela na branku
            df[f'{side}_shot_conversion'] = (
                    df[goals] / df[sot].clip(lower=1.0)
            ).clip(upper=1.0).fillna(0)

    # 2. DEFENSIVE RATING
    for side in ['home', 'away']:
        conceded = f'{side}_avg_goals_conceded_last5'

        if conceded in df.columns:
            df[f'{side}_defensive_rating'] = (1 / (df[conceded] + 1)).fillna(0)

    # 3. POSSESSION QUALITY
    for side in ['home', 'away']:
        # Místo pass_accuracy použijeme pass completion rate
        poss = f'{side}_avg_possession_last5'
        passes_total = f'{side}_avg_passes_total_last5'
        passes_completed = f'{side}_avg_passes_completed_last5'

        if all(col in df.columns for col in [poss, passes_total, passes_completed]):
            # Completion rate = completed / total
            completion_rate = (df[passes_completed] / df[passes_total].clip(lower=1)).clip(upper=1.0).fillna(0)
            df[f'{side}_possession_quality'] = (df[poss] * completion_rate).fillna(0)

    # 4. DISCIPLINE SCORE
    for side in ['home', 'away']:
        yellow = f'{side}_avg_yellow_cards_last5'
        red = f'{side}_avg_red_cards_last5'
        fouls = f'{side}_avg_fouls_last5'

        if all(col in df.columns for col in [yellow, red, fouls]):
            df[f'{side}_discipline_score'] = (
                    df[yellow] + df[red] * 3 + df[fouls] / 10
            ).fillna(0)

    # 5. GOAL DIFFERENCE AVERAGE - ZÁMĚRNĚ ODSTRANĚNO
    # Důvod: home_goal_diff_avg = home_avg_goals_last5 - home_avg_goals_conceded_last5
    # Obě source features jsou již v datasetu → pouze přidává multikolinearitu
    # a v feature selection bere slot neutral features místo market_value_diff_scaled

    # 6. HOME ADVANTAGE
    df['home_advantage'] = 1.0

    # 7. ELO DIFFERENCE
    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        df['elo_diff'] = df['home_elo'] - df['away_elo']

    # 8. NORMALIZED MARKET VALUE
    if 'market_value_diff' in df.columns:
        scaler = StandardScaler()
        df['market_value_diff_scaled'] = scaler.fit_transform(
            df[['market_value_diff']]
        )
        # Uložit scaler pro step4 — jinak step4 dostane raw diff (stovky) místo ±2 → exploze xG!
        scaler_path = os.path.abspath(os.path.join(MODEL_DIR, "market_value_scaler.pkl"))
        joblib.dump(scaler, scaler_path)
        print(f"  ✅ market_value_scaler.pkl uložen → {scaler_path}")
        print(f"     (mean={scaler.mean_[0]:.1f}, std={scaler.scale_[0]:.1f})")

    # 9. REST DAYS CATEGORIES
    for side in ['home', 'away']:
        rest = f'{side}_rest_days'
        if rest in df.columns:
            df[f'{side}_rest_category'] = pd.cut(
                df[rest],
                bins=[-1, 2, 5, 100],
                labels=[0, 1, 2]
            ).astype(float).fillna(1)

    # 10. NOVÉ: INTERACTION FEATURES (klíčové!)
    print("\n  🔗 Vytvářím interaction features...")

    # ELO x Market Value (síla × finance)
    if 'elo_diff' in df.columns and 'market_value_diff_scaled' in df.columns:
        df['elo_x_market'] = df['elo_diff'] * df['market_value_diff_scaled']
        print(f"  ✅ elo_x_market")

    # Attack x Defense (útok domácích × obrana hostů)
    if all(c in df.columns for c in ['home_avg_xgot_last5', 'away_defensive_rating']):
        df['attack_vs_defense'] = df['home_avg_xgot_last5'] * df['away_defensive_rating']
        print(f"  ✅ attack_vs_defense")

    # Home Advantage x ELO (domácí výhoda × síla)
    if 'elo_diff' in df.columns:
        df['home_x_elo'] = df['home_advantage'] * df['elo_diff']
        print(f"  ✅ home_x_elo")

    # Form x Quality (recent form × attacking quality)
    if all(c in df.columns for c in ['home_avg_points_last5', 'home_avg_xgot_last5']):
        df['form_x_attack_home'] = df['home_avg_points_last5'] * df['home_avg_xgot_last5']
        df['form_x_attack_away'] = df['away_avg_points_last5'] * df['away_avg_xgot_last5']
        print(f"  ✅ form_x_attack (home/away)")

    # Liga jako feature — rozlišuje velké ligy (PL, BL, LL, SA) od FL
    df['is_top5_league'] = (df['league'].isin(['PL', 'BL', 'LL', 'SA'])).astype(float)
    df['is_premier_league'] = (df['league'] == 'PL').astype(float)  # zachováno pro zpětnou kompatibilitu

    print("  ✅ Optimalizované features vytvořeny")

    return df


def remove_low_variance_features(df, threshold=0.005):
    """Odstraní features s velmi nízkou variancí."""
    print(f"\n  🔍 Odstraňuji low-variance features (threshold={threshold})...")

    feature_cols = [c for c in df.columns if c.startswith(('home_avg_', 'away_avg_', 'home_', 'away_'))]

    removed = []
    for col in feature_cols:
        if col in ['home_advantage', 'home_elo', 'away_elo', 'home_team', 'away_team']:
            continue

        if df[col].dtype in ['float64', 'int64']:
            var = df[col].var()
            if var < threshold and not df[col].isna().all():
                df = df.drop(columns=[col])
                removed.append((col, var))

    if removed:
        print(f"    Odstraněno {len(removed)} sloupců")
    else:
        print("    ✅ Všechny features mají dostatečnou varianci")

    return df


def save_dataset(conn, df):
    """Rozdělí dataset na historická data a budoucí predikce."""
    today = pd.Timestamp.now().normalize()  # datetime, ne date → konzistentní porovnání

    df_historical = df[df['match_date'] < today].copy()

    # Vyloučit sezóny s chudými statistikami z tréninku
    # (zůstávají v DB a přispívají k ELO výpočtu, ale model je nevidí)
    if EXCLUDE_FROM_TRAINING:
        exclude_mask = pd.Series(False, index=df_historical.index)
        for league, season in EXCLUDE_FROM_TRAINING:
            mask = (df_historical['league'] == league) & (df_historical['season'] == season)
            exclude_mask |= mask
            n = mask.sum()
            if n > 0:
                print(f"   ⚠️  Vyloučeno z tréninku: {league} {season} ({n} zápasů) — chudé statistiky")
        df_historical = df_historical[~exclude_mask].copy()

    played_count = df_historical['goals_home'].notna().sum()
    total_historical = len(df_historical)

    print(f"\n📊 DIAGNOSTIKA:")
    print(f"   Historické zápasy: {total_historical} (odehrané: {played_count})")

    df_historical = df_historical.fillna(0)
    df_historical = df_historical.loc[:, ~df_historical.columns.duplicated()].copy()

    df_fixtures = df[df['match_date'] >= today].copy()

    keep_cols = ['fixture_id', 'match_date', 'home_team', 'away_team', 'league', 'season']
    keep_cols.extend(['home_elo', 'away_elo', 'elo_diff'])

    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in [
        'home_avg_', 'away_avg_',
        'home_shot', 'away_shot',
        'home_defensive', 'away_defensive',
        'home_possession', 'away_possession',
        'home_discipline', 'away_discipline',
        'home_rest', 'away_rest',
        'home_goal_diff', 'away_goal_diff',
        'elo_x_', 'attack_vs_', 'home_x_', 'form_x_'
    ])]
    keep_cols.extend(feature_cols)

    static_cols = ['market_value_diff', 'market_value_diff_scaled', 'home_advantage']
    for c in static_cols:
        if c in df.columns:
            keep_cols.append(c)

    keep_cols = list(dict.fromkeys(keep_cols))
    final_cols = [c for c in keep_cols if c in df_fixtures.columns]

    df_fixtures = df_fixtures[final_cols]
    df_fixtures = df_fixtures.fillna(0)

    print(f"\n💾 Ukládám do DB...")
    print(f"   Trénink: {len(df_historical)} zápasů")
    print(f"   Predikce: {len(df_fixtures)} zápasů, {len(final_cols)} features")

    df_historical.to_sql("prepared_datasets", conn, if_exists="replace", index=False)
    df_fixtures.to_sql("prepared_fixtures", conn, if_exists="replace", index=False)

    # Validace ELO
    if 'home_elo' in df_fixtures.columns:
        elo_ok = df_fixtures['home_elo'].notna().all()
        if elo_ok:
            print(f"   ✅ ELO: {df_fixtures['home_elo'].min():.0f} - {df_fixtures['home_elo'].max():.0f}")
        else:
            print(f"   ⚠️  ELO má missing values!")


def main():
    with engine.begin() as conn:
        print("=" * 70)
        print("🚀 STEP2: OPTIMALIZOVANÝ DATASET")
        print("=" * 70)

        df = load_data(conn)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Zajisti datetime typ — nutné pro .dt.days a porovnání s today
        df['match_date'] = pd.to_datetime(df['match_date'])

        if df.empty:
            print("❌ Žádná data")
            return

        print(f"\n📥 Načteno: {len(df)} zápasů")
        print(f"   Odehrané: {df['goals_home'].notna().sum()}")
        print(f"   Budoucí: {len(df) - df['goals_home'].notna().sum()}")

        print("\n🔧 Zpracovávám...")
        df = normalize_numeric(df)
        df = compute_target(df)

        print("\n  📊 ELO ratings...")
        df = compute_elo(df)

        df = compute_features(df)
        df = create_enhanced_features(df)
        df = remove_low_variance_features(df, threshold=0.005)

        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        print(f"⚠️  Konstantní sloupce ({len(const_cols)}): {const_cols}")

        save_dataset(conn, df)

        print(f"\n{'=' * 70}")
        print("✅ HOTOVO!")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()