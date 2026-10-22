import os
import sys
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# DYNAMICKÝ IMPORT
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from ai_agent.config.market_values import MARKET_VALUES

# Načtení konfigurace a připojení k DB
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ Nenalezeno DATABASE_URL v .env souboru!")

engine = create_engine(DATABASE_URL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

STATS_DB_MAP = {
    "Očekávané góly (xG)": "expected_goals",
    "xGOT": "xgot",
    "Držení míče": "possession",
    "Střely celkem": "shots",
    "Střely na branku": "shots_on_target",
    "Střely mimo branku": "shots_off_target",
    "Zblokované střely": "blocked_shots",
    "Střely uvnitř vápna": "shots_inside_box",
    "Střely mimo vápno": "shots_outside_box",
    "Střely do tyče/břevna": "woodwork",
    "Velké šance": "big_chances",
    "Rohové kopy": "corners",
    "Doteky ve vápně soupeře": "box_touches",
    "Přesné průnikové přihrávky": "through_balls",
    "Ofsajdy": "offsides",
    "Přímé kopy": "free_kicks",
    "Očekávané asistence (xA)": "expected_assists",
    "Vhazování": "throw_ins",
    "Fauly": "fouls",
    "Žluté karty": "yellow_cards",
    "Červené karty": "red_cards",
    "Vyhrané souboje": "duels_won",
    "Obranné odkopy": "clearances",
    "Přerušení přihrávek": "interceptions",
    "Brankářské zákroky": "saves",
    "Zabráněné góly": "prevented_goals",
    "Chyby vedoucí ke střele": "errors_shot",
    "Chyby vedoucí ke gólu": "errors_goal",
    "Zmařené velké šance": "big_chances_missed",
    "Získané míče": "ball_recoveries",
    "Ztráty míče": "possession_lost",
}

COMPLEX_METRICS = [
    "passes", "long_balls", "passes_final_third", "crosses", "tackles"
]

ALL_STAT_COLUMNS = list(set(STATS_DB_MAP.values()))

for metric in COMPLEX_METRICS:
    ALL_STAT_COLUMNS.extend([f"{metric}_total", f"{metric}_completed", f"{metric}_accuracy"])

ALL_STAT_COLUMNS.extend(["dribbles_completed", "dribbles_total", "dribbles_accuracy"])

CORE_STATS = [
    'goals', 'possession', 'shots', 'shots_on_target', 'shots_off_target', 'blocked_shots',
    'passes_total', 'passes_completed', 'expected_goals', 'corners', 'free_kicks', 'throw_ins',
    'fouls', 'yellow_cards', 'red_cards', 'saves', 'offsides', 'interceptions',
    'big_chances_missed', 'ball_recoveries', 'possession_lost', 'dribbles_completed', 'dribbles_total'
]

STATS_TO_ROLL = CORE_STATS
OPTIONAL_CONCEDED = ['xgot', 'big_chances', 'box_touches']

EXCLUDE_FROM_TRAINING = [("FL", "2022-23")]


class DatasetBuilder:
    def __init__(self, db_engine=None):
        self.engine = db_engine if db_engine else engine

    def load_data(self, conn):
        stats_columns = []
        for stat in ALL_STAT_COLUMNS:
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

    def normalize_numeric(self, df):
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for col in df.columns:
            if col in ['fixture_id', 'match_date', 'home_team', 'away_team', 'league', 'season']:
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if "accuracy" in col or "possession" in col:
                mask = df[col] > 1.0
                df.loc[mask, col] = df.loc[mask, col] / 100.0
        return df

    def compute_target(self, df):
        played_mask = df['goals_home'].notna() & df['goals_away'].notna()
        df = df.copy()

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

    def compute_elo(self, df):
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
                actual_h = 1.0 if row['goals_home'] > row['goals_away'] else (
                    0.5 if row['goals_home'] == row['goals_away'] else 0.0
                )
                shift = K * (actual_h - exp_h)
                elo[h] += shift
                elo[a] -= shift

        df["home_elo"] = h_elos
        df["away_elo"] = a_elos
        return df

    def compute_features(self, df):
        df = df.sort_values("match_date").reset_index(drop=True)
        cols_to_avg = ['points'] + STATS_TO_ROLL
        conceded_cols = {'expected_goals', 'shots', 'shots_on_target', 'goals'}
        new_cols = {}
        unique_teams = pd.concat([df['home_team'], df['away_team']]).unique()

        for col in cols_to_avg:
            new_cols[f"home_avg_{col}_last5"] = pd.Series(dtype=float, index=df.index)
            new_cols[f"away_avg_{col}_last5"] = pd.Series(dtype=float, index=df.index)
            if col in conceded_cols:
                new_cols[f"home_avg_{col}_conceded_last5"] = pd.Series(dtype=float, index=df.index)
                new_cols[f"away_avg_{col}_conceded_last5"] = pd.Series(dtype=float, index=df.index)

        for team in unique_teams:
            mask_h = df['home_team'] == team
            mask_a = df['away_team'] == team
            team_matches = df[mask_h | mask_a].copy()

            opponent_elo = pd.Series(index=team_matches.index, dtype=float)
            opponent_elo.loc[mask_h] = team_matches.loc[mask_h, 'away_elo']
            opponent_elo.loc[mask_a] = team_matches.loc[mask_a, 'home_elo']
            elo_weights = (opponent_elo / 1500).clip(0.8, 1.2)

            for col in cols_to_avg:
                col_h, col_a = f"{col}_home", f"{col}_away"
                if col_h not in df.columns:
                    continue

                # 1. Vlastní statistiky týmu (útok / produkce)
                team_stats = pd.Series(index=team_matches.index, dtype=float)
                team_stats.loc[mask_h] = team_matches.loc[mask_h, col_h]
                team_stats.loc[mask_a] = team_matches.loc[mask_a, col_a]

                weighted_stats = (team_stats * elo_weights).fillna(0)
                roll_5 = weighted_stats.shift(1).rolling(window=5, min_periods=1).mean()

                new_cols[f"home_avg_{col}_last5"].update(roll_5[mask_h])
                new_cols[f"away_avg_{col}_last5"].update(roll_5[mask_a])

                # 2. OPRAVA: Inkasované statistiky (obrana - kolik tým dovolil soupeři)
                if col in conceded_cols:
                    conc_stats = pd.Series(index=team_matches.index, dtype=float)
                    conc_stats.loc[mask_h] = team_matches.loc[mask_h, col_a]
                    conc_stats.loc[mask_a] = team_matches.loc[mask_a, col_h]

                    # Když inkasujeme proti silnému soupeři, váha je mírnější (1500 / opponent_elo)
                    def_weights = (1500 / opponent_elo).clip(0.8, 1.2)
                    weighted_conc = (conc_stats * def_weights).fillna(0)
                    roll_conc_5 = weighted_conc.shift(1).rolling(window=5, min_periods=1).mean()

                    new_cols[f"home_avg_{col}_conceded_last5"].update(roll_conc_5[mask_h])
                    new_cols[f"away_avg_{col}_conceded_last5"].update(roll_conc_5[mask_a])

        new_cols["market_value_home"] = df["home_team"].map(MARKET_VALUES).fillna(200)
        new_cols["market_value_away"] = df["away_team"].map(MARKET_VALUES).fillna(200)
        new_cols["market_value_diff"] = new_cols["market_value_home"] - new_cols["market_value_away"]

        df_new_features = pd.DataFrame(new_cols)
        return pd.concat([df, df_new_features], axis=1)

    def create_enhanced_features(self, df):
        df = df.sort_values("match_date").reset_index(drop=True)
        played_mask = df['goals_home'].notna() & df['goals_away'].notna()

        # Výchozí celoevropské priory (pro prvních pár kol v databázi)
        DEFAULT_GOALS = 2.70
        DEFAULT_DRAW = 0.25
        DEFAULT_HOME_WIN = 0.44
        DEFAULT_BTTS = 0.53

        df['league_avg_goals'] = DEFAULT_GOALS
        df['league_draw_rate'] = DEFAULT_DRAW
        df['league_home_advantage'] = DEFAULT_HOME_WIN
        df['league_btts_rate'] = DEFAULT_BTTS

        # --- VÝPOČET LIGOVÉ DNA (Bez Data Leakage - klouzavé okno posledních 100 zápasů v lize) ---
        if 'league' in df.columns and played_mask.any():
            for lg, group in df.groupby('league'):
                p_idx = group[played_mask.loc[group.index]].index
                if len(p_idx) == 0:
                    continue

                g_home = df.loc[p_idx, 'goals_home']
                g_away = df.loc[p_idx, 'goals_away']

                tot_goals = g_home + g_away
                is_draw = (g_home == g_away).astype(float)
                is_home_w = (g_home > g_away).astype(float)
                is_btts = ((g_home > 0) & (g_away > 0)).astype(float)

                # Historie pro odehrané zápasy (shift(1) zaručuje, že zápas nevidí sám sebe)
                roll_goals = tot_goals.shift(1).rolling(window=100, min_periods=5).mean().fillna(DEFAULT_GOALS)
                roll_draw = is_draw.shift(1).rolling(window=100, min_periods=5).mean().fillna(DEFAULT_DRAW)
                roll_home = is_home_w.shift(1).rolling(window=100, min_periods=5).mean().fillna(DEFAULT_HOME_WIN)
                roll_btts = is_btts.shift(1).rolling(window=100, min_periods=5).mean().fillna(DEFAULT_BTTS)

                df.loc[p_idx, 'league_avg_goals'] = roll_goals
                df.loc[p_idx, 'league_draw_rate'] = roll_draw
                df.loc[p_idx, 'league_home_advantage'] = roll_home
                df.loc[p_idx, 'league_btts_rate'] = roll_btts

                # Pro budoucí zápasy (prepared_fixtures) použijeme nejnovější známou hodnotu ligy
                f_idx = group[~played_mask.loc[group.index]].index
                if len(f_idx) > 0:
                    df.loc[f_idx, 'league_avg_goals'] = float(tot_goals.tail(100).mean()) if len(p_idx) >= 5 else DEFAULT_GOALS
                    df.loc[f_idx, 'league_draw_rate'] = float(is_draw.tail(100).mean()) if len(p_idx) >= 5 else DEFAULT_DRAW
                    df.loc[f_idx, 'league_home_advantage'] = float(is_home_w.tail(100).mean()) if len(p_idx) >= 5 else DEFAULT_HOME_WIN
                    df.loc[f_idx, 'league_btts_rate'] = float(is_btts.tail(100).mean()) if len(p_idx) >= 5 else DEFAULT_BTTS

        # Kompatibilní multiplikátor home_advantage (poměr vůči globálnímu průměru 0.44)
        df['home_advantage'] = (df['league_home_advantage'] / DEFAULT_HOME_WIN).clip(0.75, 1.25)

        if 'home_elo' in df.columns and 'away_elo' in df.columns:
            df['elo_diff'] = df['home_elo'] - df['away_elo']

        if 'market_value_diff' in df.columns:
            scaler = StandardScaler()
            df['market_value_diff_scaled'] = scaler.fit_transform(df[['market_value_diff']])
            joblib.dump(scaler, os.path.join(MODEL_DIR, "market_value_scaler.pkl"))

        if 'elo_diff' in df.columns and 'market_value_diff_scaled' in df.columns:
            df['elo_x_market'] = df['elo_diff'] * df['market_value_diff_scaled']

        if 'elo_diff' in df.columns:
            df['home_x_elo'] = df['home_advantage'] * df['elo_diff']

        df['is_top5_league'] = (df['league'].isin(['PL', 'BL', 'LL', 'SA', 'L1'])).astype(float)
        return df

    def run(self):
        print("\n" + "=" * 70)
        print("⚙️ AGENT TOOL: Budování prepared_datasets a prepared_fixtures (s Ligovou DNA)")
        print("=" * 70)

        with self.engine.begin() as conn:
            df = self.load_data(conn)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            df['match_date'] = pd.to_datetime(df['match_date'])

            if df.empty:
                print("❌ Žádná data v databázi.")
                return

            df = self.normalize_numeric(df)
            df = self.compute_target(df)
            df = self.compute_elo(df)
            df = self.compute_features(df)
            df = self.create_enhanced_features(df)

            today = pd.Timestamp.now().normalize()
            df_historical = df[df['match_date'] < today].copy()

            if EXCLUDE_FROM_TRAINING:
                exclude_mask = pd.Series(False, index=df_historical.index)
                for league, season in EXCLUDE_FROM_TRAINING:
                    exclude_mask |= ((df_historical['league'] == league) & (df_historical['season'] == season))
                df_historical = df_historical[~exclude_mask].copy()

            df_historical = df_historical.fillna(0)
            df_historical = df_historical.loc[:, ~df_historical.columns.duplicated()].copy()

            df_fixtures = df[df['match_date'] >= today].copy()
            df_fixtures = df_fixtures.fillna(0)

            df_historical.to_sql("prepared_datasets", conn, if_exists="replace", index=False)
            df_fixtures.to_sql("prepared_fixtures", conn, if_exists="replace", index=False)

            print(f"   ✅ Tréninková data uložena: {len(df_historical)} řádků (včetně Ligové DNA)")
            print(f"   ✅ Čekárna (prepared_fixtures) naplněna: {len(df_fixtures)} budoucích zápasů")
            print("=" * 70)


def build_datasets():
    builder = DatasetBuilder()
    builder.run()


if __name__ == "__main__":
    build_datasets()