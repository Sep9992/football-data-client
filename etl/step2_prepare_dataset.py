# etl/step2_prepare_dataset.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- TRŽNÍ HODNOTY TÝMŮ (v milionech EUR) - Zdroj: Transfermarkt ---
# Aktualizujte dle potřeby. Týmy, které zde nejsou, dostanou průměrnou hodnotu.
MARKET_VALUES = {
    "Manchester City": 1270.0,
    "Arsenal FC": 1310.0,
    "Chelsea FC": 1190.0,
    "Liverpool FC": 1040.0,
    "Manchester United": 719.0,
    "Tottenham Hotspur": 839.0,
    "Newcastle United": 713.0,
    "Aston Villa": 532.0,
    "Brighton & Hove Albion": 527.0,
    "West Ham United": 381.0,
    "Nottingham Forest": 603.0,
    "Brentford": 458.0,
    "Crystal Palace": 529.0,
    "Wolverhampton Wanderers": 353.0,
    "Everton FC": 402.0,
    "Fulham FC": 357.0,
    "AFC Bournemouth": 409.0,
    # Outsiderři / Nováčci (odhad)
    "Leeds United": 313.0,
    "Burnley FC": 252.0,
    "AFC Sunderland": 362.0,
    # Fallback pro ostatní
    "DEFAULT": 200.0
}

# --- načtení .env ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


# --------------------------------
# Načtení odehraných zápasů (+ stats)
# --------------------------------
def load_data(conn):
    return pd.read_sql("""
        SELECT f.id AS fixture_id,
               f.league, f.season, f.match_date,
               f.home_team, f.away_team,

               -- výsledky
               ms.goals_home, ms.goals_away,

               -- possession & passing
               ms.possession_home, ms.possession_away,
               ms.passes_total_home, ms.passes_total_away,
               ms.passes_completed_home, ms.passes_completed_away,
               ms.pass_accuracy_home, ms.pass_accuracy_away,

               -- expected goals
               ms.expected_goals_home, ms.expected_goals_away,

               -- shooting
               ms.shots_home, ms.shots_away,
               ms.shots_on_target_home, ms.shots_on_target_away,
               ms.shots_inside_box_home, ms.shots_inside_box_away,
               ms.shots_outside_box_home, ms.shots_outside_box_away,

               -- defense
               ms.blocked_shots_home, ms.blocked_shots_away,
               ms.saves_home, ms.saves_away,

               -- discipline
               ms.yellow_cards_home, ms.yellow_cards_away,
               ms.red_cards_home, ms.red_cards_away,
               ms.fouls_home, ms.fouls_away,

               -- set pieces
               ms.corners_home, ms.corners_away

        FROM fixtures f
        JOIN match_statistics ms ON ms.fixture_id = f.id
        WHERE f.match_date IS NOT NULL
        ORDER BY f.match_date ASC
    """, conn)


# --------------------------------
# Načtení budoucích zápasů (fixtures)
# --------------------------------
def load_fixtures(conn):
    return pd.read_sql("""
        SELECT id AS fixture_id,
               league, season, match_date,
               home_team, away_team
        FROM fixtures
        WHERE match_date IS NOT NULL
        ORDER BY match_date ASC
    """, conn)


# --------------------------------
# Normalizace numerik
# --------------------------------
def normalize_numeric(df):
    int_cols = [
        "goals_home", "goals_away", "shots_home", "shots_away",
        "shots_on_target_home", "shots_on_target_away",
        "saves_home", "saves_away", "fouls_home", "fouls_away",
        "yellow_cards_home", "yellow_cards_away",
        "red_cards_home", "red_cards_away",
        "corners_home", "corners_away"
    ]
    float_cols = ["expected_goals_home", "expected_goals_away", "pass_accuracy_home", "pass_accuracy_away"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    return df


# --------------------------------
# Target 0/1/2
# --------------------------------
def compute_target(df):
    def result(row):
        if row["goals_home"] > row["goals_away"]:
            return 0  # home win
        elif row["goals_home"] < row["goals_away"]:
            return 2  # away win
        else:
            return 1  # draw

    df["target"] = df.apply(result, axis=1)


# --------------------------------
# ELO s domácí výhodou + dynamický K
# --------------------------------
def compute_elo(df):
    elo_ratings = {}
    default_elo = 1500
    home_advantage = 65
    last_match_date = {}
    match_counter = {}
    win_streaks = {}

    elo_home_list, elo_away_list = [], []
    opponent_strength_home, opponent_strength_away = [], []
    match_number_out, days_since_last_out = [], []
    win_streak_home_out, win_streak_away_out = [], []
    elo_change_home_out, elo_change_away_out = [], []

    def update_elo(r_home, r_away, result, goal_diff, xg_diff, k_dynamic):
        r_home_adj = r_home + home_advantage
        r_away_adj = r_away
        exp_home = 1 / (1 + 10 ** ((r_away_adj - r_home_adj) / 400))
        score_home = 1 if result == 0 else (0.5 if result == 1 else 0)
        margin_factor = 1 + (abs(goal_diff) ** 0.5)
        xg_factor = 1 + (xg_diff / 10.0) if xg_diff is not None else 1
        new_home = r_home + k_dynamic * margin_factor * xg_factor * (score_home - exp_home)
        new_away = r_away + k_dynamic * margin_factor * xg_factor * ((1 - score_home) - (1 - exp_home))
        return new_home, new_away

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        md = pd.to_datetime(row["match_date"])
        tgt = row["target"]

        r_home = elo_ratings.get(home, default_elo)
        r_away = elo_ratings.get(away, default_elo)

        opponent_strength_home.append(r_away)
        opponent_strength_away.append(r_home)

        match_counter[home] = match_counter.get(home, 0) + 1
        match_counter[away] = match_counter.get(away, 0) + 1
        match_number_out.append(max(match_counter[home], match_counter[away]))

        d_home = (md - last_match_date[home]).days if home in last_match_date else 0
        d_away = (md - last_match_date[away]).days if away in last_match_date else 0
        days_since_last_out.append(max(d_home, d_away))
        last_match_date[home] = md
        last_match_date[away] = md

        base_k = 20
        k_home = base_k if d_home == 0 else base_k * (1 + 1 / (1 + d_home))
        k_away = base_k if d_away == 0 else base_k * (1 + 1 / (1 + d_away))
        k_dynamic = (k_home + k_away) / 2.0

        goal_diff = int(row["goals_home"]) - int(row["goals_away"])
        xg_diff = None
        if pd.notna(row["expected_goals_home"]) and pd.notna(row["expected_goals_away"]):
            xg_diff = float(row["expected_goals_home"]) - float(row["expected_goals_away"])

        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

        # --- OPRAVA DATA LEAKAGE (WIN STREAK) ---
        win_streak_home_out.append(win_streaks.get(home, 0))
        win_streak_away_out.append(win_streaks.get(away, 0))

        win_streaks[home] = (win_streaks.get(home, 0) + 1) if tgt == 0 else 0
        win_streaks[away] = (win_streaks.get(away, 0) + 1) if tgt == 2 else 0

        new_home, new_away = update_elo(r_home, r_away, tgt, goal_diff, xg_diff, k_dynamic)
        elo_change_home_out.append(new_home - r_home)
        elo_change_away_out.append(new_away - r_away)

        elo_ratings[home] = new_home
        elo_ratings[away] = new_away

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["opponent_strength_home"] = opponent_strength_home
    df["opponent_strength_away"] = opponent_strength_away
    df["match_number"] = match_number_out
    df["days_since_last_match"] = days_since_last_out
    df["win_streak_home"] = win_streak_home_out
    df["win_streak_away"] = win_streak_away_out
    df["elo_change_home"] = elo_change_home_out
    df["elo_change_away"] = elo_change_away_out


# --------------------------------
# Feature engineering
# --------------------------------
def compute_features(df):
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df_sorted = df.sort_values("match_date")

    df_sorted["home_win"] = (df_sorted["goals_home"] > df_sorted["goals_away"]).astype(int)
    df_sorted["away_win"] = (df_sorted["goals_away"] > df_sorted["goals_home"]).astype(int)
    df_sorted["is_draw"] = (df_sorted["goals_home"] == df_sorted["goals_away"]).astype(int)
    df_sorted["home_points"] = df_sorted["home_win"] * 3 + df_sorted["is_draw"]
    df_sorted["away_points"] = df_sorted["away_win"] * 3 + df_sorted["is_draw"]
    df_sorted["goal_diff_signed"] = df_sorted["goals_home"] - df_sorted["goals_away"]

    new_cols = pd.DataFrame(index=df.index)

    # --- Základní rozdíly (pro trénink, ne pro features!) ---
    new_cols["goal_difference"] = df["goals_home"] - df["goals_away"]
    new_cols["xg_diff"] = df["expected_goals_home"] - df["expected_goals_away"]
    new_cols["saves_diff"] = df["saves_home"] - df["saves_away"]
    new_cols["fouls_diff"] = df["fouls_home"] - df["fouls_away"]
    new_cols["cards_diff"] = (df["yellow_cards_home"] + df["red_cards_home"]) - (
                df["yellow_cards_away"] + df["red_cards_away"])
    new_cols["corners_diff"] = df["corners_home"] - df["corners_away"]
    new_cols["pass_accuracy_diff"] = df["pass_accuracy_home"] - df["pass_accuracy_away"]
    new_cols["shots_diff"] = df["shots_home"] - df["shots_away"]
    new_cols["shots_on_target_diff"] = df["shots_on_target_home"] - df["shots_on_target_away"]
    new_cols["possession_diff"] = df["possession_home"] - df["possession_away"]
    new_cols["passes_diff"] = df["passes_total_home"] - df["passes_total_away"]
    new_cols["passes_completed_diff"] = df["passes_completed_home"] - df["passes_completed_away"]
    new_cols["shots_inside_box_diff"] = df["shots_inside_box_home"] - df["shots_inside_box_away"]
    new_cols["shots_outside_box_diff"] = df["shots_outside_box_home"] - df["shots_outside_box_away"]
    new_cols["blocked_shots_diff"] = df["blocked_shots_home"] - df["blocked_shots_away"]

    new_cols["clean_sheet_home"] = (df["goals_away"] == 0).astype(int)
    new_cols["clean_sheet_away"] = (df["goals_home"] == 0).astype(int)

    new_cols["home_win"] = df_sorted["home_win"].loc[df.index]
    new_cols["away_win"] = df_sorted["away_win"].loc[df.index]
    new_cols["home_points"] = df_sorted["home_points"].loc[df.index]
    new_cols["away_points"] = df_sorted["away_points"].loc[df.index]

    # --- Expanding -> EWM(span=38) pro větší důraz na recenci (shiftnuté) ---
    # Span=38 odpovídá zhruba jedné sezóně PL
    SPAN = 38

    home_avg_goals_sorted = df_sorted.groupby("home_team")["goals_home"].transform(
        lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0))
    away_avg_goals_sorted = df_sorted.groupby("away_team")["goals_away"].transform(
        lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0))
    home_win_rate_sorted = df_sorted.groupby("home_team")["home_win"].transform(
        lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0))
    away_win_rate_sorted = df_sorted.groupby("away_team")["away_win"].transform(
        lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0))

    new_cols["home_avg_goals"] = home_avg_goals_sorted.loc[df.index]
    new_cols["away_avg_goals"] = away_avg_goals_sorted.loc[df.index]
    new_cols["home_win_rate"] = home_win_rate_sorted.loc[df.index]
    new_cols["away_win_rate"] = away_win_rate_sorted.loc[df.index]

    # --- EWM form (span=5) (shiftnuté) ---
    form_home_sorted = df_sorted.groupby("home_team")["home_points"].transform(
        lambda s: s.shift(1).ewm(span=5).mean().fillna(0))
    form_away_sorted = df_sorted.groupby("away_team")["away_points"].transform(
        lambda s: s.shift(1).ewm(span=5).mean().fillna(0))
    new_cols["form_home"] = form_home_sorted.loc[df.index]
    new_cols["form_away"] = form_away_sorted.loc[df.index]
    new_cols["home_form_last5"] = new_cols["form_home"]
    new_cols["away_form_last5"] = new_cols["form_away"]

    # --- Rolling last 5 STATS ---
    stats_to_roll = {
        "shots": ("shots_home", "shots_away"),
        "shots_on_target": ("shots_on_target_home", "shots_on_target_away"),
        "corners": ("corners_home", "corners_away"),
        "fouls": ("fouls_home", "fouls_away")
    }

    # --- NOVINKA: Volatilita výkonu (Standardní odchylka gólů) ---
    # Tým, který má vysokou odchylku, je nepředvídatelný (často kvůli rotaci kádru)
    home_goals_std = df_sorted.groupby("home_team")["goals_home"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std().fillna(0))
    away_goals_std = df_sorted.groupby("away_team")["goals_away"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std().fillna(0))

    new_cols["home_goals_volatility"] = home_goals_std.loc[df.index]
    new_cols["away_goals_volatility"] = away_goals_std.loc[df.index]

    # --- NOVINKA: Market Value Diff (Logaritmický rozdíl) ---
    # Logika: Peníze hrají roli. City (1000M) vs Burnley (200M) je mismatch.
    # Používáme np.log(), protože rozdíl peněz není lineární.

    # Pomocná funkce pro mapování
    def get_mv(team_name):
        return MARKET_VALUES.get(team_name, MARKET_VALUES["DEFAULT"])

    # Mapování hodnot
    home_mv = df["home_team"].apply(get_mv)
    away_mv = df["away_team"].apply(get_mv)

    # Výpočet: ln(Home) - ln(Away)
    # Kladné číslo = Domácí jsou dražší. Záporné = Hosté jsou dražší.
    new_cols["market_value_diff"] = np.log(home_mv) - np.log(away_mv)

    # --- NOVINKA: Hustota zápasů (Fatigue proxy) ---
    # Kolik zápasů odehráli za posledních 30 dní?
    # Toto vyžaduje složitější logiku s daty, zjednodušíme to přes "průměrný rest_days za posledních 5 zápasů"
    home_rest_avg = df.groupby("home_team")["days_since_last_match"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(7)
    )
    away_rest_avg = df.groupby("away_team")["days_since_last_match"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(7)
    )

    new_cols["home_fatigue_index"] = home_rest_avg.loc[df.index]  # Čím nižší číslo, tím větší únava
    new_cols["away_fatigue_index"] = away_rest_avg.loc[df.index]

    for name, (col_h, col_a) in stats_to_roll.items():
        h_mean = df_sorted.groupby("home_team")[col_h].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0))
        a_mean = df_sorted.groupby("away_team")[col_a].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0))

        new_cols[f"home_{name}_last5"] = h_mean.loc[df.index]
        new_cols[f"away_{name}_last5"] = a_mean.loc[df.index]
        new_cols[f"{name}_diff_last5"] = new_cols[f"home_{name}_last5"] - new_cols[f"away_{name}_last5"]

    # --- Recent bonus (shift o 1) ---
    recent_home_bonus_sorted = df_sorted.groupby("home_team")["home_points"].shift(1).fillna(0)
    recent_away_bonus_sorted = df_sorted.groupby("away_team")["away_points"].shift(1).fillna(0)
    new_cols["recent_home_bonus"] = recent_home_bonus_sorted.loc[df.index]
    new_cols["recent_away_bonus"] = recent_away_bonus_sorted.loc[df.index]

    # --- Indexy / Efektivita ---
    new_cols["discipline_index"] = (new_cols["fouls_diff"] + df["yellow_cards_home"] - df["yellow_cards_away"] * 0.5)
    new_cols["attack_index"] = new_cols["corners_diff"] * 0.5 + new_cols["shots_on_target_diff"] + new_cols["xg_diff"]
    new_cols["defense_index"] = new_cols["saves_diff"] + new_cols["clean_sheet_home"] - new_cols["clean_sheet_away"]

    new_cols["shooting_accuracy_home"] = (df["shots_on_target_home"] / df["shots_home"].replace(0, pd.NA)).fillna(0.0)
    new_cols["shooting_accuracy_away"] = (df["shots_on_target_away"] / df["shots_away"].replace(0, pd.NA)).fillna(0.0)
    new_cols["shooting_accuracy_diff"] = new_cols["shooting_accuracy_home"] - new_cols["shooting_accuracy_away"]

    new_cols["passing_efficiency_home"] = (
                df["passes_completed_home"] / df["passes_total_home"].replace(0, pd.NA)).fillna(0.0)
    new_cols["passing_efficiency_away"] = (
                df["passes_completed_away"] / df["passes_total_away"].replace(0, pd.NA)).fillna(0.0)
    new_cols["passing_efficiency_diff"] = new_cols["passing_efficiency_home"] - new_cols["passing_efficiency_away"]

    new_cols["inside_shot_ratio_home"] = (df["shots_inside_box_home"] / df["shots_home"].replace(0, pd.NA)).fillna(0.0)
    new_cols["inside_shot_ratio_away"] = (df["shots_inside_box_away"] / df["shots_away"].replace(0, pd.NA)).fillna(0.0)
    new_cols["inside_shot_ratio_diff"] = new_cols["inside_shot_ratio_home"] - new_cols["inside_shot_ratio_away"]

    new_cols["outside_shot_ratio_home"] = (df["shots_outside_box_home"] / df["shots_home"].replace(0, pd.NA)).fillna(
        0.0)
    new_cols["outside_shot_ratio_away"] = (df["shots_outside_box_away"] / df["shots_away"].replace(0, pd.NA)).fillna(
        0.0)
    new_cols["outside_shot_ratio_diff"] = new_cols["outside_shot_ratio_home"] - new_cols["outside_shot_ratio_away"]

    # --- Nové rozdíly a flagy ---
    new_cols["elo_diff"] = df["elo_home"] - df["elo_away"]
    new_cols["win_streak_diff"] = df["win_streak_home"] - df["win_streak_away"]
    new_cols["form_diff"] = new_cols["form_home"] - new_cols["form_away"]
    new_cols["home_advantage"] = 1

    # --- H2H expanding (shiftnuté, také EWM) ---
    h2h_home_win_rate_sorted = df_sorted.groupby(["home_team", "away_team"])["home_win"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0))
    h2h_away_win_rate_sorted = df_sorted.groupby(["home_team", "away_team"])["away_win"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0))
    h2h_draw_rate_sorted = df_sorted.groupby(["home_team", "away_team"])["is_draw"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0))
    h2h_goal_diff_sorted = df_sorted.groupby(["home_team", "away_team"])["goal_diff_signed"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0))

    new_cols["h2h_home_win_rate"] = h2h_home_win_rate_sorted.loc[df.index]
    new_cols["h2h_away_win_rate"] = h2h_away_win_rate_sorted.loc[df.index]
    new_cols["h2h_draw_rate"] = h2h_draw_rate_sorted.loc[df.index]
    new_cols["h2h_goal_diff"] = h2h_goal_diff_sorted.loc[df.index]

    # --- Rolling last5 general (shiftnuté) ---
    home_win_rate_last5_sorted = df_sorted.groupby("home_team")["home_win"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0))
    away_win_rate_last5_sorted = df_sorted.groupby("away_team")["away_win"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0))
    home_avg_goals_last5_sorted = df_sorted.groupby("home_team")["goals_home"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0))
    away_avg_goals_last5_sorted = df_sorted.groupby("away_team")["goals_away"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0))

    new_cols["home_win_rate_last5"] = home_win_rate_last5_sorted.loc[df.index]
    new_cols["away_win_rate_last5"] = away_win_rate_last5_sorted.loc[df.index]
    new_cols["home_avg_goals_last5"] = home_avg_goals_last5_sorted.loc[df.index]
    new_cols["away_avg_goals_last5"] = away_avg_goals_last5_sorted.loc[df.index]

    # --- Rest days diff (shiftnuté) ---
    last_days_home = df.groupby("home_team")["days_since_last_match"].transform(lambda s: s.shift(1)).fillna(0)
    last_days_away = df.groupby("away_team")["days_since_last_match"].transform(lambda s: s.shift(1)).fillna(0)
    new_cols["rest_days_diff"] = (last_days_home - last_days_away).astype(float)

    # --- NOVINKA: Volatilita výkonu (Standardní odchylka gólů) ---
    # Počítáme z historie (shift 1), okno 10 zápasů pro stabilitu
    home_goals_std = df_sorted.groupby("home_team")["goals_home"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std().fillna(0))
    away_goals_std = df_sorted.groupby("away_team")["goals_away"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std().fillna(0))

    new_cols["home_goals_volatility"] = home_goals_std.loc[df.index]
    new_cols["away_goals_volatility"] = away_goals_std.loc[df.index]

    # --- NOVINKA: Hustota zápasů (Fatigue proxy) ---
    # Průměrný počet dní odpočinku za posledních 5 zápasů
    # (Nízké číslo = málo odpočinku = únava)
    home_rest_avg = df.groupby("home_team")["days_since_last_match"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(7)
    )
    away_rest_avg = df.groupby("away_team")["days_since_last_match"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(7)
    )

    new_cols["home_fatigue_index"] = home_rest_avg.loc[df.index]
    new_cols["away_fatigue_index"] = away_rest_avg.loc[df.index]

    # --- ELO trendy a slope ---
    elo_trend_home_sorted = df_sorted.groupby("home_team")["elo_home"].transform(
        lambda s: s - s.rolling(5, min_periods=1).mean()).fillna(0)
    elo_trend_away_sorted = df_sorted.groupby("away_team")["elo_away"].transform(
        lambda s: s - s.rolling(5, min_periods=1).mean()).fillna(0)

    new_cols["elo_trend_home"] = elo_trend_home_sorted.loc[df.index]
    new_cols["elo_trend_away"] = elo_trend_away_sorted.loc[df.index]

    def calc_slope(y):
        if len(y) < 2: return 0.0
        x = np.arange(len(y))
        try:
            return np.polyfit(x, y, 1)[0]
        except:
            return 0.0

    elo_slope_home_sorted = df_sorted.groupby("home_team")["elo_home"].transform(
        lambda s: s.rolling(5, min_periods=2).apply(calc_slope, raw=True)).fillna(0)
    elo_slope_away_sorted = df_sorted.groupby("away_team")["elo_away"].transform(
        lambda s: s.rolling(5, min_periods=2).apply(calc_slope, raw=True)).fillna(0)

    new_cols["elo_slope_home_last5"] = elo_slope_home_sorted.loc[df.index]
    new_cols["elo_slope_away_last5"] = elo_slope_away_sorted.loc[df.index]

    new_cols["home_form_trend"] = new_cols["home_win_rate_last5"] - new_cols["home_win_rate"]
    new_cols["away_form_trend"] = new_cols["away_win_rate_last5"] - new_cols["away_win_rate"]

    new_cols["elo_trend_diff"] = new_cols["elo_trend_home"] - new_cols["elo_trend_away"]
    new_cols["elo_slope_diff_last5"] = new_cols["elo_slope_home_last5"] - new_cols["elo_slope_away_last5"]
    new_cols["form_trend_diff"] = new_cols["home_form_trend"] - new_cols["away_form_trend"]
    new_cols["win_rate_diff"] = new_cols["home_win_rate"] - new_cols["away_win_rate"]
    new_cols["win_rate_last5_diff"] = new_cols["home_win_rate_last5"] - new_cols["away_win_rate_last5"]
    new_cols["avg_goals_last5_diff"] = new_cols["home_avg_goals_last5"] - new_cols["away_avg_goals_last5"]

    new_cols["attack_index"] = new_cols["attack_index"]
    new_cols["defense_index"] = new_cols["defense_index"]

    for c in new_cols.columns:
        new_cols[c] = new_cols[c].fillna(0)

    df = pd.concat([df, new_cols], axis=1)
    return df


# --------------------------------
# Tabulka prepared_datasets
# --------------------------------
def ensure_table_prepared_datasets(conn):
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prepared_datasets (
            fixture_id INT PRIMARY KEY,
            league TEXT,
            season TEXT,
            match_date DATE,
            home_team TEXT,
            away_team TEXT,

            -- výsledky
            target INT,
            goals_home INT,
            goals_away INT,
            home_win INT,
            away_win INT,
            home_points INT,
            away_points INT,

            -- dlouhodobé průměry
            home_avg_goals FLOAT,
            away_avg_goals FLOAT,
            home_win_rate FLOAT,
            away_win_rate FLOAT,

            -- ELO
            elo_home FLOAT,
            elo_away FLOAT,
            opponent_strength_home FLOAT,
            opponent_strength_away FLOAT,
            match_number INT,
            days_since_last_match INT,
            win_streak_home INT,
            win_streak_away INT,
            elo_change_home FLOAT,
            elo_change_away FLOAT,

            -- rozdílové featury
            goal_difference FLOAT,
            xg_diff FLOAT,
            saves_diff FLOAT,
            fouls_diff FLOAT,
            cards_diff FLOAT,
            corners_diff FLOAT,
            pass_accuracy_diff FLOAT,
            passes_diff FLOAT,
            passes_completed_diff FLOAT,
            shots_diff FLOAT,
            shots_on_target_diff FLOAT,
            shots_inside_box_diff FLOAT,
            shots_outside_box_diff FLOAT,
            blocked_shots_diff FLOAT,

            -- NOVÉ ROLLING FEATURY (LAST 5)
            shots_diff_last5 FLOAT,
            shots_on_target_diff_last5 FLOAT,
            corners_diff_last5 FLOAT,
            fouls_diff_last5 FLOAT,

            -- efficiency ratios
            shooting_accuracy_home FLOAT,
            shooting_accuracy_away FLOAT,
            shooting_accuracy_diff FLOAT,
            passing_efficiency_home FLOAT,
            passing_efficiency_away FLOAT,
            passing_efficiency_diff FLOAT,
            inside_shot_ratio_home FLOAT,
            inside_shot_ratio_away FLOAT,
            inside_shot_ratio_diff FLOAT,
            outside_shot_ratio_home FLOAT,
            outside_shot_ratio_away FLOAT,
            outside_shot_ratio_diff FLOAT,

            -- čisté konto
            clean_sheet_home INT,
            clean_sheet_away INT,

            -- form & trendy
            form_home FLOAT,
            form_away FLOAT,
            form_diff FLOAT,
            home_form_last5 FLOAT,
            away_form_last5 FLOAT,
            recent_home_bonus FLOAT,
            recent_away_bonus FLOAT,
            home_form_trend FLOAT,
            away_form_trend FLOAT,
            form_trend_diff FLOAT,

            -- indexy
            discipline_index FLOAT,
            attack_index FLOAT,
            defense_index FLOAT,

            -- ELO trendy
            elo_diff FLOAT,
            win_streak_diff FLOAT,
            home_advantage FLOAT,
            elo_trend_home FLOAT,
            elo_trend_away FLOAT,
            elo_slope_home_last5 FLOAT,
            elo_slope_away_last5 FLOAT,
            elo_trend_diff FLOAT,
            elo_slope_diff_last5 FLOAT,

            -- head-to-head
            h2h_home_win_rate FLOAT,
            h2h_away_win_rate FLOAT,
            h2h_draw_rate FLOAT,
            h2h_goal_diff FLOAT,

            -- rolling last5
            home_win_rate_last5 FLOAT,
            away_win_rate_last5 FLOAT,
            win_rate_diff FLOAT,
            win_rate_last5_diff FLOAT,
            home_avg_goals_last5 FLOAT,
            away_avg_goals_last5 FLOAT,
            avg_goals_last5_diff FLOAT,

            -- kontext
            rest_days_diff FLOAT,
            
            -- NOVÉ METRIKY (Volatilita a Únava)
            home_goals_volatility FLOAT,
            away_goals_volatility FLOAT,
            home_fatigue_index FLOAT,
            away_fatigue_index FLOAT,
            market_value_diff FLOAT
        )
    """))


def save_dataset(conn, df):
    ensure_table_prepared_datasets(conn)

    cols = [
        "fixture_id", "league", "season", "match_date", "home_team", "away_team",
        "target", "goals_home", "goals_away", "home_win", "away_win",
        "home_points", "away_points",
        "home_avg_goals", "away_avg_goals", "home_win_rate", "away_win_rate",
        "elo_home", "elo_away", "opponent_strength_home", "opponent_strength_away",
        "match_number", "days_since_last_match", "win_streak_home", "win_streak_away",
        "elo_change_home", "elo_change_away",

        "goal_difference", "xg_diff", "saves_diff", "fouls_diff", "cards_diff", "corners_diff",
        "pass_accuracy_diff", "passes_diff", "passes_completed_diff",
        "shots_diff", "shots_on_target_diff", "shots_inside_box_diff", "shots_outside_box_diff", "blocked_shots_diff",

        # NOVÉ
        "shots_diff_last5", "shots_on_target_diff_last5", "corners_diff_last5", "fouls_diff_last5",

        "shooting_accuracy_home", "shooting_accuracy_away", "shooting_accuracy_diff",
        "passing_efficiency_home", "passing_efficiency_away", "passing_efficiency_diff",
        "inside_shot_ratio_home", "inside_shot_ratio_away", "inside_shot_ratio_diff",
        "outside_shot_ratio_home", "outside_shot_ratio_away", "outside_shot_ratio_diff",

        "clean_sheet_home", "clean_sheet_away",
        "form_home", "form_away", "form_diff", "home_form_last5", "away_form_last5",
        "recent_home_bonus", "recent_away_bonus",
        "home_form_trend", "away_form_trend", "form_trend_diff",

        "discipline_index", "attack_index", "defense_index",
        "elo_diff", "win_streak_diff", "home_advantage",
        "elo_trend_home", "elo_trend_away", "elo_slope_home_last5", "elo_slope_away_last5",
        "elo_trend_diff", "elo_slope_diff_last5",

        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate", "h2h_goal_diff",

        "home_win_rate_last5", "away_win_rate_last5", "win_rate_diff", "win_rate_last5_diff",
        "home_avg_goals_last5", "away_avg_goals_last5", "avg_goals_last5_diff",

        "rest_days_diff", "home_goals_volatility", "away_goals_volatility",
        "home_fatigue_index", "away_fatigue_index", "market_value_diff"
    ]

    cols_present = [c for c in cols if c in df.columns]
    out = df[cols_present].copy()
    conn.execute(text("TRUNCATE TABLE prepared_datasets"))
    records = out.to_dict(orient="records")

    if records:
        conn.execute(text(f"""
            INSERT INTO prepared_datasets ({",".join(cols_present)})
            VALUES ({",".join([f":{c}" for c in cols_present])})
        """), records)


# --------------------------------
# Pomocná funkce pro získání posledních statistik týmu
# --------------------------------
def get_last_stats(df_played, team):
    mask = (df_played["home_team"] == team) | (df_played["away_team"] == team)
    matches = df_played[mask]
    if matches.empty: return None

    last_row = matches.iloc[-1]
    is_home = (last_row["home_team"] == team)
    stats = {"last_date": last_row["match_date"]}

    # Metriky (key, col_h, col_a)
    metrics = [
        ("elo", "elo_home", "elo_away"),
        ("form", "form_home", "form_away"),
        ("form_trend", "home_form_trend", "away_form_trend"),
        ("win_rate", "home_win_rate", "away_win_rate"),
        ("win_rate_last5", "home_win_rate_last5", "away_win_rate_last5"),
        ("avg_goals", "home_avg_goals", "away_avg_goals"),
        ("avg_goals_last5", "home_avg_goals_last5", "away_avg_goals_last5"),
        ("elo_trend", "elo_trend_home", "elo_trend_away"),
        ("elo_slope", "elo_slope_home_last5", "elo_slope_away_last5"),
        ("win_streak", "win_streak_home", "win_streak_away"),

        # NOVÉ ROLLING STATS
        ("shots_last5", "home_shots_last5", "away_shots_last5"),
        ("shots_on_target_last5", "home_shots_on_target_last5", "away_shots_on_target_last5"),
        ("corners_last5", "home_corners_last5", "away_corners_last5"),
        ("fouls_last5", "home_fouls_last5", "away_fouls_last5"),

        # Ostatní
        ("attack_index", "attack_index", "attack_index"),
        ("defense_index", "defense_index", "defense_index"),
        ("discipline_index", "discipline_index", "discipline_index"),
        ("match_number", "match_number", "match_number"),
        ("shooting_accuracy", "shooting_accuracy_home", "shooting_accuracy_away"),
        ("passing_efficiency", "passing_efficiency_home", "passing_efficiency_away"),
        ("inside_shot_ratio", "inside_shot_ratio_home", "inside_shot_ratio_away"),
        ("outside_shot_ratio", "outside_shot_ratio_home", "outside_shot_ratio_away"),
        ("goals_volatility", "home_goals_volatility", "away_goals_volatility"),
        ("fatigue_index", "home_fatigue_index", "away_fatigue_index")
    ]

    for key, col_h, col_a in metrics:
        stats[key] = last_row.get(col_h, 0) if is_home else last_row.get(col_a, 0)

    return stats


# --------------------------------
# Featury pro budoucí fixtures
# --------------------------------
def compute_fixture_features(df_fixt, df_played):
    if df_played.empty: return df_fixt
    unique_teams = set(df_played["home_team"]).union(set(df_played["away_team"]))
    team_stats_cache = {t: get_last_stats(df_played, t) for t in unique_teams}
    updated_rows = []

    for idx, row in df_fixt.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["match_date"]
        s_home = team_stats_cache.get(home)
        s_away = team_stats_cache.get(away)

        def get_val(stats, key, default=0):
            return stats[key] if stats and key in stats else default

        row["match_number"] = get_val(s_home, "match_number", 0) + 1

        elo_h, elo_a = get_val(s_home, "elo", 1500), get_val(s_away, "elo", 1500)
        row["elo_home"], row["elo_away"], row["elo_diff"] = elo_h, elo_a, elo_h - elo_a
        row["opponent_strength_home"], row["opponent_strength_away"] = elo_a, elo_h

        form_h, form_a = get_val(s_home, "form", 0), get_val(s_away, "form", 0)
        row["form_home"], row["form_away"], row["form_diff"] = form_h, form_a, form_h - form_a
        row["home_form_trend"], row["away_form_trend"] = get_val(s_home, "form_trend"), get_val(s_away, "form_trend")
        row["form_trend_diff"] = row["home_form_trend"] - row["away_form_trend"]

        row["elo_trend_home"], row["elo_trend_away"] = get_val(s_home, "elo_trend"), get_val(s_away, "elo_trend")
        row["elo_trend_diff"] = row["elo_trend_home"] - row["elo_trend_away"]
        row["elo_slope_home_last5"], row["elo_slope_away_last5"] = get_val(s_home, "elo_slope"), get_val(s_away,
                                                                                                         "elo_slope")
        row["elo_slope_diff_last5"] = row["elo_slope_home_last5"] - row["elo_slope_away_last5"]

        row["home_win_rate"], row["away_win_rate"] = get_val(s_home, "win_rate"), get_val(s_away, "win_rate")
        row["win_rate_diff"] = row["home_win_rate"] - row["away_win_rate"]
        row["home_win_rate_last5"], row["away_win_rate_last5"] = get_val(s_home, "win_rate_last5"), get_val(s_away,
                                                                                                            "win_rate_last5")
        row["win_rate_last5_diff"] = row["home_win_rate_last5"] - row["away_win_rate_last5"]

        row["home_avg_goals"], row["away_avg_goals"] = get_val(s_home, "avg_goals"), get_val(s_away, "avg_goals")
        row["home_avg_goals_last5"], row["away_avg_goals_last5"] = get_val(s_home, "avg_goals_last5"), get_val(s_away,
                                                                                                               "avg_goals_last5")
        row["avg_goals_last5_diff"] = row["home_avg_goals_last5"] - row["away_avg_goals_last5"]

        last_date_h = s_home["last_date"] if s_home and pd.notna(s_home["last_date"]) else pd.Timestamp.min
        last_date_a = s_away["last_date"] if s_away and pd.notna(s_away["last_date"]) else pd.Timestamp.min
        diff_h = (date - last_date_h).days if last_date_h != pd.Timestamp.min else 7
        diff_a = (date - last_date_a).days if last_date_a != pd.Timestamp.min else 7
        row["days_since_last_match"] = max(diff_h, diff_a)
        row["rest_days_diff"] = float(diff_h - diff_a)
        row["home_goals_volatility"] = get_val(s_home, "goals_volatility")
        row["away_goals_volatility"] = get_val(s_away, "goals_volatility")

        # Market Value pro budoucí zápas
        mv_h = MARKET_VALUES.get(home, MARKET_VALUES["DEFAULT"])
        mv_a = MARKET_VALUES.get(away, MARKET_VALUES["DEFAULT"])
        row["market_value_diff"] = np.log(mv_h) - np.log(mv_a)

        # Pro únavu můžeme použít aktuální 'rest_days' jako součást průměru,
        # ale pro zjednodušení převezmeme trend z minula:
        row["home_fatigue_index"] = get_val(s_home, "fatigue_index", 7)
        row["away_fatigue_index"] = get_val(s_away, "fatigue_index", 7)

        # --- NOVÉ ROLLING STATS DIFFS ---
        row["shots_diff_last5"] = get_val(s_home, "shots_last5") - get_val(s_away, "shots_last5")
        row["shots_on_target_diff_last5"] = get_val(s_home, "shots_on_target_last5") - get_val(s_away,
                                                                                               "shots_on_target_last5")
        row["corners_diff_last5"] = get_val(s_home, "corners_last5") - get_val(s_away, "corners_last5")
        row["fouls_diff_last5"] = get_val(s_home, "fouls_last5") - get_val(s_away, "fouls_last5")

        # Ratios & Indexes
        row["shooting_accuracy_home"] = get_val(s_home, "shooting_accuracy")
        row["shooting_accuracy_away"] = get_val(s_away, "shooting_accuracy")
        row["shooting_accuracy_diff"] = row["shooting_accuracy_home"] - row["shooting_accuracy_away"]

        row["passing_efficiency_home"] = get_val(s_home, "passing_efficiency")
        row["passing_efficiency_away"] = get_val(s_away, "passing_efficiency")
        row["passing_efficiency_diff"] = row["passing_efficiency_home"] - row["passing_efficiency_away"]

        row["inside_shot_ratio_home"] = get_val(s_home, "inside_shot_ratio")
        row["inside_shot_ratio_away"] = get_val(s_away, "inside_shot_ratio")
        row["inside_shot_ratio_diff"] = row["inside_shot_ratio_home"] - row["inside_shot_ratio_away"]

        row["outside_shot_ratio_home"] = get_val(s_home, "outside_shot_ratio")
        row["outside_shot_ratio_away"] = get_val(s_away, "outside_shot_ratio")
        row["outside_shot_ratio_diff"] = row["outside_shot_ratio_home"] - row["outside_shot_ratio_away"]

        row["attack_index"] = get_val(s_home, "attack_index") - get_val(s_away, "attack_index")
        row["defense_index"] = get_val(s_home, "defense_index") - get_val(s_away, "defense_index")
        row["discipline_index"] = get_val(s_home, "discipline_index") - get_val(s_away, "discipline_index")

        row["home_advantage"] = 1
        row["win_streak_home"] = get_val(s_home, "win_streak", 0)
        row["win_streak_away"] = get_val(s_away, "win_streak", 0)
        row["win_streak_diff"] = row["win_streak_home"] - row["win_streak_away"]

        h2h_matches = df_played[((df_played["home_team"] == home) & (df_played["away_team"] == away)) |
                                ((df_played["home_team"] == away) & (df_played["away_team"] == home))]
        if not h2h_matches.empty:
            goals_h = h2h_matches.apply(lambda x: x["goals_home"] if x["home_team"] == home else x["goals_away"],
                                        axis=1).mean()
            goals_a = h2h_matches.apply(lambda x: x["goals_away"] if x["home_team"] == home else x["goals_home"],
                                        axis=1).mean()
            wins_h = h2h_matches.apply(lambda x: 1 if (x["home_team"] == home and x["target"] == 0) or (
                        x["away_team"] == home and x["target"] == 2) else 0, axis=1).mean()
            wins_a = h2h_matches.apply(lambda x: 1 if (x["home_team"] == away and x["target"] == 0) or (
                        x["away_team"] == away and x["target"] == 2) else 0, axis=1).mean()
            draws = h2h_matches.apply(lambda x: 1 if x["target"] == 1 else 0, axis=1).mean()

            row["h2h_goal_diff"] = goals_h - goals_a
            row["h2h_home_win_rate"] = wins_h
            row["h2h_away_win_rate"] = wins_a
            row["h2h_draw_rate"] = draws
        else:
            row["h2h_goal_diff"] = 0.0
            row["h2h_home_win_rate"] = 0.0
            row["h2h_away_win_rate"] = 0.0
            row["h2h_draw_rate"] = 0.0

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)


# --------------------------------
# Uložení prepared_fixtures
# --------------------------------
def save_fixtures(conn, df_fixt):
    conn.execute(text("DROP TABLE IF EXISTS prepared_fixtures"))
    ensure_table_prepared_datasets(conn)
    conn.execute(text("CREATE TABLE prepared_fixtures AS SELECT * FROM prepared_datasets WHERE 1=0"))

    cols = [
        "fixture_id", "league", "season", "match_date", "home_team", "away_team",
        "match_number", "days_since_last_match", "win_streak_home", "win_streak_away",
        "home_avg_goals", "away_avg_goals", "home_win_rate", "away_win_rate",
        "elo_home", "elo_away", "opponent_strength_home", "opponent_strength_away",
        "elo_diff", "win_streak_diff", "form_diff", "home_advantage",

        "goal_difference", "xg_diff", "saves_diff", "fouls_diff", "cards_diff", "corners_diff",
        "pass_accuracy_diff", "passes_diff", "passes_completed_diff",
        "shots_diff", "shots_on_target_diff", "shots_inside_box_diff", "shots_outside_box_diff", "blocked_shots_diff",
        "possession_diff",

        # NOVÉ
        "shots_diff_last5", "shots_on_target_diff_last5", "corners_diff_last5", "fouls_diff_last5",

        "shooting_accuracy_home", "shooting_accuracy_away", "shooting_accuracy_diff",
        "passing_efficiency_home", "passing_efficiency_away", "passing_efficiency_diff",
        "inside_shot_ratio_home", "inside_shot_ratio_away", "inside_shot_ratio_diff",
        "outside_shot_ratio_home", "outside_shot_ratio_away", "outside_shot_ratio_diff",

        "form_home", "form_away", "home_form_last5", "away_form_last5",
        "home_form_trend", "away_form_trend", "form_trend_diff",

        "discipline_index", "attack_index", "defense_index",

        "elo_trend_home", "elo_trend_away", "elo_slope_home_last5", "elo_slope_away_last5",
        "elo_trend_diff", "elo_slope_diff_last5",

        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate", "h2h_goal_diff",

        "home_win_rate_last5", "away_win_rate_last5", "win_rate_diff", "win_rate_last5_diff",
        "home_avg_goals_last5", "away_avg_goals_last5", "avg_goals_last5_diff",

        "rest_days_diff", "home_goals_volatility", "away_goals_volatility",
        "home_fatigue_index", "away_fatigue_index", "market_value_diff"
    ]

    cols_present = [c for c in cols if c in df_fixt.columns]
    out = df_fixt[cols_present].copy().fillna(0)
    records = out.to_dict(orient="records")

    if records:
        conn.execute(text(f"""
            INSERT INTO prepared_fixtures ({",".join(cols_present)})
            VALUES ({",".join([f":{c}" for c in cols_present])})
        """), records)


# --------------------------------
# Main
# --------------------------------
def main():
    with engine.begin() as conn:
        df = load_data(conn)
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("match_date").reset_index(drop=True)

        df = normalize_numeric(df)
        compute_target(df)
        compute_elo(df)
        df = compute_features(df)
        save_dataset(conn, df)

        df_fixt = load_fixtures(conn)
        df_fixt["match_date"] = pd.to_datetime(df_fixt["match_date"], errors="coerce")
        if not df.empty:
            last_date = df["match_date"].max()
            df_fixt = df_fixt[df_fixt["match_date"] > last_date]
        df_fixt = df_fixt.reset_index(drop=True)

        df_fixt = compute_fixture_features(df_fixt, df)
        save_fixtures(conn, df_fixt)

    print("✅ step2_prepare_dataset hotovo – prepared_datasets i prepared_fixtures aktualizovány.")


if __name__ == "__main__":
    main()