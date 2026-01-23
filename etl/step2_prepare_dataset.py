# etl/step2_prepare_dataset.py
# FIX: Oprava NULL hodnot (home_win, points, diffs) a chybějících sloupců.
# Tuning 2.1: Split Logic pro Home/Away formu.

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- TRŽNÍ HODNOTY TÝMŮ (v milionech EUR) ---
MARKET_VALUES = {
    "Manchester City": 1270.0, "Arsenal FC": 1310.0, "Chelsea FC": 1190.0,
    "Liverpool FC": 1040.0, "Manchester United": 719.0, "Tottenham Hotspur": 839.0,
    "Newcastle United": 713.0, "Aston Villa": 532.0, "Brighton & Hove Albion": 527.0,
    "West Ham United": 381.0, "Nottingham Forest": 603.0, "Brentford": 458.0,
    "Crystal Palace": 529.0, "Wolverhampton Wanderers": 353.0, "Everton FC": 402.0,
    "Fulham FC": 357.0, "AFC Bournemouth": 409.0,
    "Leeds United": 313.0, "Burnley FC": 252.0, "AFC Sunderland": 362.0,
    "DEFAULT": 200.0
}

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def load_data(conn):
    return pd.read_sql("""
        SELECT f.id AS fixture_id, f.league, f.season, f.match_date, f.home_team, f.away_team,
               ms.goals_home, ms.goals_away,
               ms.possession_home, ms.possession_away,
               ms.passes_total_home, ms.passes_total_away,
               ms.passes_completed_home, ms.passes_completed_away,
               ms.pass_accuracy_home, ms.pass_accuracy_away,
               ms.expected_goals_home, ms.expected_goals_away,
               ms.shots_home, ms.shots_away,
               ms.shots_on_target_home, ms.shots_on_target_away,
               ms.shots_inside_box_home, ms.shots_inside_box_away,
               ms.shots_outside_box_home, ms.shots_outside_box_away,
               ms.blocked_shots_home, ms.blocked_shots_away,
               ms.saves_home, ms.saves_away,
               ms.yellow_cards_home, ms.yellow_cards_away,
               ms.red_cards_home, ms.red_cards_away,
               ms.fouls_home, ms.fouls_away,
               ms.corners_home, ms.corners_away
        FROM fixtures f
        JOIN match_statistics ms ON ms.fixture_id = f.id
        WHERE f.match_date IS NOT NULL
        ORDER BY f.match_date ASC
    """, conn)


def load_fixtures(conn):
    return pd.read_sql("""
        SELECT id AS fixture_id, league, season, match_date, home_team, away_team
        FROM fixtures
        WHERE match_date IS NOT NULL
        ORDER BY match_date ASC
    """, conn)


def normalize_numeric(df):
    int_cols = ["goals_home", "goals_away", "shots_home", "shots_away",
                "shots_on_target_home", "shots_on_target_away", "saves_home", "saves_away",
                "fouls_home", "fouls_away", "yellow_cards_home", "yellow_cards_away",
                "red_cards_home", "red_cards_away", "corners_home", "corners_away"]
    float_cols = ["expected_goals_home", "expected_goals_away", "pass_accuracy_home", "pass_accuracy_away",
                  "possession_home", "possession_away"]

    for c in int_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    return df


def compute_target(df):
    def result(row):
        if row["goals_home"] > row["goals_away"]:
            return 0
        elif row["goals_home"] < row["goals_away"]:
            return 2
        else:
            return 1

    df["target"] = df.apply(result, axis=1)


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
        xg_diff = float(row["expected_goals_home"]) - float(row["expected_goals_away"]) if pd.notna(
            row["expected_goals_home"]) else None

        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

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


def compute_features(df):
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df_sorted = df.sort_values("match_date")

    # --- VÝPOČET VÝSLEDKŮ (ZÁKLAD) ---
    new_cols = pd.DataFrame(index=df.index)

    new_cols["home_win"] = (df["goals_home"] > df["goals_away"]).astype(int)
    new_cols["away_win"] = (df["goals_away"] > df["goals_home"]).astype(int)
    new_cols["is_draw"] = (df["goals_home"] == df["goals_away"]).astype(int)

    new_cols["home_points"] = new_cols["home_win"] * 3 + new_cols["is_draw"]
    new_cols["away_points"] = new_cols["away_win"] * 3 + new_cols["is_draw"]

    new_cols["goal_difference"] = df["goals_home"] - df["goals_away"]
    new_cols["xg_diff"] = df["expected_goals_home"] - df["expected_goals_away"]
    new_cols["possession_diff"] = df["possession_home"] - df["possession_away"]

    # Základní diffy
    for c in ["saves", "fouls", "corners", "pass_accuracy", "passes_total", "passes_completed", "shots",
              "shots_on_target", "shots_inside_box", "shots_outside_box", "blocked_shots"]:
        col_h, col_a = f"{c}_home", f"{c}_away"
        if col_h in df.columns:
            new_cols[f"{c.replace('_total', '')}_diff"] = df[col_h] - df[col_a]

    new_cols["cards_diff"] = (df["yellow_cards_home"] + df["red_cards_home"]) - (
                df["yellow_cards_away"] + df["red_cards_away"])
    new_cols["discipline_diff"] = new_cols["fouls_diff"] + new_cols["cards_diff"]

    new_cols["clean_sheet_home"] = (df["goals_away"] == 0).astype(int)
    new_cols["clean_sheet_away"] = (df["goals_home"] == 0).astype(int)

    # --- SPECIFICKÁ FORMA (HOME vs AWAY) ---
    # 1. Pomocné sloupce v df_sorted pro výpočty klouzavých průměrů
    df_sorted["temp_home_pts"] = (df_sorted["goals_home"] > df_sorted["goals_away"]).astype(int) * 3 + (
                df_sorted["goals_home"] == df_sorted["goals_away"]).astype(int)
    df_sorted["temp_away_pts"] = (df_sorted["goals_away"] > df_sorted["goals_home"]).astype(int) * 3 + (
                df_sorted["goals_home"] == df_sorted["goals_away"]).astype(int)
    df_sorted["temp_home_win"] = (df_sorted["goals_home"] > df_sorted["goals_away"]).astype(int)
    df_sorted["temp_away_win"] = (df_sorted["goals_away"] > df_sorted["goals_home"]).astype(int)
    # OPRAVA: Přidán chybějící sloupec 'is_draw' do df_sorted
    df_sorted["is_draw"] = (df_sorted["goals_home"] == df_sorted["goals_away"]).astype(int)

    SPAN = 38
    # Home Stats (pouze z domácích zápasů)
    new_cols["home_avg_goals"] = \
    df_sorted.groupby("home_team")["goals_home"].transform(lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0)).loc[
        df.index]
    new_cols["home_win_rate"] = \
    df_sorted.groupby("home_team")["temp_home_win"].transform(lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0)).loc[
        df.index]
    new_cols["form_home"] = \
    df_sorted.groupby("home_team")["temp_home_pts"].transform(lambda s: s.shift(1).ewm(span=5).mean().fillna(0)).loc[
        df.index]

    # Away Stats (pouze z venkovních zápasů)
    new_cols["away_avg_goals"] = \
    df_sorted.groupby("away_team")["goals_away"].transform(lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0)).loc[
        df.index]
    new_cols["away_win_rate"] = \
    df_sorted.groupby("away_team")["temp_away_win"].transform(lambda s: s.shift(1).ewm(span=SPAN).mean().fillna(0)).loc[
        df.index]
    new_cols["form_away"] = \
    df_sorted.groupby("away_team")["temp_away_pts"].transform(lambda s: s.shift(1).ewm(span=5).mean().fillna(0)).loc[
        df.index]

    new_cols["home_form_last5"] = new_cols["form_home"]
    new_cols["away_form_last5"] = new_cols["form_away"]

    # Rolling Stats Last 5
    stats_to_roll = {"shots": "shots", "shots_on_target": "shots_on_target", "corners": "corners", "fouls": "fouls"}
    for name, base in stats_to_roll.items():
        new_cols[f"home_{name}_last5"] = df_sorted.groupby("home_team")[f"{base}_home"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0)).loc[df.index]
        new_cols[f"away_{name}_last5"] = df_sorted.groupby("away_team")[f"{base}_away"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(0)).loc[df.index]
        new_cols[f"{name}_diff_last5"] = new_cols[f"home_{name}_last5"] - new_cols[f"away_{name}_last5"]

    # Volatilita & Fatigue
    new_cols["home_goals_volatility"] = df_sorted.groupby("home_team")["goals_home"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std().fillna(0)).loc[df.index]
    new_cols["away_goals_volatility"] = df_sorted.groupby("away_team")["goals_away"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std().fillna(0)).loc[df.index]

    home_rest_avg = df.groupby("home_team")["days_since_last_match"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(7))
    away_rest_avg = df.groupby("away_team")["days_since_last_match"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean().fillna(7))
    new_cols["home_fatigue_index"] = home_rest_avg.loc[df.index]
    new_cols["away_fatigue_index"] = away_rest_avg.loc[df.index]

    # Market Value
    def get_mv(t):
        return MARKET_VALUES.get(t, MARKET_VALUES["DEFAULT"])

    new_cols["market_value_diff"] = np.log(df["home_team"].apply(get_mv)) - np.log(df["away_team"].apply(get_mv))

    # Bonus & Indexy
    new_cols["recent_home_bonus"] = df_sorted.groupby("home_team")["temp_home_pts"].shift(1).fillna(0).loc[df.index]
    new_cols["recent_away_bonus"] = df_sorted.groupby("away_team")["temp_away_pts"].shift(1).fillna(0).loc[df.index]

    new_cols["discipline_index"] = (
                new_cols["fouls_diff_last5"] + df["yellow_cards_home"] - df["yellow_cards_away"] * 0.5)
    new_cols["attack_index"] = new_cols["corners_diff_last5"] * 0.5 + new_cols["shots_on_target_diff_last5"]
    new_cols["defense_index"] = new_cols.get("saves_diff", 0) + new_cols["clean_sheet_home"] - new_cols[
        "clean_sheet_away"]

    # Efficiency
    for side in ["home", "away"]:
        shots = df[f"shots_{side}"].replace(0, pd.NA)
        passes = df[f"passes_total_{side}"].replace(0, pd.NA)
        new_cols[f"shooting_accuracy_{side}"] = (df[f"shots_on_target_{side}"] / shots).fillna(0.0)
        new_cols[f"passing_efficiency_{side}"] = (df[f"passes_completed_{side}"] / passes).fillna(0.0)
        new_cols[f"inside_shot_ratio_{side}"] = (df[f"shots_inside_box_{side}"] / shots).fillna(0.0)
        new_cols[f"outside_shot_ratio_{side}"] = (df[f"shots_outside_box_{side}"] / shots).fillna(0.0)

    for m in ["shooting_accuracy", "passing_efficiency", "inside_shot_ratio", "outside_shot_ratio"]:
        new_cols[f"{m}_diff"] = new_cols[f"{m}_home"] - new_cols[f"{m}_away"]

    # ELO & Trends
    new_cols["elo_diff"] = df["elo_home"] - df["elo_away"]
    new_cols["win_streak_diff"] = df["win_streak_home"] - df["win_streak_away"]
    new_cols["form_diff"] = new_cols["form_home"] - new_cols["form_away"]
    new_cols["home_advantage"] = 1

    # H2H - Nyní už 'is_draw' v df_sorted existuje
    new_cols["h2h_home_win_rate"] = df_sorted.groupby(["home_team", "away_team"])["temp_home_win"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0)).loc[df.index]
    new_cols["h2h_away_win_rate"] = df_sorted.groupby(["home_team", "away_team"])["temp_away_win"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0)).loc[df.index]
    new_cols["h2h_draw_rate"] = df_sorted.groupby(["home_team", "away_team"])["is_draw"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0)).loc[df.index]

    df_sorted["temp_goal_diff"] = df_sorted["goals_home"] - df_sorted["goals_away"]
    new_cols["h2h_goal_diff"] = df_sorted.groupby(["home_team", "away_team"])["temp_goal_diff"].transform(
        lambda s: s.shift(1).ewm(span=20).mean().fillna(0)).loc[df.index]

    # Last 5 specific
    new_cols["home_win_rate_last5"] = \
    df_sorted.groupby("home_team")["temp_home_win"].transform(lambda s: s.shift(1).rolling(5).mean().fillna(0)).loc[
        df.index]
    new_cols["away_win_rate_last5"] = \
    df_sorted.groupby("away_team")["temp_away_win"].transform(lambda s: s.shift(1).rolling(5).mean().fillna(0)).loc[
        df.index]
    new_cols["home_avg_goals_last5"] = \
    df_sorted.groupby("home_team")["goals_home"].transform(lambda s: s.shift(1).rolling(5).mean().fillna(0)).loc[
        df.index]
    new_cols["away_avg_goals_last5"] = \
    df_sorted.groupby("away_team")["goals_away"].transform(lambda s: s.shift(1).rolling(5).mean().fillna(0)).loc[
        df.index]

    new_cols["win_rate_diff"] = new_cols["home_win_rate"] - new_cols["away_win_rate"]
    new_cols["win_rate_last5_diff"] = new_cols["home_win_rate_last5"] - new_cols["away_win_rate_last5"]
    new_cols["avg_goals_last5_diff"] = new_cols["home_avg_goals_last5"] - new_cols["away_avg_goals_last5"]

    # ELO Trends
    new_cols["elo_trend_home"] = \
    df_sorted.groupby("home_team")["elo_home"].transform(lambda s: s - s.rolling(5).mean()).fillna(0).loc[df.index]
    new_cols["elo_trend_away"] = \
    df_sorted.groupby("away_team")["elo_away"].transform(lambda s: s - s.rolling(5).mean()).fillna(0).loc[df.index]
    new_cols["elo_trend_diff"] = new_cols["elo_trend_home"] - new_cols["elo_trend_away"]

    def calc_slope(y):
        if len(y) < 2: return 0.0
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    new_cols["elo_slope_home_last5"] = \
    df_sorted.groupby("home_team")["elo_home"].transform(lambda s: s.rolling(5).apply(calc_slope, raw=True)).fillna(
        0).loc[df.index]
    new_cols["elo_slope_away_last5"] = \
    df_sorted.groupby("away_team")["elo_away"].transform(lambda s: s.rolling(5).apply(calc_slope, raw=True)).fillna(
        0).loc[df.index]
    new_cols["elo_slope_diff_last5"] = new_cols["elo_slope_home_last5"] - new_cols["elo_slope_away_last5"]

    new_cols["home_form_trend"] = new_cols["home_win_rate_last5"] - new_cols["home_win_rate"]
    new_cols["away_form_trend"] = new_cols["away_win_rate_last5"] - new_cols["away_win_rate"]
    new_cols["form_trend_diff"] = new_cols["home_form_trend"] - new_cols["away_form_trend"]

    # Rest days
    last_days_home = df.groupby("home_team")["days_since_last_match"].transform(lambda s: s.shift(1)).fillna(0)
    last_days_away = df.groupby("away_team")["days_since_last_match"].transform(lambda s: s.shift(1)).fillna(0)
    new_cols["rest_days_diff"] = (last_days_home - last_days_away).astype(float)

    # FINAL CLEANUP: Ensure NO NaN values
    new_cols = new_cols.fillna(0)
    return pd.concat([df, new_cols], axis=1)

def ensure_table_prepared_datasets(conn):
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prepared_datasets (
            fixture_id INT PRIMARY KEY, league TEXT, season TEXT, match_date DATE, home_team TEXT, away_team TEXT,
            target INT, goals_home INT, goals_away INT, home_win INT, away_win INT, home_points INT, away_points INT,
            home_avg_goals FLOAT, away_avg_goals FLOAT, home_win_rate FLOAT, away_win_rate FLOAT,
            elo_home FLOAT, elo_away FLOAT, opponent_strength_home FLOAT, opponent_strength_away FLOAT,
            match_number INT, days_since_last_match INT, win_streak_home INT, win_streak_away INT,
            elo_change_home FLOAT, elo_change_away FLOAT,
            goal_difference FLOAT, xg_diff FLOAT, possession_diff FLOAT, saves_diff FLOAT, fouls_diff FLOAT, 
            discipline_diff FLOAT, cards_diff FLOAT, corners_diff FLOAT,
            pass_accuracy_diff FLOAT, passes_diff FLOAT, passes_completed_diff FLOAT, shots_diff FLOAT,
            shots_on_target_diff FLOAT, shots_inside_box_diff FLOAT, shots_outside_box_diff FLOAT, blocked_shots_diff FLOAT,
            shots_diff_last5 FLOAT, shots_on_target_diff_last5 FLOAT, corners_diff_last5 FLOAT, fouls_diff_last5 FLOAT,
            shooting_accuracy_home FLOAT, shooting_accuracy_away FLOAT, shooting_accuracy_diff FLOAT,
            passing_efficiency_home FLOAT, passing_efficiency_away FLOAT, passing_efficiency_diff FLOAT,
            inside_shot_ratio_home FLOAT, inside_shot_ratio_away FLOAT, inside_shot_ratio_diff FLOAT,
            outside_shot_ratio_home FLOAT, outside_shot_ratio_away FLOAT, outside_shot_ratio_diff FLOAT,
            clean_sheet_home INT, clean_sheet_away INT,
            form_home FLOAT, form_away FLOAT, form_diff FLOAT, home_form_last5 FLOAT, away_form_last5 FLOAT,
            recent_home_bonus FLOAT, recent_away_bonus FLOAT, home_form_trend FLOAT, away_form_trend FLOAT, form_trend_diff FLOAT,
            discipline_index FLOAT, attack_index FLOAT, defense_index FLOAT,
            elo_diff FLOAT, win_streak_diff FLOAT, home_advantage FLOAT,
            elo_trend_home FLOAT, elo_trend_away FLOAT, elo_slope_home_last5 FLOAT, elo_slope_away_last5 FLOAT,
            elo_trend_diff FLOAT, elo_slope_diff_last5 FLOAT,
            h2h_home_win_rate FLOAT, h2h_away_win_rate FLOAT, h2h_draw_rate FLOAT, h2h_goal_diff FLOAT,
            home_win_rate_last5 FLOAT, away_win_rate_last5 FLOAT, win_rate_diff FLOAT, win_rate_last5_diff FLOAT,
            home_avg_goals_last5 FLOAT, away_avg_goals_last5 FLOAT, avg_goals_last5_diff FLOAT,
            rest_days_diff FLOAT, home_goals_volatility FLOAT, away_goals_volatility FLOAT,
            home_fatigue_index FLOAT, away_fatigue_index FLOAT, market_value_diff FLOAT
        )
    """))


def save_dataset(conn, df):
    ensure_table_prepared_datasets(conn)
    cols = [
        "fixture_id", "league", "season", "match_date", "home_team", "away_team", "target", "goals_home", "goals_away",
        "home_win", "away_win", "home_points", "away_points", "home_avg_goals", "away_avg_goals", "home_win_rate",
        "away_win_rate",
        "elo_home", "elo_away", "opponent_strength_home", "opponent_strength_away", "match_number",
        "days_since_last_match",
        "win_streak_home", "win_streak_away", "elo_change_home", "elo_change_away",
        "goal_difference", "xg_diff", "possession_diff", "saves_diff", "fouls_diff", "discipline_diff", "cards_diff",
        "corners_diff",
        "pass_accuracy_diff", "passes_diff", "passes_completed_diff", "shots_diff",
        "shots_on_target_diff", "shots_inside_box_diff", "shots_outside_box_diff", "blocked_shots_diff",
        "shots_diff_last5",
        "shots_on_target_diff_last5", "corners_diff_last5", "fouls_diff_last5", "shooting_accuracy_home",
        "shooting_accuracy_away",
        "shooting_accuracy_diff", "passing_efficiency_home", "passing_efficiency_away", "passing_efficiency_diff",
        "inside_shot_ratio_home", "inside_shot_ratio_away", "inside_shot_ratio_diff", "outside_shot_ratio_home",
        "outside_shot_ratio_away",
        "outside_shot_ratio_diff", "clean_sheet_home", "clean_sheet_away", "form_home", "form_away", "form_diff",
        "home_form_last5",
        "away_form_last5", "recent_home_bonus", "recent_away_bonus", "home_form_trend", "away_form_trend",
        "form_trend_diff",
        "discipline_index", "attack_index", "defense_index", "elo_diff", "win_streak_diff", "home_advantage",
        "elo_trend_home",
        "elo_trend_away", "elo_slope_home_last5", "elo_slope_away_last5", "elo_trend_diff", "elo_slope_diff_last5",
        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate", "h2h_goal_diff", "home_win_rate_last5",
        "away_win_rate_last5",
        "win_rate_diff", "win_rate_last5_diff", "home_avg_goals_last5", "away_avg_goals_last5", "avg_goals_last5_diff",
        "rest_days_diff", "home_goals_volatility", "away_goals_volatility", "home_fatigue_index", "away_fatigue_index",
        "market_value_diff"
    ]

    # CRITICAL: Fillna(0) to prevent NULLs in DB
    cols_present = [c for c in cols if c in df.columns]
    out = df[cols_present].copy().fillna(0)

    conn.execute(text("TRUNCATE TABLE prepared_datasets"))
    records = out.to_dict(orient="records")
    if records:
        conn.execute(text(
            f"INSERT INTO prepared_datasets ({','.join(cols_present)}) VALUES ({','.join([f':{c}' for c in cols_present])})"),
                     records)


def get_team_state_split(df_played, team):
    # 1. GLOBAL STATS (z absolutně posledního zápasu)
    team_matches = df_played[(df_played["home_team"] == team) | (df_played["away_team"] == team)]
    if team_matches.empty: return None
    last_global = team_matches.iloc[-1]
    is_home_global = (last_global["home_team"] == team)

    global_stats = {
        "last_date": last_global["match_date"],
        "elo": last_global["elo_home"] if is_home_global else last_global["elo_away"],
        "elo_trend": last_global["elo_trend_home"] if is_home_global else last_global["elo_trend_away"],
        "elo_slope": last_global["elo_slope_home_last5"] if is_home_global else last_global["elo_slope_away_last5"],
        "win_streak": last_global["win_streak_home"] if is_home_global else last_global["win_streak_away"],
        "fatigue_index": last_global["home_fatigue_index"] if is_home_global else last_global["away_fatigue_index"],
        "match_number": last_global["match_number"]
    }

    # 2. SPECIFIC HOME STATS (z posledního zápasu DOMA)
    home_matches = df_played[df_played["home_team"] == team]
    if not home_matches.empty:
        last_home = home_matches.iloc[-1]
        home_specific = {
            "form": last_home["form_home"],
            "avg_goals": last_home["home_avg_goals"],
            "avg_goals_last5": last_home["home_avg_goals_last5"],
            "win_rate": last_home["home_win_rate"],
            "win_rate_last5": last_home["home_win_rate_last5"],
            "form_trend": last_home["home_form_trend"],
            "shots_last5": last_home["home_shots_last5"],
            "shots_on_target_last5": last_home["home_shots_on_target_last5"],
            "corners_last5": last_home["home_corners_last5"],
            "fouls_last5": last_home["home_fouls_last5"],
            "goals_volatility": last_home["home_goals_volatility"],

            "attack_index": last_home["attack_index"],
            "defense_index": last_home["defense_index"],
            "discipline_index": last_home["discipline_index"],

            "shooting_accuracy": last_home["shooting_accuracy_home"],
            "passing_efficiency": last_home["passing_efficiency_home"],
            "inside_shot_ratio": last_home["inside_shot_ratio_home"],
            "outside_shot_ratio": last_home["outside_shot_ratio_home"]
        }
    else:
        # Fallback
        home_specific = {k: 0 for k in
                         ["form", "avg_goals", "avg_goals_last5", "win_rate", "win_rate_last5", "form_trend",
                          "shots_last5", "shots_on_target_last5", "corners_last5", "fouls_last5", "goals_volatility",
                          "attack_index", "defense_index", "discipline_index", "shooting_accuracy",
                          "passing_efficiency", "inside_shot_ratio", "outside_shot_ratio"]}

    # 3. SPECIFIC AWAY STATS (z posledního zápasu VENKU)
    away_matches = df_played[df_played["away_team"] == team]
    if not away_matches.empty:
        last_away = away_matches.iloc[-1]
        away_specific = {
            "form": last_away["form_away"],
            "avg_goals": last_away["away_avg_goals"],
            "avg_goals_last5": last_away["away_avg_goals_last5"],
            "win_rate": last_away["away_win_rate"],
            "win_rate_last5": last_away["away_win_rate_last5"],
            "form_trend": last_away["away_form_trend"],
            "shots_last5": last_away["away_shots_last5"],
            "shots_on_target_last5": last_away["away_shots_on_target_last5"],
            "corners_last5": last_away["away_corners_last5"],
            "fouls_last5": last_away["away_fouls_last5"],
            "goals_volatility": last_away["away_goals_volatility"],

            "attack_index": last_away["attack_index"],
            "defense_index": last_away["defense_index"],
            "discipline_index": last_away["discipline_index"],

            "shooting_accuracy": last_away["shooting_accuracy_away"],
            "passing_efficiency": last_away["passing_efficiency_away"],
            "inside_shot_ratio": last_away["inside_shot_ratio_away"],
            "outside_shot_ratio": last_away["outside_shot_ratio_away"]
        }
    else:
        away_specific = {k: 0 for k in home_specific.keys()}

    return {"global": global_stats, "home": home_specific, "away": away_specific}


def compute_fixture_features(df_fixt, df_played):
    if df_played.empty: return df_fixt
    unique_teams = set(df_played["home_team"]).union(set(df_played["away_team"]))
    team_stats_cache = {t: get_team_state_split(df_played, t) for t in unique_teams}
    updated_rows = []

    for idx, row in df_fixt.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["match_date"]

        state_h = team_stats_cache.get(home)
        state_a = team_stats_cache.get(away)

        if not state_h or not state_a:
            updated_rows.append(row)
            continue

        row["match_number"] = state_h["global"]["match_number"] + 1

        elo_h = state_h["global"]["elo"]
        elo_a = state_a["global"]["elo"]
        row["elo_home"], row["elo_away"], row["elo_diff"] = elo_h, elo_a, elo_h - elo_a
        row["opponent_strength_home"], row["opponent_strength_away"] = elo_a, elo_h

        row["elo_trend_home"], row["elo_trend_away"] = state_h["global"]["elo_trend"], state_a["global"]["elo_trend"]
        row["elo_trend_diff"] = row["elo_trend_home"] - row["elo_trend_away"]

        row["elo_slope_home_last5"], row["elo_slope_away_last5"] = state_h["global"]["elo_slope"], state_a["global"][
            "elo_slope"]
        row["elo_slope_diff_last5"] = row["elo_slope_home_last5"] - row["elo_slope_away_last5"]

        row["win_streak_home"], row["win_streak_away"] = state_h["global"]["win_streak"], state_a["global"][
            "win_streak"]
        row["win_streak_diff"] = row["win_streak_home"] - row["win_streak_away"]

        s_h = state_h["home"]
        s_a = state_a["away"]

        form_h, form_a = s_h["form"], s_a["form"]
        row["form_home"], row["form_away"], row["form_diff"] = form_h, form_a, form_h - form_a
        row["home_form_last5"], row["away_form_last5"] = form_h, form_a

        row["home_form_trend"], row["away_form_trend"] = s_h["form_trend"], s_a["form_trend"]
        row["form_trend_diff"] = row["home_form_trend"] - row["away_form_trend"]

        row["home_win_rate"], row["away_win_rate"] = s_h["win_rate"], s_a["win_rate"]
        row["win_rate_diff"] = row["home_win_rate"] - row["away_win_rate"]

        row["home_win_rate_last5"], row["away_win_rate_last5"] = s_h["win_rate_last5"], s_a["win_rate_last5"]
        row["win_rate_last5_diff"] = row["home_win_rate_last5"] - row["away_win_rate_last5"]

        row["home_avg_goals"], row["away_avg_goals"] = s_h["avg_goals"], s_a["avg_goals"]
        row["home_avg_goals_last5"], row["away_avg_goals_last5"] = s_h["avg_goals_last5"], s_a["avg_goals_last5"]
        row["avg_goals_last5_diff"] = row["home_avg_goals_last5"] - row["away_avg_goals_last5"]

        row["home_goals_volatility"], row["away_goals_volatility"] = s_h["goals_volatility"], s_a["goals_volatility"]

        row["shots_diff_last5"] = s_h["shots_last5"] - s_a["shots_last5"]
        row["shots_on_target_diff_last5"] = s_h["shots_on_target_last5"] - s_a["shots_on_target_last5"]
        row["corners_diff_last5"] = s_h["corners_last5"] - s_a["corners_last5"]
        row["fouls_diff_last5"] = s_h["fouls_last5"] - s_a["fouls_last5"]

        row["attack_index"] = (row["corners_diff_last5"] * 0.5) + row["shots_on_target_diff_last5"]
        row["defense_index"] = s_h["defense_index"] - s_a["defense_index"]
        row["discipline_index"] = s_h["discipline_index"] - s_a["discipline_index"]

        row["shooting_accuracy_home"], row["shooting_accuracy_away"] = s_h["shooting_accuracy"], s_a[
            "shooting_accuracy"]
        row["shooting_accuracy_diff"] = row["shooting_accuracy_home"] - row["shooting_accuracy_away"]

        row["passing_efficiency_home"], row["passing_efficiency_away"] = s_h["passing_efficiency"], s_a[
            "passing_efficiency"]
        row["passing_efficiency_diff"] = row["passing_efficiency_home"] - row["passing_efficiency_away"]

        row["inside_shot_ratio_home"], row["inside_shot_ratio_away"] = s_h["inside_shot_ratio"], s_a[
            "inside_shot_ratio"]
        row["inside_shot_ratio_diff"] = row["inside_shot_ratio_home"] - row["inside_shot_ratio_away"]

        row["outside_shot_ratio_home"], row["outside_shot_ratio_away"] = s_h["outside_shot_ratio"], s_a[
            "outside_shot_ratio"]
        row["outside_shot_ratio_diff"] = row["outside_shot_ratio_home"] - row["outside_shot_ratio_away"]

        last_date_h = state_h["global"]["last_date"]
        last_date_a = state_a["global"]["last_date"]
        diff_h = (date - last_date_h).days if pd.notnull(last_date_h) else 7
        diff_a = (date - last_date_a).days if pd.notnull(last_date_a) else 7
        row["days_since_last_match"] = max(diff_h, diff_a)
        row["rest_days_diff"] = float(diff_h - diff_a)

        row["home_fatigue_index"] = state_h["global"]["fatigue_index"]
        row["away_fatigue_index"] = state_a["global"]["fatigue_index"]

        mv_h = MARKET_VALUES.get(home, MARKET_VALUES["DEFAULT"])
        mv_a = MARKET_VALUES.get(away, MARKET_VALUES["DEFAULT"])
        row["market_value_diff"] = np.log(mv_h) - np.log(mv_a)
        row["home_advantage"] = 1

        h2h_matches = df_played[((df_played["home_team"] == home) & (df_played["away_team"] == away)) |
                                ((df_played["home_team"] == away) & (df_played["away_team"] == home))]
        if not h2h_matches.empty:
            wins_h = h2h_matches.apply(lambda x: 1 if (x["home_team"] == home and x["target"] == 0) or (
                        x["away_team"] == home and x["target"] == 2) else 0, axis=1).mean()
            wins_a = h2h_matches.apply(lambda x: 1 if (x["home_team"] == away and x["target"] == 0) or (
                        x["away_team"] == away and x["target"] == 2) else 0, axis=1).mean()
            draws = h2h_matches.apply(lambda x: 1 if x["target"] == 1 else 0, axis=1).mean()

            goals_h_avg = h2h_matches.apply(lambda x: x["goals_home"] if x["home_team"] == home else x["goals_away"],
                                            axis=1).mean()
            goals_a_avg = h2h_matches.apply(lambda x: x["goals_away"] if x["home_team"] == home else x["goals_home"],
                                            axis=1).mean()

            row["h2h_home_win_rate"] = wins_h
            row["h2h_away_win_rate"] = wins_a
            row["h2h_draw_rate"] = draws
            row["h2h_goal_diff"] = goals_h_avg - goals_a_avg
        else:
            row["h2h_home_win_rate"] = 0.0
            row["h2h_away_win_rate"] = 0.0
            row["h2h_draw_rate"] = 0.0
            row["h2h_goal_diff"] = 0.0

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)


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
        "goal_difference", "xg_diff", "possession_diff", "saves_diff", "fouls_diff", "discipline_diff", "cards_diff",
        "corners_diff",
        "pass_accuracy_diff", "passes_diff", "passes_completed_diff",
        "shots_diff", "shots_on_target_diff", "shots_inside_box_diff", "shots_outside_box_diff", "blocked_shots_diff",
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
        conn.execute(text(
            f"INSERT INTO prepared_fixtures ({','.join(cols_present)}) VALUES ({','.join([f':{c}' for c in cols_present])})"),
                     records)


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

    print("✅ step2_prepare_dataset (Robust + Split Logic) hotovo.")


if __name__ == "__main__":
    main()