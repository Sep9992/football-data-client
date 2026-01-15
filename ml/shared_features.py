# ml/shared_features.py
# Sjednocený set featur pro trénink i predikci
# VERZE: BEZ DATA LEAKAGE (pouze pre-match features)

performance_features = [
    # --- ELO Metriky (Známé před zápasem) ---
    "elo_home", "elo_away", "elo_diff",
    "opponent_strength_home", "opponent_strength_away",

    # ELO trendy
    "elo_trend_home", "elo_trend_away", "elo_trend_diff",
    "elo_slope_home_last5", "elo_slope_away_last5", "elo_slope_diff_last5",

    # --- Forma a Trendy (Známé z historie) ---
    "form_home", "form_away", "form_diff",
    "home_form_last5", "away_form_last5",
    "home_form_trend", "away_form_trend", "form_trend_diff",
    "win_streak_diff",

    # --- Dlouhodobé Výkonnostní Průměry (Shiftnuté o 1 zápas) ---
    # Win Rates
    "home_win_rate", "away_win_rate", "win_rate_diff",
    "home_win_rate_last5", "away_win_rate_last5", "win_rate_last5_diff",

    # Goals (Průměry vstřelených gólů z minula)
    "home_avg_goals", "away_avg_goals",
    "home_avg_goals_last5", "away_avg_goals_last5", "avg_goals_last5_diff",

    # --- Head-to-Head (Historie vzájemných zápasů) ---
    "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate", "h2h_goal_diff",

    # --- Kontext Zápasu ---
    "home_advantage",
    "rest_days_diff",  # Rozdíl dnů odpočinku

    # NOVÉ FEATURES (Legální statistiky z historie)
    "shots_diff_last5",
    "shots_on_target_diff_last5",
    "corners_diff_last5",
    "fouls_diff_last5",

    # === NOVÉ FEATURY (Volatilita & Únava) ===
    "home_goals_volatility",
    "away_goals_volatility",
    "home_fatigue_index",
    "away_fatigue_index",
    "market_value_diff"

    # ---------------------------------------------------------
    # ❌ ZAKÁZANÉ FEATURE (DATA LEAKAGE) ❌
    # Tyto sloupce obsahují data z aktuálního zápasu.
    # Model je nesmí vidět, jinak bude podvádět (100% accuracy).
    # ---------------------------------------------------------

    # "goal_difference", "xg_diff",
    # "saves_diff", "fouls_diff", "cards_diff", "corners_diff",
    # "pass_accuracy_diff", "passes_diff", "passes_completed_diff",
    # "shots_diff", "shots_on_target_diff",
    # "shots_inside_box_diff", "shots_outside_box_diff", "blocked_shots_diff",
    # "possession_diff",

    # "shooting_accuracy_home", "shooting_accuracy_away", "shooting_accuracy_diff",
    # "passing_efficiency_home", "passing_efficiency_away", "passing_efficiency_diff",
    # "inside_shot_ratio_home", "inside_shot_ratio_away", "inside_shot_ratio_diff",
    # "outside_shot_ratio_home", "outside_shot_ratio_away", "outside_shot_ratio_diff",

    # "attack_index", "defense_index", "discipline_index", # Počítáno z aktuálních střel/karet
    # "elo_change_home", "elo_change_away", # Změna ELO až po zápase
]