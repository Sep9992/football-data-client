# ml/shared_features.py

def get_performance_features(df_columns):
    """
    Dynamicky generuje seznam featur a striktně filtruje DATA LEAKAGE.
    """

    # 1. DEFINICE ZAKÁZANÝCH SLOUPCŮ (Aktuální statistiky zápasu)
    # Tyto sloupce model nesmí nikdy vidět jako vstup (X).
    forbidden_prefixes = [
        "goals_", "expected_goals_", "shots_", "possession_", "passes_",
        "corners_", "yellow_cards_", "red_cards_", "fouls_", "offsides_",
        "saves_", "xgot_", "big_chances_", "box_touches_", "tackles_",
        "interceptions_", "clearances_", "duels_", "crosses_"
    ]

    # 2. SBĚR LEGÁLNÍCH FEATUR (To, co známe před zápasem)
    all_features = []

    for col in df_columns:
        # Povolíme pouze průměry z historie (avg), ELO a statické údaje
        is_avg = col.startswith("home_avg_") or col.startswith("away_avg_")
        is_elo = "elo" in col.lower()
        is_static = col in ["market_value_diff", "home_rest_days", "away_rest_days"]

        if is_avg or is_elo or is_static:
            # Kontrola, zda to náhodou není základní statistika (pro jistotu)
            # Např. 'home_avg_goals_last5' projde, ale 'goals_home' ne.
            if not any(col.startswith(p) for p in forbidden_prefixes):
                all_features.append(col)

    # Odstraníme targety, pokud by se tam náhodou vloudily
    final_features = [f for f in all_features if f not in ["home_win", "target", "fixture_id"]]

    return sorted(list(set(final_features)))


# Ponecháme i původní seznam pro starší verze skriptů
performance_features = [
    "home_elo", "away_elo", "market_value_diff", "home_rest_days", "away_rest_days"
]