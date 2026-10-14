"""
team_names.py
Centrální slovník pro normalizaci jmen týmů z Livesportu na jména v naší DB.
"""

TEAM_ALIASES = {
    # --- Premier League ---
    "Manchester Utd":    "Manchester United",
    "Brighton":          "Brighton & Hove Albion",
    "Wolves":            "Wolverhampton Wanderers",
    "Nottingham":        "Nottingham Forest",
    "Bournemouth":       "AFC Bournemouth",
    "Sunderland":        "AFC Sunderland",
    "Newcastle":         "Newcastle United",
    "Tottenham":         "Tottenham Hotspur",
    "West Ham":          "West Ham United",
    "Everton":           "Everton FC",
    "Fulham":            "Fulham FC",
    "Arsenal":           "Arsenal FC",
    "Chelsea":           "Chelsea FC",
    "Liverpool":         "Liverpool FC",
    "Leeds":             "Leeds United",
    "Burnley":           "Burnley FC",
    "Ipswich":           "Ipswich Town",

    # --- La Liga ---
    "Málaga":            "Malaga",

    # Pozice pro další ligy (Sparta Praha, Bayern atd. přidávat nemusíme,
    # pokud jméno na Livesportu PŘESNĚ odpovídá jménu v naší databázi).
}

def normalize_team(team_name: str) -> str:
    """
    Přijme jméno týmu z Livesportu a vrátí jeho oficiální jméno pro databázi.
    Pokud tým nemá alias v TEAM_ALIASES, vrátí ho v původním znění.
    """
    if not team_name:
        return ""

    # Odstraní případné mezery na začátku nebo na konci
    clean_name = team_name.strip()

    # .get(klíč, výchozí_hodnota)
    # Zkusí najít jméno ve slovníku. Pokud ho nenajde, vrátí clean_name beze změny.
    return TEAM_ALIASES.get(clean_name, clean_name)