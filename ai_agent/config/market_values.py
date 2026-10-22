"""
market_values.py
Centrální slovník s odhadovanými tržními hodnotami týmů (v milionech EUR).
"""

MARKET_VALUES = {
    # --- PREMIER LEAGUE ---
    "Manchester City": 1430.0, "Arsenal FC": 1330.0, "Chelsea FC": 1080.0,
    "Liverpool FC": 1030.0, "Manchester United": 924.3, "Tottenham Hotspur": 824.0,
    "Newcastle United": 606.1, "Aston Villa": 572.5, "Brighton & Hove Albion": 603.5,
    "West Ham United": 267.5, "Nottingham Forest": 520.5, "Brentford": 584.8,
    "Crystal Palace": 579.2, "Wolverhampton Wanderers": 256.5, "Everton FC": 394.1,
    "Fulham FC": 396.8, "AFC Bournemouth": 574.6,
    "Leeds United": 407.6, "Burnley FC": 147.8, "AFC Sunderland": 421.6,
    "Coventry": 296.7, "Ipswich Town": 330.6, "Hull": 252.9,

    # --- CHANCE LIGA ---
    "Sparta Praha": 83.2, "Slavia Praha": 102.0, "Viktoria Plzeň": 63.1,
    "Baník Ostrava": 17.1, "Mladá Boleslav": 10.7, "Bohemians": 7.3,
    "Slovácko": 8.2, "Sigma Olomouc": 29.2, "Hradec Králové": 15.1,
    "Slovan Liberec": 19.1, "Teplice": 8.8, "Jablonec": 15.5,
    "Pardubice": 16.8, "Zlín": 6.7, "Karviná": 6.3,
    "České Budějovice": 0.8, "Dukla Praha": 6.5,
    "Zbrojovka Brno": 11.0, "Artis Brno": 14.4,

    # --- BUNDESLIGA ---
    "Bayern": 1040.0, "Leverkusen": 481.0, "Dortmund": 565.0,
    "RB Lipsko": 488.0, "Stuttgart": 380.0, "Frankfurt": 288.0,
    "Brémy": 122.0, "Wolfsburg": 112.0, "Hoffenheim": 281.0,
    "Mohuč": 165.0, "Union Berlín": 116.0, "Augsburg": 178.0,
    "Kolín n. R.": 156.0, "Freiburg": 211.0, "Mönchengladbach": 139.0,
    "Heidenheim": 38.0, "St. Pauli": 48.0, "Hamburk": 153.0,
    "Elversberg": 58.0, "Paderborn": 48.0, "Schalke": 71.0,

    # --- LA LIGA ---
    "Real Madrid": 1460.0, "Barcelona": 1260.0, "Atl. Madrid": 680.0,
    "Ath. Bilbao": 231.0, "Real Sociedad": 278.0, "Villarreal": 332.0,
    "Betis": 255.0, "Sevilla": 169.0, "Girona": 69.0,
    "Valencia": 149.0, "Celta Vigo": 173.0, "Osasuna": 78.0,
    "Mallorca": 50.0, "Vallecano": 87.0, "Getafe": 92.0,
    "Espanyol": 128.0, "Alavés": 67.0, "Levante": 87.0,
    "Elche": 74.0, "Oviedo": 35.0, "Malaga": 55.0,
    "Racing Santander": 108.0, "A Coruňa": 116.0,

    # --- SERIE A ---
    "Inter": 730.0, "Juventus": 619.0, "Neapol": 433.0,
    "AC Milán": 474.0, "Atalanta": 436.0, "AS Řím": 485.0,
    "Lazio": 256.0, "Fiorentina": 323.0, "Bologna": 220.0,
    "Turín FC": 177.0, "Udinese": 151.0, "FC Janov": 169.0,
    "Cagliari": 121.0, "Parma": 149.0, "Como": 535.0,
    "Verona": 49.0, "Lecce": 83.0, "Sassuolo": 199.0,
    "Cremonese": 50.0, "Pisa": 64.0, "Monza": 104.0,
    "Frosinone": 94.0, "Venezia": 116.0,

    # --- LIGUE 1 ---
    "PSG": 1360.0,  "Štrasburk": 319.9, "Lyon": 264.5,
    "Rennes": 253.4, "Lille": 253.3, "Lens": 243.0,
    "Paris FC": 191.6, "Marseille": 188.8, "Nice": 150.3,
    "Toulouse": 123.9, "Lorient": 99.1, "Auxerre": 96.0,
    "Brest": 87.8, "Angers": 65.5, "Le Havre": 62.3,
    "Le Mans": 33.5, "Troyes": 43.3, "Monako": 377.7,

    # --- EREDIVISIE ---
    "PSV": 360.5, "Feyenoord": 245.2, "Ajax": 228.5,
    "Alkmaar": 161.5, "Twente": 101.3, "Utrecht": 64.9,
    "Nijmegen": 58.2, "Heerenveen": 57.0, "Groningen": 47.4,
    "Sparta Rotterdam": 32.0, "Zwolle": 25.3, "G.A. Eagles": 24.9,
    "Willem II": 23.8, "Excelsior": 21.1, "Den Haag": 21.0,
    "Sittard": 30.6, "Cambuur": 15.2, "Telstar": 9.4,

    # --- LIGA PORTUGAL ---
    "FC Porto": 471.5, "Benfica": 422.0, "Sporting": 413.7,
    "Braga": 158.6, "Famalicao": 69.6, "Guimaraes": 59.4,
    "Alverca": 49.2, "Estrela": 45.4, "Estoril": 43.9,
    "Rio Ave": 43.6, "Arouca": 31.3, "Casa Pia": 28.2,
    "Moreirense": 28.2, "Gil Vicente": 27.0, "Santa Clara": 22.0,
    "Nacional": 21.8, "Academico Viseu": 20.3, "Maritimo": 17.1,

    # --- JUPILER PRO LEAGUE ---
    "Club Bruggy": 238.2, "Genk": 122.9, "Royale Union SG": 119.4,
    "Anderlecht": 106.1, "Gent": 77.0, "St. Lutych": 67.2,
    "Antverpy": 66.1, "Westerlo": 51.0, "St. Truiden": 43.2,
    "KV Mechelen": 42.0, "Cercle Bruggy": 37.8, "Charleroi": 33.7,
    "Leuven": 33.0, "Waregem": 29.5, "Lommel SK": 27.0,
    "RAAL La Louviere": 23.7, "Kortrijk": 23.1, "Beveren": 19.2
}