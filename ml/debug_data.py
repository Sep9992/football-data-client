import pandas as pd
import os
import unicodedata
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# Zápasy blíž než tolik dnů = skutečná duplicita.
# Dál od sebe = legitimní odveta / česká nadstavba.
DUP_WINDOW_DAYS = 14


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s).lower())
                   if unicodedata.category(c) != 'Mn').strip()


def check_table(name):
    print(f"\n{'=' * 70}")
    print(f"📋 TABULKA: {name}")
    print('=' * 70)

    try:
        df = pd.read_sql(f"SELECT * FROM {name}", engine)
    except Exception as e:
        print(f"   ❌ Nelze načíst: {e}")
        return

    df['match_date'] = pd.to_datetime(df['match_date'])
    print(f"   Řádků: {len(df)}  |  "
          f"Rozsah: {df['match_date'].min().date()} → {df['match_date'].max().date()}")

    if 'home_team' not in df.columns:
        return

    df['_h'] = df['home_team'].map(norm)
    df['_a'] = df['away_team'].map(norm)

    # --- 1. DUPLICITNÍ NÁZVY TÝMŮ ---
    teams = sorted(set(df['home_team'].dropna()) | set(df['away_team'].dropna()))
    groups = {}
    for t in teams:
        groups.setdefault(norm(t), []).append(t)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"\n   👥 Týmů: {len(teams)}  |  po normalizaci: {len(groups)}")
    if dupes:
        print(f"   ⚠️  DUPLICITNÍ NÁZVY ({len(dupes)}):")
        for variants in dupes.values():
            det = ', '.join(
                f"'{v}' ({((df['home_team'] == v) | (df['away_team'] == v)).sum()}x)"
                for v in variants)
            print(f"        {det}")
    else:
        print("   ✅ Žádné duplicitní názvy")

    # --- 2. DUPLICITNÍ ZÁPASY (v časovém okně) ---
    # Stejná dvojice do DUP_WINDOW_DAYS = duplicita.
    # Dál = odveta / nadstavba (legitimní).
    grp = ['season', '_h', '_a'] if 'season' in df.columns else ['_h', '_a']
    real_dupes = []
    legit_repeats = 0

    for keys, g in df.groupby(grp):
        if len(g) < 2:
            continue
        g = g.sort_values('match_date')
        dates = g['match_date'].tolist()
        gap = (max(dates) - min(dates)).days
        if gap <= DUP_WINDOW_DAYS:
            real_dupes.append((keys, g))
        else:
            legit_repeats += 1

    print(f"\n   🔁 Opakované dvojice s odstupem > {DUP_WINDOW_DAYS} dnů: "
          f"{legit_repeats}  (odveta/nadstavba — OK)")

    if not real_dupes:
        print(f"   ✅ Skutečné duplicity (do {DUP_WINDOW_DAYS} dnů): 0")
    else:
        print(f"   ⚠️  SKUTEČNÉ DUPLICITY: {len(real_dupes)}")
        for keys, g in real_dupes[:25]:
            season = f"[{keys[0]}] " if 'season' in df.columns else ""
            print(f"        {season}{len(g)}x:")
            for _, m in g.iterrows():
                print(f"           id={m.get('fixture_id', '?')}  "
                      f"{m['match_date'].date()}  "
                      f"{m['home_team']} vs {m['away_team']}")
        if len(real_dupes) > 25:
            print(f"        ... a dalších {len(real_dupes) - 25}")

    # --- 3. POČET ZÁPASŮ NA TÝM (jen kde je dost týmů na smysluplný medián) ---
    if 'season' in df.columns and 'league' in df.columns:
        long = pd.concat([
            df[['season', 'league', '_h']].rename(columns={'_h': 'team'}),
            df[['season', 'league', '_a']].rename(columns={'_a': 'team'}),
        ])
        per = long.groupby(['season', 'league', 'team']).size().reset_index(name='n')
        per['med'] = per.groupby(['season', 'league'])['n'].transform('median')
        per['tymu'] = per.groupby(['season', 'league'])['team'].transform('size')

        # medián dává smysl jen když má liga rozumný počet týmů a zápasů
        valid = per[(per['tymu'] >= 10) & (per['med'] >= 10)]
        odd = valid[valid['n'] > valid['med']].sort_values('n', ascending=False)
        if not odd.empty:
            print("\n   ⚠️  Týmy s více zápasy než je v lize obvyklé:")
            for _, r in odd.head(10).iterrows():
                print(f"        [{r['season']}] {r['league']:3s} {r['team']:22s} "
                      f"{r['n']} (obvykle {int(r['med'])})")

        # podezřelé sezóny — málo zápasů na tým
        suspect = per[per['med'] < 10].groupby(['season', 'league']).agg(
            tymu=('team', 'size'), zapasu=('n', 'sum'), median=('med', 'first')
        ).reset_index()
        if not suspect.empty:
            print("\n   ⚠️  Podezřelé sezóny (málo zápasů na tým — chyba značení?):")
            for _, r in suspect.iterrows():
                print(f"        [{r['season']}] {r['league']:3s} "
                      f"{r['tymu']} týmů, {r['zapasu'] // 2} zápasů, "
                      f"medián {int(r['median'])}/tým")


print("🔍 KONTROLA DAT v4")
check_table("prepared_datasets")
check_table("prepared_fixtures")
print(f"\n{'=' * 70}\n✅ HOTOVO\n{'=' * 70}")