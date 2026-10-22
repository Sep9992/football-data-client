import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Načtení databáze
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

def fix_dates():
    print("🛠️ SPUŠTĚN OPRAVÁŘ DATUMŮ PRO LIGUE 1...")
    try:
        with engine.begin() as conn:
            # 1. Oprava datumu ve fixtures (vezme páteční zápasy a přesune je do léta 2025)
            query_update = """
                UPDATE fixtures
                SET match_date = '2025-08-01'
                WHERE match_date = '2026-10-09' 
                AND id IN (SELECT fixture_id FROM match_statistics);
            """
            result = conn.execute(text(query_update))
            print(f"✅ Přesunuto {result.rowcount} zápasů z 9.10.2026 hluboko do historie (2025).")

            # 2. Vyčištění čekárny na sázení (aby agent přestal tyto zápasy predikovat)
            query_delete = """
                DELETE FROM prepared_fixtures 
                WHERE fixture_id IN (SELECT fixture_id FROM match_statistics);
            """
            result_del = conn.execute(text(query_delete))
            print(f"✅ Vymazáno {result_del.rowcount} již odehraných zápasů z fronty pro sázení.")

            print("\nNyní je vše 100% čisté. Můžete znovu spustit agent_core.py.")
    except Exception as e:
        print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    fix_dates()