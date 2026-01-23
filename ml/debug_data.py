import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Načtení připojení
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

print("🔍 Načítám data z prepared_datasets...")
try:
    df = pd.read_sql("SELECT * FROM prepared_datasets", engine)

    # 1. Základní info o NULL hodnotách
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    print(f"\n📊 Nalezeno {len(df)} řádků.")
    print(f"⚠️ Sloupce s NULL hodnotami:\n{null_cols}")

    # 2. Uložení vzorku dat (prvních 100 řádků)
    df.head(1000).to_csv("data_sample.csv", index=False)
    print("\n✅ Soubor 'data_sample.csv' byl vytvořen. Nahrajte ho do chatu.")

    # 3. Uložení info o strukturách
    with open("data_info.txt", "w", encoding="utf-8") as f:
        df.info(buf=f)
        f.write("\n\n--- NULL VALUES ---\n")
        f.write(null_cols.to_string())
    print("✅ Soubor 'data_info.txt' byl vytvořen. Nahrajte ho také.")

except Exception as e:
    print(f"❌ Chyba: {e}")