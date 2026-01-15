# ml/step3b_poisson_model.py
# Dixon-Coles Model (Poisson + Rho Correction)
# UPDATED: Výpočet parametru Rho pro korekci nízko skórujících remíz

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from ml.shared_features import performance_features

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def train_poisson():
    print("📥 Načítám data pro Dixon-Coles (Poisson) model...")
    df = pd.read_sql("SELECT * FROM prepared_datasets", engine)

    # Cílové proměnné
    y_home = df["goals_home"]
    y_away = df["goals_away"]

    # Features
    try:
        df_fixt_cols = pd.read_sql("SELECT * FROM prepared_fixtures LIMIT 0", engine).columns
        valid_features = [f for f in performance_features if f in df.columns and f in df_fixt_cols]
    except:
        valid_features = [f for f in performance_features if f in df.columns]

    X = df[valid_features].replace([np.inf, -np.inf], np.nan)

    # Preprocessing
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    print(f"🚀 Trénuji regresory na {len(valid_features)} features...")

    # 1. Trénink základních Poisson modelů
    reg_home = PoissonRegressor(alpha=0.5, max_iter=1000)
    reg_home.fit(X_scaled, y_home)

    reg_away = PoissonRegressor(alpha=0.5, max_iter=1000)
    reg_away.fit(X_scaled, y_away)

    # 2. Výpočet parametru RHO (Dixon-Coles Correction)
    # Rho měří závislost mezi góly domácích a hostů (korelace reziduí)
    pred_home = reg_home.predict(X_scaled)
    pred_away = reg_away.predict(X_scaled)

    # Rezidua (rozdíl mezi realitou a predikcí)
    res_home = y_home - pred_home
    res_away = y_away - pred_away

    # Kovariance reziduí / (std_h * std_a) -> Pearsonova korelace
    # Zjednodušeně použijeme korelaci chyb
    rho = np.corrcoef(res_home, res_away)[0, 1]

    print(f"   📐 Vypočítané Rho (závislost): {rho:.4f}")
    print("      (Záporné Rho znamená, že nízko skórující remízy jsou častější, než model čeká)")

    # Evaluace (MAE)
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_test = X_scaled[test_idx]
        y_test_h = y_home.iloc[test_idx]
        y_test_a = y_away.iloc[test_idx]

        pred_h = reg_home.predict(X_test)
        pred_a = reg_away.predict(X_test)

        mae_h = mean_absolute_error(y_test_h, pred_h)
        mae_a = mean_absolute_error(y_test_a, pred_a)
        mae_scores.append((mae_h + mae_a) / 2)

    avg_mae = np.mean(mae_scores)
    print(f"   📊 Průměrná chyba (MAE): {avg_mae:.3f} gólů")

    # Uložení - Přidáváme RHO do balíčku
    artifact = (imputer, scaler, reg_home, reg_away, valid_features, rho)
    joblib.dump(artifact, os.path.join(DATA_DIR, "model_poisson.pkl"))
    print("✅ Dixon-Coles model (s Rho parametrem) uložen.")


if __name__ == "__main__":
    train_poisson()