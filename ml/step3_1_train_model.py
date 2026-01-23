# ml/step3_1_train_model.py
# KOMPLETNÍ TRÉNINK: Feature Selection -> Voting Ensemble -> Poisson (Dixon-Coles)
# Tento skript nahrazuje step3 i step3b.

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Sklearn imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, f1_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Načtení featur
from ml.shared_features import performance_features

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def train_all_models():
    print("🚀 START: Komplexní trénink modelů...")

    # --- 1. NAČTENÍ DAT ---
    print("📥 Načítám data z DB...")
    df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    valid_features = [f for f in performance_features if f in df.columns]
    X_raw = df[valid_features].replace([np.inf, -np.inf], np.nan)

    # Cílové proměnné
    y_class = df["target"].astype(int)  # 0, 1, 2 (pro Klasifikaci)
    y_home = df["goals_home"]  # Góly (pro Poisson)
    y_away = df["goals_away"]  # Góly (pro Poisson)

    print(f"✅ Data načtena. {len(df)} záznamů, {len(valid_features)} features.")

    # --- 2. SELEKCE FEATURES (RFECV) ---
    # Vybereme features jednou a použijeme je pro OBA typy modelů.
    # To zaručí, že se nestane chyba "shape mismatch".
    tmp_imputer = SimpleImputer(strategy='mean')
    X_tmp = tmp_imputer.fit_transform(X_raw)

    print("🧹 Provádím selekci features (RFECV)...")
    selector = RFECV(
        estimator=RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=5),
        step=1,
        cv=TimeSeriesSplit(3),
        scoring='neg_log_loss',
        min_features_to_select=10
    )
    selector.fit(X_tmp, y_class)

    selected_mask = selector.support_
    selected_features = [f for f, s in zip(valid_features, selected_mask) if s]

    print(f"   Vybráno: {len(selected_features)} features (z původních {len(valid_features)}).")

    # Příprava finálních dat (Impute + Scale)
    X_selected = X_raw[selected_features]

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_selected)
    X_scaled = scaler.fit_transform(X_imp)

    tscv = TimeSeriesSplit(n_splits=3)

    # =========================================================================
    # ČÁST A: KLASIFIKACE (Voting Ensemble)
    # =========================================================================
    print("\n🤖 ČÁST A: Trénink klasifikátorů (1X2)...")

    models = {
        "logreg": LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', C=0.1),
        "randomforest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=4,
                                               class_weight='balanced', random_state=42),
        "gradientboost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        "svc": SVC(probability=True, kernel='rbf', C=0.5, class_weight='balanced', random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "xgboost": XGBClassifier(eval_metric='mlogloss', n_estimators=100, max_depth=4, learning_rate=0.05,
                                 random_state=42)
    }

    trained_models = {}

    for name, clf in models.items():
        calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=tscv)

        # Rychlý fit na celých datech (validaci jsme už ladili)
        try:
            calibrated_clf.fit(X_scaled, y_class)
            trained_models[name] = calibrated_clf

            # Uložení jednotlivých modelů
            joblib.dump((imputer, scaler, calibrated_clf, selected_features),
                        os.path.join(DATA_DIR, f"model_{name}.pkl"))
            print(f"   ✅ {name}: Natrénováno a uloženo.")

        except Exception as e:
            print(f"   ❌ Chyba u {name}: {e}")

    # Voting Ensemble
    print("   🚀 Skládám Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_scaled, y_class)

    # Uložení Voting modelu
    joblib.dump((imputer, scaler, voting_clf, selected_features), os.path.join(DATA_DIR, "model_voting_ensemble.pkl"))
    print("🏆 Vítěz: model_voting_ensemble.pkl uložen.")

    # =========================================================================
    # ČÁST B: REGRESE (Poisson / Dixon-Coles)
    # =========================================================================
    print("\n⚽ ČÁST B: Trénink Poisson modelů (Góly)...")

    # Použijeme stejná škálovaná data (X_scaled) - to zaručuje kompatibilitu
    reg_home = PoissonRegressor(alpha=0.5, max_iter=1000)
    reg_home.fit(X_scaled, y_home)

    reg_away = PoissonRegressor(alpha=0.5, max_iter=1000)
    reg_away.fit(X_scaled, y_away)

    # Výpočet Rho (Dixon-Coles korelace)
    pred_h = reg_home.predict(X_scaled)
    pred_a = reg_away.predict(X_scaled)

    res_h = y_home - pred_h
    res_a = y_away - pred_a
    rho = np.corrcoef(res_h, res_a)[0, 1]

    print(f"   📐 Rho (korelace chyb): {rho:.4f}")

    # Uložení Poisson modelu
    # Formát: (imputer, scaler, reg_home, reg_away, selected_features, rho)
    joblib.dump((imputer, scaler, reg_home, reg_away, selected_features, rho),
                os.path.join(DATA_DIR, "model_poisson.pkl"))
    print("✅ model_poisson.pkl uložen.")

    print("\n🏁 HOTOVO. Všechny modely jsou synchronizované a připravené.")


if __name__ == "__main__":
    train_all_models()