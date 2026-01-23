# ml/step3_2_train_xgboost.py
# TRÉNINK MODELU: XGBoost Classifier (The Ferrari) 🏎️
# Cíl: Porazit Random Forest v přesnosti a Log Loss.

import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Nastavení
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "model_xgboost_ferrari.pkl")

# Importujeme featury ze sdíleného souboru
from ml.shared_features import performance_features


def train_xgboost():
    print("🚀 Startuji trénink XGBoost (The Ferrari)...")

    # 1. Načtení dat
    print("⏳ Načítám data z DB...")
    df = pd.read_sql("SELECT * FROM prepared_datasets", engine)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # Použijeme stejné featury jako u RF
    X = df[performance_features]
    y = df["target"].astype(int)

    print(f"📊 Dataset: {len(df)} řádků, {len(performance_features)} features")

    # 2. Rozdělení (Časové, ne náhodné - aby se neučil z budoucnosti)
    # Prvních 80% zápasů na trénink, posledních 20% na test
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"✂️ Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Definice Modelu (XGBoost)
    # Parametry nastaveny konzervativně pro fotbal (prevence overfittingu)
    xgb = XGBClassifier(
        n_estimators=150,  # Počet stromů
        learning_rate=0.05,  # Pomalejší učení = lepší stabilita
        max_depth=4,  # Menší hloubka = méně overfittingu
        subsample=0.8,  # Bere jen 80% dat pro každý strom (šum)
        colsample_bytree=0.8,  # Bere jen 80% featur pro každý strom
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )

    # DŮLEŽITÉ: Kalibrace pravděpodobností
    # XGBoost si často věří moc (např. 0.99). Kalibrace ho vrátí na zem (např. 0.85).
    calibrated_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=3)

    # 4. Pipeline
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # XGB umí NaN, ale imputer je jistota
        ('scaler', StandardScaler()),  # XGB nepotřebuje škálování, ale pomáhá konvergenci
        ('clf', calibrated_xgb)
    ])

    # 5. Trénink
    print("🧠 Trénuji model...")
    model_pipeline.fit(X_train, y_train)

    # 6. Evaluace
    print("📈 Vyhodnocuji na testovacích datech...")
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    print("\n" + "=" * 40)
    print(f"🏆 VÝSLEDKY XGBoost")
    print(f"✅ Accuracy: {acc:.4f} (Náhodný tip = 0.33)")
    print(f"📉 Log Loss: {loss:.4f} (Čím méně, tím lépe)")
    print("=" * 40)
    print("\nReport klasifikace:")
    print(classification_report(y_test, y_pred, target_names=["Home", "Draw", "Away"]))

    # 7. Feature Importance (Trochu složitější u Pipeline + Calibrated)
    # Vytáhneme vnitřní model
    try:
        base_xgb = model_pipeline.named_steps['clf'].calibrated_classifiers_[0].estimator
        importances = base_xgb.feature_importances_
        feature_imp = pd.DataFrame({'Feature': performance_features, 'Importance': importances})
        feature_imp = feature_imp.sort_values(by='Importance', ascending=False).head(10)
        print("\n🔝 TOP 10 Nejdůležitějších faktorů:")
        print(feature_imp.to_string(index=False))
    except:
        print("\n⚠️ Feature importance nelze zobrazit u kalibrovaného modelu jednoduše.")

    # 8. Uložení
    # Ukládáme to jako NOVÝ soubor, nepřepisujeme ten starý!
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\n💾 Model uložen do: {MODEL_PATH}")
    print("✅ Hotovo. Nyní můžete porovnat s Random Forest.")


if __name__ == "__main__":
    train_xgboost()