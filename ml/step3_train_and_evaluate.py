# ml/step3_train_and_evaluate.py
# Trénink modelů se selekcí features (RFECV)
# FIX: Odstraněno varování XGBoostu (use_label_encoder)

import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, f1_score
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Načtení featur
from ml.shared_features import performance_features

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def train_and_evaluate():
    print("📥 Načítám trénovací data...")
    df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # 1. Příprava X a y
    valid_features = [f for f in performance_features if f in df.columns]

    X_raw = df[valid_features].replace([np.inf, -np.inf], np.nan)
    y = df["target"].astype(int)

    print(f"✅ Použito {len(valid_features)} features pro trénink.")

    # 2. Selekce Features (RFECV)
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
    selector.fit(X_tmp, y)

    selected_mask = selector.support_
    selected_features = [f for f, s in zip(valid_features, selected_mask) if s]

    print(f"   Původně: {len(valid_features)} -> Nyní: {len(selected_features)} features.")

    # 3. Finální příprava dat
    X_selected = X_raw[selected_features]

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_selected)
    X_scaled = scaler.fit_transform(X_imp)

    tscv = TimeSeriesSplit(n_splits=3)

    # Definice modelů
    # FIX: Odstraněn parametr use_label_encoder=False (způsoboval varování)
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

    print("🔍 Ladím hyperparametry a trénuji modely...")
    trained_models = {}
    results = []

    for name, clf in models.items():
        calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=tscv)

        acc_scores, ll_scores, brier_scores, f1_scores = [], [], [], []

        try:
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                calibrated_clf.fit(X_train, y_train)
                probs = calibrated_clf.predict_proba(X_test)
                preds = calibrated_clf.predict(X_test)

                acc_scores.append(accuracy_score(y_test, preds))
                ll_scores.append(log_loss(y_test, probs))

                bs = 0
                for i in range(3):
                    y_true_binary = (y_test == i).astype(int)
                    bs += brier_score_loss(y_true_binary, probs[:, i])
                brier_scores.append(bs / 3)
                f1_scores.append(f1_score(y_test, preds, average='weighted'))
        except Exception as e:
            print(f"   ⚠️ Chyba při CV u modelu {name}: {e}. Přeskakuji CV, trénuji finální model.")
            acc_scores = [0]
            ll_scores = [0]

        # Finální fit
        calibrated_clf.fit(X_scaled, y)
        trained_models[name] = calibrated_clf

        # Ukládáme selected_features
        joblib.dump((imputer, scaler, calibrated_clf, selected_features), os.path.join(DATA_DIR, f"model_{name}.pkl"))

        print(f"   📊 {name}: Acc={np.mean(acc_scores):.3f}, LogLoss={np.mean(ll_scores):.3f}")
        results.append((name, np.mean(ll_scores)))

    # 4. Voting Ensemble
    print("\n🚀 Trénuji Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_scaled, y)

    joblib.dump((imputer, scaler, voting_clf, selected_features), os.path.join(DATA_DIR, "model_voting_ensemble.pkl"))

    print(f"🏆 Vítěz tréninku: voting_ensemble uložen.")
    print(f"✅ Hotovo. Modely nyní očekávají přesně {len(selected_features)} features.")


if __name__ == "__main__":
    train_and_evaluate()