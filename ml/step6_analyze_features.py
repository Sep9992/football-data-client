# ml/step6_analyze_features.py
# Analýza důležitosti příznaků (Feature Importance)
# UPDATED: Podpora pro CalibratedClassifierCV

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV

# --- Nastavení ---
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Vybereme model pro analýzu
# Doporučuji 'randomforest' nebo 'xgboost' (pokud existuje)
MODEL_FILE = "model_randomforest.pkl"

model_path = os.path.join(DATA_DIR, MODEL_FILE)


def analyze_features():
    if not os.path.exists(model_path):
        print(f"❌ Model {MODEL_FILE} nebyl nalezen v {DATA_DIR}.")
        return

    print(f"🔍 Analyzuji model: {MODEL_FILE} ...")

    try:
        artifact = joblib.load(model_path)
        # Rozbalení podle verze (očekáváme 5 položek)
        if len(artifact) == 5:
            imputer, scaler, selector, model, valid_features = artifact
        else:
            print("⚠️ Neznámý formát modelu. Očekáváno 5 položek.")
            return
    except Exception as e:
        print(f"❌ Chyba při načítání modelu: {e}")
        return

    # 1. Zjistíme features po selekci
    if selector:
        selected_indices = selector.get_support(indices=True)
        selected_features = [valid_features[i] for i in selected_indices]
        print(f"✅ Model používá {len(selected_features)} z původních {len(valid_features)} featur.")
    else:
        selected_features = valid_features
        print(f"✅ Model používá všechny featury.")

    # 2. Získání důležitosti (Feature Importances)
    importances = None

    # A) Pokud je model přímo stromový (RF, XGB)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # B) Pokud je to CalibratedClassifierCV (musíme jít dovnitř)
    elif hasattr(model, "calibrated_classifiers_"):
        print("   ℹ️ Detekován kalibrovaný model. Průměruji důležitost z pod-modelů...")
        imp_list = []
        for clf in model.calibrated_classifiers_:
            # clf.estimator je ten skutečný model (RF, XGB...)
            if hasattr(clf.estimator, "feature_importances_"):
                imp_list.append(clf.estimator.feature_importances_)
            elif hasattr(clf.estimator, "coef_"):
                # Pro lineární modely (LogReg)
                imp_list.append(abs(clf.estimator.coef_[0]))

        if imp_list:
            # Zprůměrujeme hodnoty ze všech foldů
            importances = np.mean(imp_list, axis=0)

    # C) Pokud je to lineární model (LogReg) bez kalibrace
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])

    if importances is None:
        print("⚠️ Tento model neposkytuje metriku důležitosti (např. KNN, SVM s jádrem).")
        print("   Zkuste změnit MODEL_FILE na 'model_randomforest.pkl' nebo 'model_xgboost.pkl'.")
        return

    # 3. Vytvoření DataFrame
    if len(selected_features) != len(importances):
        print(f"⚠️ Nesedí počet: Názvy={len(selected_features)}, Hodnoty={len(importances)}")
        return

    df_imp = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    # Normalizace na procenta (aby součet byl 100)
    df_imp["Importance"] = 100 * df_imp["Importance"] / df_imp["Importance"].sum()

    # Výpis TOP 15
    print("\n🏆 TOP 15 Klíčových faktorů (%):")
    print(df_imp.head(15).to_string(index=False, float_format="%.2f"))

    # 4. Graf
    plt.figure(figsize=(12, 10))
    sns.barplot(x="Importance", y="Feature", data=df_imp.head(25), palette="viridis")
    plt.title(f"Faktory rozhodující o výsledku (Model: {MODEL_FILE})")
    plt.xlabel("Vliv na predikci (%)")
    plt.ylabel(None)
    plt.tight_layout()

    out_img = os.path.join(DATA_DIR, f"feature_importance_{MODEL_FILE.replace('.pkl', '')}.png")
    plt.savefig(out_img)
    print(f"\n📊 Graf uložen do: {out_img}")
    plt.show()


if __name__ == "__main__":
    analyze_features()