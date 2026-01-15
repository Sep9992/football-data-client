# ml/step5_analyze_features.py
# Analýza důležitosti příznaků (Feature Importance)
# FIX: Podpora nového formátu (4 položky) a CalibratedClassifierCV

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

# Pro analýzu je nejlepší RandomForest nebo XGBoost (mají jasnou feature_importances_)
# Voting ani LogReg nejsou pro vizualizaci tak vhodné.
MODEL_FILE = "model_randomforest.pkl"

model_path = os.path.join(DATA_DIR, MODEL_FILE)


def analyze_features():
    if not os.path.exists(model_path):
        print(f"❌ Model {MODEL_FILE} nebyl nalezen v {DATA_DIR}.")
        return

    print(f"🔍 Analyzuji model: {MODEL_FILE} ...")

    try:
        artifact = joblib.load(model_path)

        # --- 1. UNPACKING (Rozbalení) ---
        # Nový formát ze Step 3 má 4 položky
        if len(artifact) == 4:
            imputer, scaler, model, selected_features = artifact
        # Starý formát (pro jistotu)
        elif len(artifact) == 5:
            imputer, scaler, selector, model, selected_features = artifact
        else:
            print(f"❌ Neznámý formát modelu. Počet položek: {len(artifact)}")
            return

        print(f"✅ Model načten. Počet features: {len(selected_features)}")

        # --- 2. ZÍSKÁNÍ IMPORTANCES ---
        importances = None

        # Pokud je model zabalený v CalibratedClassifierCV (což v Step 3 děláme),
        # musíme se dostat k vnitřnímu modelu.
        base_model = model
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            # Vezmeme první z kalibrovaných modelů (reprezentativní vzorek)
            base_model = model.calibrated_classifiers_[0].estimator

        # Nyní získáme důležitosti z base_modelu
        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            # Pro lineární modely (LogReg) - vezmeme absolutní hodnotu koeficientů
            importances = np.abs(base_model.coef_[0])

        if importances is None:
            print("⚠️ Tento model neposkytuje metriku důležitosti (např. KNN, SVM s jádrem, nebo Voting).")
            print("   Zkuste změnit MODEL_FILE na 'model_randomforest.pkl' nebo 'model_xgboost.pkl'.")
            return

        # Kontrola délek
        if len(selected_features) != len(importances):
            print(f"⚠️ Nesedí počet: Názvy={len(selected_features)}, Hodnoty={len(importances)}")
            # Fallback: zkusíme oříznout nebo doplnit, ale raději reportujeme chybu
            return

        # --- 3. VYTVOŘENÍ DATAFRAME ---
        df_imp = pd.DataFrame({
            "Feature": selected_features,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        # Normalizace na procenta (aby součet byl 100)
        df_imp["Importance"] = 100 * df_imp["Importance"] / df_imp["Importance"].sum()

        # Výpis TOP 15
        print("\n🏆 TOP 15 Klíčových faktorů (%):")
        print(df_imp.head(15).to_string(index=False, float_format="%.2f"))

        # --- 4. GRAF ---
        plt.figure(figsize=(12, 10))
        sns.barplot(x="Importance", y="Feature", hue="Feature", legend=False, data=df_imp.head(20), palette="viridis")
        plt.title(f"Feature Importance ({MODEL_FILE})")
        plt.xlabel("Důležitost (%)")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Uložení grafu
        plot_path = os.path.join(DATA_DIR, "feature_importance.png")
        plt.savefig(plot_path)
        print(f"\n📊 Graf uložen do: {plot_path}")
        plt.show()

    except Exception as e:
        print(f"❌ Chyba při analýze: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_features()