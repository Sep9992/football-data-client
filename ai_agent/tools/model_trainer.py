"""
model_trainer.py
Nástroj AI agenta pro trénink prediktivních modelů.
UPGRADE: Podpora pro Ligovou DNA (4 core neutral + 4 League DNA + 8 párů = 24 features)
a automatická filtrace konstantních sloupců.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight


class ModelTrainer:
    def __init__(self, db_url=None):
        if not db_url:
            db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL není nastaveno.")

        self.engine = create_engine(db_url)
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
        os.makedirs(self.model_dir, exist_ok=True)

        self.log_file = os.path.join(self.model_dir, "training_log.json")

        self.CLASS_LABELS = {0: "Away", 1: "Draw", 2: "Home"}
        self.N_PAIRS = 8
        self.N_NEUTRAL = 4
        self.LEAGUE_DNA_COLS = [
            'league_avg_goals', 'league_draw_rate', 'league_home_advantage', 'league_btts_rate'
        ]
        self.DRAW_BOOST_THRESHOLD = 0.22

    def train_all(self) -> dict:
        print("\n📥 Načítám data z prepared_datasets...")
        df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", self.engine)
        if df.empty:
            return {"status": "error", "message": "Tabulka prepared_datasets je prázdná."}

        print(f"  Načteno: {len(df)} zápasů ({df['match_date'].min()} → {df['match_date'].max()})")

        all_feature_cols = self._get_feature_columns(df)
        print(f"  Kandidátní features (bez leakage, bez konstant): {len(all_feature_cols)}")

        X_all, y_class, y_goals_h, y_goals_a = self._prepare_data(df, all_feature_cols)

        # 1. Symetrický Feature Selection + Ligová DNA
        print("\n" + "=" * 70)
        print("🔬 SYMETRICKÝ FEATURE SELECTION + LIGOVÁ DNA")
        print("=" * 70)
        selected_features = self._select_features_symmetric(X_all.values, y_class, all_feature_cols)
        X = X_all[selected_features]
        joblib.dump(selected_features, os.path.join(self.model_dir, "feature_cols.pkl"))

        # 2. Voting Classifier
        print("\n" + "=" * 70)
        print("🗳️  A) VOTING CLASSIFIER (RF + GBM + LR)")
        print("=" * 70)
        voting_pipe = self._build_voting_classifier()
        acc_v, f1_v, std_v, thr_v, folds_v, recall_v = self._validate_classifier(voting_pipe, X, y_class, "Voting")
        print("\n  🔥 Finální trénink Voting modelu...")
        voting_pipe.fit(X, y_class)

        # 3. XGBoost Classifier
        print("\n" + "=" * 70)
        print("🌲 B) XGBOOST CLASSIFIER")
        print("=" * 70)
        xgb_pipe = self._build_xgboost_classifier()
        acc_x, f1_x, std_x, thr_x, folds_x, recall_x = self._validate_classifier(xgb_pipe, X, y_class, "XGBoost")
        print("\n  🔥 Finální trénink XGBoost modelu...")
        sw_all = compute_sample_weight('balanced', y_class)
        xgb_pipe.fit(X, y_class, xgb__sample_weight=sw_all)

        # 4. Porovnání
        print("\n" + "=" * 70)
        print("🏆 C) SROVNÁNÍ MODELŮ")
        print("=" * 70)
        baseline = (y_class == 2).sum() / len(y_class)
        print(f"\n  {'Model':22s} {'Accuracy':>10s} {'±Std':>8s} {'F1-macro':>10s} {'vs Baseline':>12s}")
        print("  " + "─" * 65)
        print(f"  {'Voting (RF+GBM+LR)':22s} {acc_v:>10.4f} {std_v:>8.4f} {f1_v:>10.4f} {acc_v-baseline:>+12.4f}")
        print(f"  {'XGBoost':22s} {acc_x:>10.4f} {std_x:>8.4f} {f1_x:>10.4f} {acc_x-baseline:>+12.4f}")
        print(f"  {'Baseline (always Home)':22s} {baseline:>10.4f} {'N/A':>8s} {'N/A':>10s} {'±0':>12s}")

        best_clf_name = "Voting" if f1_v >= f1_x else "XGBoost"
        best_clf = voting_pipe if f1_v >= f1_x else xgb_pipe
        best_thr = thr_v if f1_v >= f1_x else thr_x
        print(f"\n  🥇 Nejlepší klasifikátor: {best_clf_name} (F1={max(f1_v, f1_x):.4f})")
        print(f"  🎯 Optimální Draw threshold: {best_thr:.2f}")

        # 5. Regresory (xG)
        print("\n" + "=" * 70)
        print("⚽ D) REGRESORY PRO POČET GÓLŮ")
        print("=" * 70)
        regs = self._build_regressors()
        reg_results, best_reg = self._validate_regressors(regs, X, y_goals_h, y_goals_a)
        print("\n  🔥 Finální trénink regresorů...")
        for name, reg in regs.items():
            y_t = y_goals_h if name.endswith('h') else y_goals_a
            reg.fit(X, y_t)

        # 6. Ukládání modelů
        to_save = {
            "voting_classifier.pkl": voting_pipe,
            "xgb_classifier.pkl": xgb_pipe,
            "best_classifier.pkl": best_clf,
            "poisson_home_goals.pkl": regs['poisson_h'],
            "poisson_away_goals.pkl": regs['poisson_a'],
            "xgb_home_goals.pkl": regs['xgb_h'],
            "xgb_away_goals.pkl": regs['xgb_a'],
            "draw_threshold.pkl": best_thr,
        }
        for fname, obj in to_save.items():
            joblib.dump(obj, os.path.join(self.model_dir, fname))

        # 7. Logování
        results_log = {
            "n_samples": len(df),
            "n_features_sel": len(selected_features),
            "baseline": round(baseline, 4),
            "best_classifier": best_clf_name,
            "best_f1_score": round(max(f1_v, f1_x), 4),
            "draw_threshold": round(best_thr, 2),
            "voting_acc": round(acc_v, 4),
            "xgb_acc": round(acc_x, 4)
        }
        self._save_log(results_log)
        results_log["status"] = "success"

        return results_log

    # =========================================================================
    # INTERNÍ FUNKCE
    # =========================================================================

    def _get_feature_columns(self, df):
        allowed_prefixes = [
            "home_avg_", "away_avg_", "home_elo", "away_elo", "elo_",
            "home_shot_conv", "away_shot_conv", "home_defensive", "away_defensive",
            "home_possession_q", "away_possession_q", "home_discipline", "away_discipline",
            "home_goal_diff", "away_goal_diff", "home_advantage", "attack_vs_defense",
            "home_x_elo", "form_x_attack", "home_rest", "away_rest",
            "league_"  # Nové proměnné Ligové DNA
        ]
        market_keep = {"market_value_diff_scaled"}
        forbidden = {"goals_home", "goals_away", "home_win", "points_home", "points_away",
                     "fixture_id", "result", "market_value_diff", "market_value_home", "market_value_away"}

        features = []
        for col in df.columns:
            if df[col].dtype == 'object' or 'date' in col.lower() or col in forbidden:
                continue
            if col in market_keep or any(col.startswith(p) for p in allowed_prefixes):
                # Odfiltrujeme konstantní sloupce (kde jsou samé nuly), abychom předešli varováním
                if df[col].nunique(dropna=True) > 1:
                    features.append(col)
        return sorted(features)

    def _prepare_data(self, df, feature_cols):
        X = df[feature_cols].copy()
        conditions = [(df['goals_home'] > df['goals_away']), (df['goals_home'] == df['goals_away'])]
        y_class = np.select(conditions, [2, 1], default=0)

        unique, counts = np.unique(y_class, return_counts=True)
        print(f"\n  🎯 Distribuce výsledků:")
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y_class) * 100
            bar = "█" * int(pct / 2)
            print(f"    {self.CLASS_LABELS[cls]:5s} ({cls}): {cnt:3d} ({pct:.1f}%) {bar}")
        print(f"    Baseline (vždy Home): {counts[unique==2][0]/len(y_class)*100:.1f}%")

        return X, y_class, df["goals_home"].values, df["goals_away"].values

    def _select_features_symmetric(self, X_vals, y, feature_cols):
        dna_available = [f for f in self.LEAGUE_DNA_COLS if f in feature_cols]
        total_target = self.N_NEUTRAL + len(dna_available) + self.N_PAIRS * 2
        print(f"\n  🔍 Hledám optimální features... Cíl: {self.N_NEUTRAL} neutral + {len(dna_available)} League DNA + {self.N_PAIRS} párů×2 = {total_target} features")

        imp = SimpleImputer(strategy='mean')
        scl = StandardScaler()
        X_prep = scl.fit_transform(imp.fit_transform(X_vals))

        sel_f = SelectKBest(f_classif, k='all')
        sel_f.fit(X_prep, y)
        f_ranks = pd.Series(np.nan_to_num(sel_f.scores_, nan=0.0), index=feature_cols).rank(ascending=False)
        mi_ranks = pd.Series(mutual_info_classif(X_prep, y, random_state=42), index=feature_cols).rank(ascending=False)

        rf_sel = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_sel.fit(X_prep, y)
        rf_ranks = pd.Series(rf_sel.feature_importances_, index=feature_cols).rank(ascending=False)

        consensus = ((f_ranks + mi_ranks + rf_ranks) / 3).sort_values()

        neutral_candidates = ['elo_diff', 'market_value_diff_scaled', 'home_x_elo', 'home_advantage',
                              'attack_vs_defense', 'form_x_attack_home', 'home_elo', 'away_elo',
                              'elo_x_market', 'home_goal_diff_avg', 'away_goal_diff_avg']
        neutral_available = [f for f in neutral_candidates if f in feature_cols]
        selected_neutral = sorted(neutral_available, key=lambda f: consensus.get(f, 999))[:self.N_NEUTRAL]

        home_feats = [f for f in feature_cols if f.startswith('home_avg_')]
        pair_scores = []
        for hf in home_feats:
            af = hf.replace('home_avg_', 'away_avg_')
            if af in feature_cols:
                pair_scores.append((hf, af, (consensus.get(hf, 999) + consensus.get(af, 999)) / 2))

        pair_scores.sort(key=lambda x: x[2])
        selected = selected_neutral.copy() + dna_available
        for hf, af, _ in pair_scores[:self.N_PAIRS]:
            selected.extend([hf, af])

        ratio = len(X_vals) / len(selected)
        print(f"  ✅ Vybráno {len(selected)} features  |  ratio {len(X_vals)}:{len(selected)} = {ratio:.0f}:1")

        print(f"\n  📌 Neutral features ({len(selected_neutral)}):")
        for feat in selected_neutral:
            print(f"    {'✅':3s} {feat:45s}  rank={consensus.get(feat, 999):.1f}")

        if dna_available:
            print(f"\n  🧬 League DNA features ({len(dna_available)}):")
            for feat in dna_available:
                print(f"    {'✅':3s} {feat:45s}  rank={consensus.get(feat, 999):.1f}")

        print(f"\n  📌 Párové features ({self.N_PAIRS} párů):")
        print(f"  {'HOME':45s}  {'AWAY':45s}  {'Avg Rank':>6}")
        print(f"  {'─'*105}")
        for hf, af, avg in pair_scores[:self.N_PAIRS]:
            print(f"  ✅ {hf[:43]:43s}  ✅ {af[:43]:43s}  {avg:6.1f}")

        return selected

    def _build_voting_classifier(self):
        preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        clf_rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        clf_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
        clf_lr = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced', random_state=42)
        voting = VotingClassifier(estimators=[('rf', clf_rf), ('gb', clf_gb), ('lr', clf_lr)], voting='soft')
        return Pipeline([('prep', preprocessor), ('clf', voting)])

    def _build_xgboost_classifier(self):
        preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                eval_metric='mlogloss', objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
        return Pipeline([('prep', preprocessor), ('xgb', xgb_clf)])

    def _build_regressors(self):
        prep_scaled = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        prep_tree = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        return {
            'poisson_h': Pipeline([('prep', prep_scaled), ('poi', PoissonRegressor(alpha=0.5, max_iter=1000))]),
            'poisson_a': Pipeline([('prep', prep_scaled), ('poi', PoissonRegressor(alpha=0.5, max_iter=1000))]),
            'xgb_h': Pipeline([('prep', prep_tree), ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, n_jobs=-1))]),
            'xgb_a': Pipeline([('prep', prep_tree), ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, n_jobs=-1))]),
        }

    def _validate_classifier(self, model, X, y_class, model_name, n_splits=5):
        print(f"\n  🔄 {model_name} — TimeSeriesSplit (n={n_splits})...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs, f1s, cms, all_proba, all_y = [], [], [], [], []

        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_tr, y_tr = X.iloc[train_idx], y_class[train_idx]
            X_te, y_te = X.iloc[test_idx],  y_class[test_idx]

            if 'XGBoost' in model_name:
                sw = compute_sample_weight('balanced', y_tr)
                model.fit(X_tr, y_tr, xgb__sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)

            preds = model.predict(X_te)
            accs.append(accuracy_score(y_te, preds))
            f1s.append(f1_score(y_te, preds, average='macro', zero_division=0))
            cms.append(confusion_matrix(y_te, preds, labels=[0, 1, 2]))
            all_proba.append(model.predict_proba(X_te))
            all_y.append(y_te)

            print(f"    Fold {fold_i}: acc={accs[-1]:.3f} | F1={f1s[-1]:.3f}  (train={len(train_idx)}, test={len(test_idx)})")

        avg_acc, avg_f1, std_acc = np.mean(accs), np.mean(f1s), np.std(accs)
        avg_cm = np.mean(cms, axis=0).astype(int)

        away_recall = avg_cm[0, 0] / max(avg_cm[0].sum(), 1) * 100
        draw_recall = avg_cm[1, 1] / max(avg_cm[1].sum(), 1) * 100
        home_recall = avg_cm[2, 2] / max(avg_cm[2].sum(), 1) * 100

        print(f"\n  📊 {model_name}: Accuracy={avg_acc:.4f} ± {std_acc:.4f}  |  F1={avg_f1:.4f}")
        print(f"  Per-class recall:  Away={away_recall:.0f}%  Draw={draw_recall:.0f}%  Home={home_recall:.0f}%")
        print(f"  Confusion Matrix (predikce →):")
        print(f"    {'':8s} Away  Draw  Home")
        for i, label in enumerate(['Away', 'Draw', 'Home']):
            print(f"    {label:8s} {avg_cm[i][0]:4d}  {avg_cm[i][1]:4d}  {avg_cm[i][2]:4d}")

        recall_dict = {"away": away_recall, "draw": draw_recall, "home": home_recall}

        best_thr = self.DRAW_BOOST_THRESHOLD
        if all_proba:
            print(f"\n  🎯 Draw threshold tuning...")
            all_p = np.vstack(all_proba)
            all_y_arr = np.concatenate(all_y)
            best_thr = self._find_optimal_draw_threshold(all_p, all_y_arr)
            print(f"  📈 Optimální Draw threshold pro {model_name} nalezen na hodnotě: {best_thr:.2f}")

        return avg_acc, avg_f1, std_acc, best_thr, [], recall_dict

    def _find_optimal_draw_threshold(self, all_proba, all_y):
        thresholds = np.arange(0.15, 0.40, 0.02)
        best_f1, best_thr = 0, self.DRAW_BOOST_THRESHOLD
        for thr in thresholds:
            preds = []
            for p in all_proba:
                if p[1] >= thr and p[1] > min(p[0], p[2]):
                    preds.append(1)
                else:
                    preds.append(int(np.argmax(p)))
            f1 = f1_score(all_y, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        return best_thr

    def _validate_regressors(self, regs, X, y_goals_h, y_goals_a, n_splits=5):
        print(f"\n  🔄 Regresory — TimeSeriesSplit (n={n_splits})...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes = {k: [] for k in regs}
        for train_idx, test_idx in tscv.split(X):
            for name, reg in regs.items():
                y_tr = y_goals_h[train_idx] if name.endswith('h') else y_goals_a[train_idx]
                y_te = y_goals_h[test_idx]  if name.endswith('h') else y_goals_a[test_idx]
                reg.fit(X.iloc[train_idx], y_tr)
                maes[name].append(mean_absolute_error(y_te, reg.predict(X.iloc[test_idx])))

        res = {k: np.mean(v) for k, v in maes.items()}
        poi_avg, xgb_avg = (res['poisson_h'] + res['poisson_a']) / 2, (res['xgb_h'] + res['xgb_a']) / 2

        print(f"\n  {'Model':15s}  {'MAE home':>10s}  {'MAE away':>10s}  {'MAE avg':>10s}")
        print(f"  {'─'*50}")
        print(f"  {'Poisson':15s}  {res['poisson_h']:>10.3f}  {res['poisson_a']:>10.3f}  {poi_avg:>10.3f}")
        print(f"  {'XGBoost':15s}  {res['xgb_h']:>10.3f}  {res['xgb_a']:>10.3f}  {xgb_avg:>10.3f}")

        best = 'Poisson' if poi_avg <= xgb_avg else 'XGBoost'
        print(f"  🥇 Nejlepší regresor: {best}")

        return res, best

    def _save_log(self, results):
        existing = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        })
        with open(self.log_file, 'w') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)