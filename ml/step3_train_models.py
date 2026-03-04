""""
step3_train_models.py  v4 - Symetrický výběr Home/Away
=======================================================
VERZE ZMĚN:
  v1 - původní 2 samostatné skripty
  v2 - sloučení + oprava data leakage + class imbalance
  v3 - feature selection (k=15, ratio 17:1)
  v4 - SYMETRICKÝ HOME/AWAY výběr (tato verze)

PROBLÉM v3:
  Všech 15 vybraných features bylo HOME nebo neutral:
    → Away recall: pouze 46%
    → Home recall: 71%
    → Draw recall: 10%
  Model "neviděl" hosty → neschopnost predikovat výhry hostů

ŘEŠENÍ v4 - Soft Symmetry:
  1. Vybere top 4 neutral features (elo_diff, market_value...)
  2. Vybere top 8 PÁRŮ home+away podle průměrného ranku obou
  3. Výsledek: 20 features = 8 home + 8 away + 4 neutral
  4. Ratio: 260/20 = 13:1 (stále zdravý poměr)

  Současně:
  - Odstraněna duplicita market_value (nech jen diff_scaled)
  - Odstraněn home_avg_free_kicks (nízká kvalita)
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
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
from dotenv import load_dotenv

# --- KONFIGURACE ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)
LOG_FILE = os.path.join(MODEL_DIR, "training_log.json")

CLASS_LABELS = {0: "Away", 1: "Draw", 2: "Home"}

# Počet PÁRŮ home+away (výsledek = n_pairs*2 + n_neutral features)
N_PAIRS   = 8   # → 16 párových features
N_NEUTRAL = 4   # → 4 neutral features
# Celkem: 20 features, ratio 260/20 = 13:1

# Draw recall boost — threshold pod kterým se predikuje Draw místo argmax
DRAW_BOOST_THRESHOLD = 0.22


# =============================================================================
# 1. KANDIDÁTNÍ FEATURES (whitelist - bez data leakage)
# =============================================================================

def get_feature_columns(df):
    """
    WHITELIST: pouze sloupce existující PŘED zápasem.
    Odstraněna duplicita market_value:
      ❌ market_value_diff     (= market_value_diff_scaled * constant)
      ❌ market_value_home     (součást diff)
      ❌ market_value_away     (součást diff)
      ✅ market_value_diff_scaled  (normalizovaná verze, stačí jedna)
    """
    allowed_prefixes = [
        "home_avg_", "away_avg_",
        "home_elo", "away_elo", "elo_",
        "home_shot_conv", "away_shot_conv",
        "home_defensive", "away_defensive",
        "home_possession_q", "away_possession_q",
        "home_discipline", "away_discipline",
        "home_goal_diff", "away_goal_diff",
        "home_advantage",
        "attack_vs_defense",
        "home_x_elo",
        "form_x_attack",
        "home_rest", "away_rest",
    ]

    # ✅ Nech jen market_value_diff_scaled (eliminuj duplicity)
    market_keep = {"market_value_diff_scaled"}

    forbidden = {
        "goals_home", "goals_away", "home_win",
        "points_home", "points_away", "fixture_id", "result",
        # Duplicitní market_value:
        "market_value_diff", "market_value_home", "market_value_away",
    }

    features = []
    for col in df.columns:
        if df[col].dtype == 'object': continue
        if 'date' in col.lower(): continue
        if col in forbidden: continue
        if col in market_keep:
            features.append(col)
            continue
        if any(col.startswith(p) for p in allowed_prefixes):
            features.append(col)

    return sorted(features)


# =============================================================================
# 2. SYMETRICKÝ FEATURE SELECTION (v4 klíčová novinka)
# =============================================================================

def select_features_symmetric(X_vals, y, feature_cols, n_pairs=N_PAIRS, n_neutral=N_NEUTRAL):
    """
    SYMETRICKÝ výběr features zaručující rovnováhu Home/Away.

    PROČ SYMETRIE POMÁHÁ:
    Model musí vidět OBĚ strany stejně:
      - Jak útočí domácí?   → home_avg_xg
      - Jak útočí hosté?    → away_avg_xg  ← v3 CHYBĚLO!
      - Jak brání domácí?   → home_avg_interceptions
      - Jak brání hosté?    → away_avg_interceptions  ← v3 CHYBĚLO!

    Bez away features model vidí jen domácí tým → predikuje Home příliš

    ALGORITMUS:
      1. Spočítej konsensuální rank (F + MI + RF) pro každou feature
      2. Neutral features (elo_diff, market_value_diff_scaled...):
         vyber top n_neutral
      3. Párové features (home_avg_X + away_avg_X):
         seřaď páry podle avg(rank_home, rank_away)
         vyber top n_pairs párů → přidej OBA (home i away)
      4. Výsledek: n_neutral + n_pairs*2 features, perfektní symetrie
    """
    k_total = n_neutral + n_pairs * 2
    print(f"\n  🔍 Symetrický Feature Selection...")
    print(f"     Cíl: {n_neutral} neutral + {n_pairs} párů×2 = {k_total} features")

    # Preprocessing pro feature selection
    imp = SimpleImputer(strategy='mean')
    scl = StandardScaler()
    X_prep = scl.fit_transform(imp.fit_transform(X_vals))

    # --- Ranky všech 3 metod ---
    sel_f = SelectKBest(f_classif, k='all')
    sel_f.fit(X_prep, y)
    f_scores = np.nan_to_num(sel_f.scores_, nan=0.0)
    f_ranks  = pd.Series(f_scores, index=feature_cols).rank(ascending=False)

    mi_scores = mutual_info_classif(X_prep, y, random_state=42)
    mi_ranks  = pd.Series(mi_scores, index=feature_cols).rank(ascending=False)

    rf_sel = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_sel.fit(X_prep, y)
    rf_ranks = pd.Series(rf_sel.feature_importances_, index=feature_cols).rank(ascending=False)

    consensus = ((f_ranks + mi_ranks + rf_ranks) / 3).sort_values()

    # --- KROK 1: Neutral features (nespárované) ---
    # Tyto features nemají přirozený home/away pár
    neutral_candidates = [
        'elo_diff', 'market_value_diff_scaled', 'home_x_elo',
        'home_advantage', 'attack_vs_defense', 'form_x_attack_home',
        'home_elo', 'away_elo', 'elo_x_market',
        'home_goal_diff_avg', 'away_goal_diff_avg',
    ]
    neutral_available = [f for f in neutral_candidates if f in feature_cols]
    # Seřaď podle konsensuálního ranku, vyber top n_neutral
    neutral_ranked = sorted(neutral_available, key=lambda f: consensus.get(f, 999))
    selected_neutral = neutral_ranked[:n_neutral]

    # --- KROK 2: Párové features (home_avg_X + away_avg_X) ---
    home_feats = [f for f in feature_cols if f.startswith('home_avg_')]
    pair_scores = []
    for hf in home_feats:
        af = hf.replace('home_avg_', 'away_avg_')
        if af in feature_cols:
            h_rank = consensus.get(hf, 999)
            a_rank = consensus.get(af, 999)
            avg_rank = (h_rank + a_rank) / 2
            pair_scores.append((hf, af, h_rank, a_rank, avg_rank))

    # Seřaď páry podle průměrného ranku
    pair_scores.sort(key=lambda x: x[4])

    selected_pairs = []
    for hf, af, h_rank, a_rank, avg_rank in pair_scores[:n_pairs]:
        selected_pairs.append((hf, af, h_rank, a_rank, avg_rank))

    # Finální seznam
    selected = selected_neutral.copy()
    for hf, af, _, _, _ in selected_pairs:
        selected.extend([hf, af])

    # --- VÝPIS ---
    ratio = len(X_vals) / len(selected)
    print(f"  ✅ Vybráno {len(selected)} features  |  ratio {len(X_vals)}:{len(selected)} = {ratio:.0f}:1")

    home_cnt = sum(1 for f in selected if f.startswith('home_avg_'))
    away_cnt = sum(1 for f in selected if f.startswith('away_avg_'))
    neut_cnt = len(selected) - home_cnt - away_cnt
    print(f"     Složení: {home_cnt} HOME + {away_cnt} AWAY + {neut_cnt} neutral")

    print(f"\n  📌 Neutral features ({n_neutral}):")
    for feat in selected_neutral:
        rank = consensus.get(feat, 999)
        print(f"    {'✅':3s} {feat:45s}  rank={rank:.1f}")

    print(f"\n  📌 Párové features ({n_pairs} párů):")
    print(f"  {'HOME':45s}  {'H-rank':>7}  {'AWAY':45s}  {'A-rank':>7}  {'Avg':>6}")
    print(f"  {'─'*115}")
    for hf, af, hr, ar, avg in selected_pairs:
        sig_h = "✅" if sel_f.pvalues_[feature_cols.index(hf)] < 0.05 else "⚠️ "
        sig_a = "✅" if sel_f.pvalues_[feature_cols.index(af)] < 0.05 else "⚠️ "
        print(f"  {sig_h} {hf[:43]:43s} {hr:7.1f}  {sig_a} {af[:43]:43s} {ar:7.1f}  {avg:6.1f}")

    return selected


# =============================================================================
# 3. PŘÍPRAVA DAT
# =============================================================================

def prepare_data(df, feature_cols):
    X = df[feature_cols].copy()
    conditions = [
        (df['goals_home'] > df['goals_away']),
        (df['goals_home'] == df['goals_away'])
    ]
    y_class   = np.select(conditions, [2, 1], default=0)
    y_goals_h = df["goals_home"].values
    y_goals_a = df["goals_away"].values

    unique, counts = np.unique(y_class, return_counts=True)
    print(f"\n  🎯 Distribuce výsledků:")
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(y_class) * 100
        bar = "█" * int(pct / 2)
        print(f"    {CLASS_LABELS[cls]:5s} ({cls}): {cnt:3d} ({pct:.1f}%) {bar}")
    print(f"    Baseline (vždy Home): {counts[unique==2][0]/len(y_class)*100:.1f}%")
    return X, y_class, y_goals_h, y_goals_a


# =============================================================================
# 4. MODELY
# =============================================================================

def build_voting_classifier():
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    clf_rf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf_gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
        # GBM nemá class_weight, sample_weight nejde předat přes VotingClassifier pipeline
        # Vyvažování zajišťují RF (class_weight='balanced') a LR (class_weight='balanced')
    )
    clf_lr = LogisticRegression(
        max_iter=2000, C=0.5,
        class_weight='balanced', random_state=42
    )
    voting = VotingClassifier(
        estimators=[('rf', clf_rf), ('gb', clf_gb), ('lr', clf_lr)],
        voting='soft'
    )
    return Pipeline([('prep', preprocessor), ('clf', voting)])


def build_xgboost_classifier():
    preprocessor = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    xgb_clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', objective='multi:softprob', num_class=3,
        random_state=42, n_jobs=-1
        # sample_weight se předá při fit() — XGBoost Pipeline to podporuje nativně
    )
    return Pipeline([('prep', preprocessor), ('xgb', xgb_clf)])


def build_regressors():
    prep_scaled = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    prep_tree = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    return {
        'poisson_h': Pipeline([('prep', prep_scaled),
                                ('poi', PoissonRegressor(alpha=0.5, max_iter=1000))]),
        'poisson_a': Pipeline([('prep', prep_scaled),
                                ('poi', PoissonRegressor(alpha=0.5, max_iter=1000))]),
        'xgb_h':     Pipeline([('prep', prep_tree),
                                ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=3, subsample=0.8,
                                                     random_state=42, n_jobs=-1))]),
        'xgb_a':     Pipeline([('prep', prep_tree),
                                ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=3, subsample=0.8,
                                                     random_state=42, n_jobs=-1))]),
    }


# =============================================================================
# 5. VALIDACE
# =============================================================================

# =============================================================================
# 5. DRAW RECALL BOOST
# =============================================================================

def predict_with_draw_boost(proba, threshold=DRAW_BOOST_THRESHOLD):
    """
    Predikuj Draw pokud P(Draw) >= threshold AND P(Draw) > min(P(Away), P(Home)).
    Druhá podmínka zabrání přebití jasných Away/Home výsledků (Liverpool vs Wolves apod.)
    """
    preds = []
    for p in proba:
        p_away, p_draw, p_home = p[0], p[1], p[2]
        if p_draw >= threshold and p_draw > min(p_away, p_home):
            preds.append(1)
        else:
            preds.append(int(np.argmax(p)))
    return np.array(preds)


def find_optimal_draw_threshold(proba_list, y_true_list, thresholds=None):
    """Najde threshold maximalizující F1-macro přes všechny CV foldy."""
    if thresholds is None:
        thresholds = np.arange(0.15, 0.40, 0.02)
    all_proba = np.vstack(proba_list)
    all_y     = np.concatenate(y_true_list)
    results = []
    for thr in thresholds:
        preds = predict_with_draw_boost(all_proba, threshold=thr)
        f1    = f1_score(all_y, preds, average='macro', zero_division=0)
        draw_recall = confusion_matrix(all_y, preds, labels=[0,1,2])[1,1] / \
                      max((all_y == 1).sum(), 1) * 100
        results.append((thr, f1, draw_recall))
    best = max(results, key=lambda x: x[1])
    return best[0], best[1], results


# =============================================================================
# 5b. VALIDACE
# =============================================================================

def validate_classifier(model, X, y_class, model_name, n_splits=5):
    print(f"\n  🔄 {model_name} — TimeSeriesSplit (n={n_splits})...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, f1s, cms = [], [], []
    all_proba, all_y = [], []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X.iloc[train_idx], y_class[train_idx]
        X_te, y_te = X.iloc[test_idx],  y_class[test_idx]

        if 'XGBoost' in model_name:
            # XGBoost Pipeline podporuje sample_weight nativně
            sw = compute_sample_weight('balanced', y_tr)
            model.fit(X_tr, y_tr, xgb__sample_weight=sw)
        else:
            # VotingClassifier: sklearn blokuje per-estimator sample_weight bez metadata routing
            # RF a LR mají class_weight='balanced' — GBM vyvažují ostatní dva
            model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average='macro', zero_division=0))
        cms.append(confusion_matrix(y_te, preds, labels=[0, 1, 2]))

        if hasattr(model, 'predict_proba'):
            all_proba.append(model.predict_proba(X_te))
            all_y.append(y_te)

        print(f"    Fold {fold_i}: acc={accs[-1]:.3f} | F1={f1s[-1]:.3f}"
              f"  (train={len(train_idx)}, test={len(test_idx)})")

    avg_acc = np.mean(accs)
    avg_f1  = np.mean(f1s)
    std_acc = np.std(accs)
    avg_cm  = np.mean(cms, axis=0).astype(int)

    away_recall = avg_cm[0, 0] / max(avg_cm[0].sum(), 1) * 100
    draw_recall = avg_cm[1, 1] / max(avg_cm[1].sum(), 1) * 100
    home_recall = avg_cm[2, 2] / max(avg_cm[2].sum(), 1) * 100
    print(f"\n  📊 {model_name}: Accuracy={avg_acc:.4f} ± {std_acc:.4f}  |  F1={avg_f1:.4f}")
    print(f"  Per-class recall:  Away={away_recall:.0f}%  Draw={draw_recall:.0f}%  Home={home_recall:.0f}%")
    print(f"  Confusion Matrix (predikce →):")
    print(f"    {'':8s} Away  Draw  Home")
    for i, label in enumerate(['Away', 'Draw', 'Home']):
        print(f"    {label:8s} {avg_cm[i][0]:4d}  {avg_cm[i][1]:4d}  {avg_cm[i][2]:4d}")

    # --- THRESHOLD TUNING ---
    best_thr = DRAW_BOOST_THRESHOLD
    if all_proba:
        print(f"\n  🎯 Draw threshold tuning...")
        best_thr, best_thr_f1, thr_results = find_optimal_draw_threshold(all_proba, all_y)

        thr_results_sorted = sorted(thr_results, key=lambda x: x[1], reverse=True)[:5]
        print(f"  {'Threshold':>10}  {'F1-macro':>10}  {'Draw recall':>12}")
        print(f"  {'─'*36}")
        for thr, f1, dr in thr_results_sorted:
            marker = " ← optimal" if thr == best_thr else ""
            print(f"  {thr:>10.2f}  {f1:>10.4f}  {dr:>11.0f}%{marker}")

        all_p     = np.vstack(all_proba)
        all_y_arr = np.concatenate(all_y)
        thr_preds = predict_with_draw_boost(all_p, threshold=best_thr)
        thr_cm    = confusion_matrix(all_y_arr, thr_preds, labels=[0,1,2])
        thr_away  = thr_cm[0,0] / max(thr_cm[0].sum(), 1) * 100
        thr_draw  = thr_cm[1,1] / max(thr_cm[1].sum(), 1) * 100
        thr_home  = thr_cm[2,2] / max(thr_cm[2].sum(), 1) * 100
        print(f"\n  📈 S threshold={best_thr:.2f}:  F1={best_thr_f1:.4f}"
              f"  |  Away={thr_away:.0f}%  Draw={thr_draw:.0f}%  Home={thr_home:.0f}%")

    # Fold detaily pro training_log.json
    n_folds = len(accs)
    # train sizes: tscv splits — rekonstruujeme ze split sizes
    fold_details = []
    tscv_tmp = TimeSeriesSplit(n_splits=n_folds)
    for i, (tr_idx, te_idx) in enumerate(tscv_tmp.split(X)):
        fold_details.append({
            "fold": i + 1,
            "train": len(tr_idx),
            "test": len(te_idx),
            "accuracy": round(accs[i], 3),
            "f1": round(f1s[i], 3),
        })

    recall_dict = {
        "away": f"{away_recall:.0f}%",
        "draw": f"{draw_recall:.0f}%",
        "home": f"{home_recall:.0f}%",
    }

    return avg_acc, avg_f1, std_acc, best_thr, fold_details, recall_dict


def validate_regressors(regs, X, y_goals_h, y_goals_a, n_splits=5):
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
    poi_avg = (res['poisson_h'] + res['poisson_a']) / 2
    xgb_avg = (res['xgb_h']    + res['xgb_a'])    / 2
    print(f"\n  {'Model':15s}  {'MAE home':>10s}  {'MAE away':>10s}  {'MAE avg':>10s}")
    print(f"  {'─'*50}")
    print(f"  {'Poisson':15s}  {res['poisson_h']:>10.3f}  {res['poisson_a']:>10.3f}  {poi_avg:>10.3f}")
    print(f"  {'XGBoost':15s}  {res['xgb_h']:>10.3f}  {res['xgb_a']:>10.3f}  {xgb_avg:>10.3f}")
    best = 'Poisson' if poi_avg <= xgb_avg else 'XGBoost'
    print(f"  🥇 Nejlepší regresor: {best}")
    return res, best


# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================

def print_feature_importance(model_pipeline, feature_cols, model_name, top_n=15):
    try:
        final_step = model_pipeline.steps[-1][1]
        importances = None
        if hasattr(final_step, 'named_estimators_'):
            for name, est in final_step.named_estimators_.items():
                if hasattr(est, 'feature_importances_'):
                    importances = est.feature_importances_
                    print(f"    (ze sub-modelu: {name})")
                    break
        elif hasattr(final_step, 'feature_importances_'):
            importances = final_step.feature_importances_
        if importances is None or len(importances) != len(feature_cols):
            return

        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}) \
                   .sort_values('importance', ascending=False)

        print(f"\n  🏆 Top {top_n} features ({model_name}):")
        print(f"  {'Feature':45s}  {'Imp':>6}  {'H/A':>4}")
        print(f"  {'─'*60}")
        for _, row in imp_df.head(top_n).iterrows():
            side = "🏠" if row['feature'].startswith('home_') else \
                   "✈️ " if row['feature'].startswith('away_') else "⚖️ "
            bar = "█" * int(row['importance'] * 300)
            print(f"  {row['feature']:45s}  {row['importance']:.4f}  {side} {bar}")

        # Symetrie check
        home_imp = imp_df[imp_df['feature'].str.startswith('home_avg_')]['importance'].sum()
        away_imp = imp_df[imp_df['feature'].str.startswith('away_avg_')]['importance'].sum()
        if away_imp > 0:
            ratio = home_imp / away_imp
            print(f"\n  ⚖️  Poměr Home/Away importance: {ratio:.2f}x  "
                  f"({'OK' if 0.7 < ratio < 1.5 else '⚠️  asymetrické'})")

    except Exception as e:
        print(f"    ⚠️  Feature importance chyba: {str(e)[:60]}")


# =============================================================================
# 7. LOGGING
# =============================================================================

def save_log(results):
    existing = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                existing = json.load(f)
        except:
            pass
    existing.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "v4_symmetric",
        "results": results
    })
    with open(LOG_FILE, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"\n  📝 Log uložen: {LOG_FILE}")


# =============================================================================
# 8. HLAVNÍ FUNKCE
# =============================================================================

def train_models():
    print("=" * 70)
    print("🚀 STEP3 v4: TRÉNINK MODELŮ + SYMETRICKÝ HOME/AWAY VÝBĚR")
    print("=" * 70)

    # --- Načtení dat ---
    print("\n📥 Načítám data...")
    df = pd.read_sql("SELECT * FROM prepared_datasets ORDER BY match_date ASC", engine)
    if df.empty:
        print("❌ Tabulka prázdná. Spusť step2.")
        return

    print(f"  Načteno: {len(df)} zápasů  ({df['match_date'].min()} → {df['match_date'].max()})")

    all_feature_cols = get_feature_columns(df)
    print(f"  Kandidátní features (bez leakage, bez duplikátů): {len(all_feature_cols)}")

    X_all, y_class, y_goals_h, y_goals_a = prepare_data(df, all_feature_cols)

    # ==========================================================================
    # FEATURE SELECTION - SYMETRICKÝ
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 SYMETRICKÝ FEATURE SELECTION")
    print("=" * 70)

    selected_features = select_features_symmetric(
        X_all.values, y_class, all_feature_cols,
        n_pairs=N_PAIRS, n_neutral=N_NEUTRAL
    )
    X = X_all[selected_features]
    joblib.dump(selected_features, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print(f"\n  💾 feature_cols.pkl uložen ({len(selected_features)} features)")

    # ==========================================================================
    # A) VOTING CLASSIFIER
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🗳️  A) VOTING CLASSIFIER (RF + GBM + LR)")
    print("=" * 70)

    voting_pipe = build_voting_classifier()
    acc_v, f1_v, std_v, thr_v, folds_v, recall_v = validate_classifier(voting_pipe, X, y_class, "Voting")
    print("\n  🔥 Finální trénink...")
    voting_pipe.fit(X, y_class)  # Voting: RF+LR mají class_weight='balanced'
    print_feature_importance(voting_pipe, selected_features, "Voting (RF)")

    # ==========================================================================
    # B) XGBOOST CLASSIFIER
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🌲 B) XGBOOST CLASSIFIER")
    print("=" * 70)

    xgb_pipe = build_xgboost_classifier()
    acc_x, f1_x, std_x, thr_x, folds_x, recall_x = validate_classifier(xgb_pipe, X, y_class, "XGBoost")
    print("\n  🔥 Finální trénink...")
    sw_all = compute_sample_weight('balanced', y_class)
    xgb_pipe.fit(X, y_class, xgb__sample_weight=sw_all)  # XGBoost Pipeline to podporuje
    print_feature_importance(xgb_pipe, selected_features, "XGBoost")

    # ==========================================================================
    # C) SROVNÁNÍ
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🏆 C) SROVNÁNÍ MODELŮ")
    print("=" * 70)

    baseline = 0.431
    print(f"\n  {'Model':22s} {'Accuracy':>10s} {'±Std':>8s} {'F1-macro':>10s} {'vs Baseline':>12s}")
    print("  " + "─" * 65)
    print(f"  {'v3 Voting (15 feat)':22s} {'0.4651':>10s} {'0.1291':>8s} {'0.4101':>10s} {'+0.034':>12s}")
    print(f"  {'─'*65}")
    print(f"  {'Voting (RF+GBM+LR)':22s} {acc_v:>10.4f} {std_v:>8.4f} {f1_v:>10.4f} {acc_v-baseline:>+12.4f}")
    print(f"  {'XGBoost':22s} {acc_x:>10.4f} {std_x:>8.4f} {f1_x:>10.4f} {acc_x-baseline:>+12.4f}")
    print(f"  {'Baseline (always Home)':22s} {baseline:>10.4f} {'N/A':>8s} {'N/A':>10s} {'±0':>12s}")

    best_clf_name = "Voting" if f1_v >= f1_x else "XGBoost"
    best_clf      = voting_pipe if f1_v >= f1_x else xgb_pipe
    best_thr      = thr_v if f1_v >= f1_x else thr_x
    print(f"\n  🥇 Nejlepší klasifikátor: {best_clf_name} (F1={max(f1_v, f1_x):.4f})")
    print(f"  🎯 Optimální Draw threshold: {best_thr:.2f}")

    # ==========================================================================
    # D) REGRESORY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("⚽ D) REGRESORY PRO POČET GÓLŮ")
    print("=" * 70)
    print(f"  (v3 reference: Poisson MAE=0.877 | XGBoost MAE=0.982)")

    regs = build_regressors()
    reg_results, best_reg = validate_regressors(regs, X, y_goals_h, y_goals_a)
    print("\n  🔥 Finální trénink regresorů...")
    for name, reg in regs.items():
        y_t = y_goals_h if name.endswith('h') else y_goals_a
        reg.fit(X, y_t)

    # ==========================================================================
    # E) ULOŽENÍ
    # ==========================================================================
    print("\n" + "=" * 70)
    print("💾 E) UKLÁDÁNÍ MODELŮ")
    print("=" * 70)

    to_save = {
        "voting_classifier.pkl":   voting_pipe,
        "xgb_classifier.pkl":      xgb_pipe,
        "best_classifier.pkl":     best_clf,
        "poisson_home_goals.pkl":  regs['poisson_h'],
        "poisson_away_goals.pkl":  regs['poisson_a'],
        "xgb_home_goals.pkl":      regs['xgb_h'],
        "xgb_away_goals.pkl":      regs['xgb_a'],
        "feature_cols.pkl":        selected_features,
        "draw_threshold.pkl":      best_thr,
    }

    print()
    for fname, obj in to_save.items():
        path = os.path.join(MODEL_DIR, fname)
        joblib.dump(obj, path)
        kb = os.path.getsize(path) / 1024
        print(f"  ✅ {fname:35s} ({kb:.0f} KB)")

    # ==========================================================================
    # F) LOG + SHRNUTÍ
    # ==========================================================================
    results_log = {
        "n_samples":        len(df),
        "n_features_all":   len(all_feature_cols),
        "n_features_sel":   len(selected_features),
        "n_pairs":          N_PAIRS,
        "n_neutral":        N_NEUTRAL,
        "selected":         selected_features,
        "baseline":         round(baseline, 4),
        "voting": {
            "accuracy": round(acc_v, 4), "f1": round(f1_v, 4), "draw_thr": round(thr_v, 2),
            "folds": folds_v,
            "per_class_recall": recall_v,
        },
        "xgboost": {
            "accuracy": round(acc_x, 4), "f1": round(f1_x, 4), "draw_thr": round(thr_x, 2),
            "folds": folds_x,
            "per_class_recall": recall_x,
        },
        "best_classifier":  best_clf_name,
        "draw_threshold":   round(best_thr, 2),
        "regressors":       {k: round(v, 3) for k, v in reg_results.items()},
        "best_regressor":   best_reg,
    }
    save_log(results_log)

    avg_poi = (reg_results['poisson_h'] + reg_results['poisson_a']) / 2
    avg_xgb = (reg_results['xgb_h']    + reg_results['xgb_a'])    / 2

    print("\n" + "=" * 70)
    print("✅ HOTOVO! FINÁLNÍ PŘEHLED")
    print("=" * 70)
    print(f"\n  Dataset:            {len(df)} zápasů")
    print(f"  Features:           {len(all_feature_cols)} → {len(selected_features)}")
    print(f"  Složení:            {sum(1 for f in selected_features if f.startswith('home_avg_'))} HOME"
          f" + {sum(1 for f in selected_features if f.startswith('away_avg_'))} AWAY"
          f" + {sum(1 for f in selected_features if not f.startswith(('home_avg_','away_avg_')))} neutral")
    print(f"  Sample/feature:     {len(df)/len(selected_features):.0f}:1")
    print(f"  Voting:             acc={acc_v:.4f}  F1={f1_v:.4f}  ({acc_v-baseline:+.4f} vs baseline)  thr={thr_v:.2f}")
    print(f"  XGBoost:            acc={acc_x:.4f}  F1={f1_x:.4f}  ({acc_x-baseline:+.4f} vs baseline)  thr={thr_x:.2f}")
    print(f"  Nejlepší model:     {best_clf_name}")
    print(f"  Draw threshold:     {best_thr:.2f}  (uložen do draw_threshold.pkl)")
    print(f"  MAE góly:           {avg_poi:.3f} (Poisson)  |  {avg_xgb:.3f} (XGBoost)")
    print(f"  Modely uloženy:     {MODEL_DIR}")
    print()


if __name__ == "__main__":
    train_models()