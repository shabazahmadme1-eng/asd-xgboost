"""
Train the MediaPipe XGBoost ASD classifier, using the methodology from the
original Colab code (model1_xgboost.py) adapted to the MediaPipe feature set:

  * GroupShuffleSplit 20% subject hold-out "vault" (no window leakage)
  * GridSearchCV over XGBoost params with GroupKFold(5)
  * scale_pos_weight = (neg/pos) * 1.1
  * Decision-threshold optimization on the hold-out (F1 + a high-specificity option)

Also reports 5-fold subject-grouped CV AUC for a stable generalization estimate.

Label convention: ASD = 1, Typical = 0  (server reads predict_proba[:, 1]).
Writes:  xgboost_asd_mediapipe.json  and  train/mediapipe_threshold.json
Run:  python -m train.train_model
"""
import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, classification_report, recall_score)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MODEL = os.path.join(ROOT, "xgboost_asd_mediapipe.json")
OUT_THRESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe_threshold.json")

# Grid from the original code, lightly trimmed for the smaller MediaPipe set.
PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05],
    "n_estimators": [400, 800],
    "gamma": [0.5, 1.0],
    "reg_lambda": [5.0, 10.0],
    "subsample": [0.8],
    "colsample_bytree": [0.3, 0.5],
}


def subject_auc(groups_te, p, y_te):
    sp, st = {}, {}
    for s, pi, ti in zip(groups_te, p, y_te):
        sp.setdefault(s, []).append(pi)
        st[s] = ti
    ss = np.array([np.mean(sp[s]) for s in sp])
    stt = np.array([st[s] for s in sp])
    return roc_auc_score(stt, ss) if len(set(stt)) > 1 else float("nan")


def main():
    X = np.load(os.path.join(CACHE, "_X.npy"))
    y = np.load(os.path.join(CACHE, "_y.npy"))
    g = np.load(os.path.join(CACHE, "_groups.npy"), allow_pickle=True)
    print(f"Loaded X={X.shape}  ASD windows={int((y==1).sum())}  TD windows={int((y==0).sum())}  subjects={len(set(g))}")

    # ---- 20% hold-out vault ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(gss.split(X, y, groups=g))
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    Xte, yte, gte = X[te], y[te], g[te]
    print(f"Train subjects={len(set(gtr))}  Vault subjects={len(set(gte))}")

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1) * 1.1
    base = xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                             scale_pos_weight=spw, tree_method="hist",
                             n_jobs=-1, random_state=42)
    gkf = GroupKFold(n_splits=5)
    print("Running GridSearchCV (GroupKFold=5)...")
    gs = GridSearchCV(base, PARAM_GRID, scoring="f1", cv=gkf, n_jobs=-1, verbose=0)
    gs.fit(Xtr, ytr, groups=gtr)
    champ = gs.best_estimator_
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV F1: {gs.best_score_:.3f}")

    # ---- Threshold optimization on the vault ----
    p_te = champ.predict_proba(Xte)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(yte, (p_te >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    # High-specificity option: smallest threshold giving >=0.85 TD specificity.
    spec_t = None
    for t in np.arange(0.30, 0.96, 0.01):
        tn = ((p_te < t) & (yte == 0)).sum(); fp = ((p_te >= t) & (yte == 0)).sum()
        spec = tn / max(tn + fp, 1)
        if spec >= 0.85:
            spec_t = float(t); break

    print(f"\nVault window AUC={roc_auc_score(yte, p_te):.3f}  subject AUC={subject_auc(gte, p_te, yte):.3f}")
    print(f"F1-optimal threshold={best_t:.2f} (F1={best_f1:.3f})")
    print(f"High-specificity(>=0.85 TD) threshold={spec_t}")
    print("\nVault confusion @F1-opt:\n", confusion_matrix(yte, (p_te >= best_t).astype(int)))
    print("\n", classification_report(yte, (p_te >= best_t).astype(int), target_names=["TD", "ASD"]))

    # ---- Stable 5-fold subject CV AUC (whole dataset) for reporting ----
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tri, tei in sgkf.split(X, y, groups=g):
        m = xgb.XGBClassifier(**{**gs.best_params_, "scale_pos_weight": (y[tri]==0).sum()/max((y[tri]==1).sum(),1)*1.1,
                                 "tree_method": "hist", "n_jobs": -1, "random_state": 42,
                                 "eval_metric": "aucpr"})
        m.fit(X[tri], y[tri])
        aucs.append(subject_auc(g[tei], m.predict_proba(X[tei])[:, 1], y[tei]))
    print(f"\n5-fold subject-AUC (tuned params): {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")

    # ---- Final model on ALL data ----
    spw_all = (y == 0).sum() / max((y == 1).sum(), 1) * 1.1
    final = xgb.XGBClassifier(**{**gs.best_params_, "scale_pos_weight": spw_all,
                                 "tree_method": "hist", "n_jobs": -1, "random_state": 42,
                                 "eval_metric": "aucpr"})
    final.fit(X, y)
    final.save_model(OUT_MODEL)
    with open(OUT_THRESH, "w") as f:
        json.dump({"f1_optimal": best_t, "high_specificity": spec_t}, f, indent=2)
    print(f"\nSaved model -> {OUT_MODEL}")
    print(f"Saved thresholds -> {OUT_THRESH}")
    print(f"n_features_in_={final.n_features_in_}  classes_={list(final.classes_)}")


if __name__ == "__main__":
    main()
