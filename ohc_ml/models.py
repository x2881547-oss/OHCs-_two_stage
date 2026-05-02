import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

from .metrics import class_weights_alpha, two_stage_combine


def svc_params_from_json(meta):
    best = meta.get("best_params", {})
    c2 = float(best.get("C2", 3.0))
    gamma2 = best.get("gamma2", "scale")
    gamma2 = float(gamma2) if isinstance(gamma2, (int, float)) else gamma2
    c1 = float(best.get("C1", 3.0))
    gamma1 = best.get("gamma1", "scale")
    gamma1 = float(gamma1) if isinstance(gamma1, (int, float)) else gamma1
    return {"stage1": {"C": c1, "gamma": gamma1}, "stage2": {"C": c2, "gamma": gamma2}}


def model_params_used_from_json(meta):
    used = meta.get("best_model_params_used", meta.get("best_model_params"))
    if used is None:
        raise ValueError("Cannot find best_model_params_used or best_model_params in JSON.")
    if "hidden_layer_sizes" in used and isinstance(used["hidden_layer_sizes"], list):
        used["hidden_layer_sizes"] = tuple(used["hidden_layer_sizes"])
    return used


def eval_fold_two_stage(model_kind, params, Xtr, ytr_lab, Xte, seed=7, alpha_class=1.0):
    start = time.perf_counter()

    if model_kind == "SVC":
        p1 = params["stage1"]
        p2 = params["stage2"]
        y1 = (ytr_lab == "I").astype(int)
        w1 = class_weights_alpha(np.where(ytr_lab == "I", "I", "nonI"), alpha_class)
        scaler1 = StandardScaler()
        Xtr1 = scaler1.fit_transform(Xtr)
        Xte1 = scaler1.transform(Xte)
        m1 = SVC(kernel="rbf", probability=True, random_state=seed, C=p1["C"], gamma=p1["gamma"])
        m1.fit(Xtr1, y1, sample_weight=w1)
        p_i = m1.predict_proba(Xte1)[:, 1]

        mask2 = np.isin(ytr_lab, ["Cl", "Br"])
        y2_lab = ytr_lab[mask2]
        y2 = (y2_lab == "Br").astype(int)
        w2 = class_weights_alpha(y2_lab, alpha_class)
        scaler2 = StandardScaler()
        Xtr2 = scaler2.fit_transform(Xtr[mask2])
        Xte2 = scaler2.transform(Xte)
        m2 = SVC(kernel="rbf", probability=True, random_state=seed, C=p2["C"], gamma=p2["gamma"])
        m2.fit(Xtr2, y2, sample_weight=w2)
        p_br_cond = m2.predict_proba(Xte2)[:, 1]

    elif model_kind in ["RF", "XGB"]:
        cls = RandomForestClassifier if model_kind == "RF" else xgb.XGBClassifier
        y1 = (ytr_lab == "I").astype(int)
        w1 = class_weights_alpha(np.where(ytr_lab == "I", "I", "nonI"), alpha_class)
        m1 = cls(**params)
        m1.fit(Xtr, y1, sample_weight=w1)
        p_i = m1.predict_proba(Xte)[:, 1]

        mask2 = np.isin(ytr_lab, ["Cl", "Br"])
        y2_lab = ytr_lab[mask2]
        y2 = (y2_lab == "Br").astype(int)
        w2 = class_weights_alpha(y2_lab, alpha_class)
        m2 = cls(**params)
        m2.fit(Xtr[mask2], y2, sample_weight=w2)
        p_br_cond = m2.predict_proba(Xte)[:, 1]

    elif model_kind == "MLP":
        y1 = (ytr_lab == "I").astype(int)
        w1 = class_weights_alpha(np.where(ytr_lab == "I", "I", "nonI"), alpha_class)
        scaler1 = StandardScaler()
        Xtr1 = scaler1.fit_transform(Xtr)
        Xte1 = scaler1.transform(Xte)
        m1 = MLPClassifier(**params)
        try:
            m1.fit(Xtr1, y1, sample_weight=w1)
        except TypeError:
            m1.fit(Xtr1, y1)
        p_i = m1.predict_proba(Xte1)[:, 1]

        mask2 = np.isin(ytr_lab, ["Cl", "Br"])
        y2_lab = ytr_lab[mask2]
        y2 = (y2_lab == "Br").astype(int)
        w2 = class_weights_alpha(y2_lab, alpha_class)
        scaler2 = StandardScaler()
        Xtr2 = scaler2.fit_transform(Xtr[mask2])
        Xte2 = scaler2.transform(Xte)
        m2 = MLPClassifier(**params)
        try:
            m2.fit(Xtr2, y2, sample_weight=w2)
        except TypeError:
            m2.fit(Xtr2, y2)
        p_br_cond = m2.predict_proba(Xte2)[:, 1]
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    fit_done = time.perf_counter()
    proba3 = two_stage_combine(p_i, p_br_cond)
    pred_done = time.perf_counter()
    return proba3, fit_done - start, pred_done - fit_done
