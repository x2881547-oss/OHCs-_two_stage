import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

from .data import load_dataset
from .metrics import class_weights_alpha, two_stage_combine


def _score_cv(model_kind, params, X, y_label, y_int, splits, seed, alpha_class, lambda_logloss):
    scores = []
    for tr_idx, te_idx in splits:
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr_lab = y_label[tr_idx]
        yte = y_int[te_idx]
        proba3 = _predict_proba_for_tuning(model_kind, params, Xtr, ytr_lab, Xte, seed, alpha_class)
        pred = np.argmax(proba3, axis=1)
        score = recall_score(yte, pred, average="macro") - lambda_logloss * log_loss(yte, proba3, labels=[0, 1, 2])
        scores.append(float(score))
    return float(np.mean(scores))


def _predict_proba_for_tuning(model_kind, params, Xtr, ytr_lab, Xte, seed, alpha_class):
    if model_kind == "SVC":
        p1 = params.get("stage1", {"C": 3.0, "gamma": "scale"})
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
        p_br = m2.predict_proba(Xte2)[:, 1]
        return two_stage_combine(p_i, p_br)

    cls = {"RF": RandomForestClassifier, "XGB": xgb.XGBClassifier, "MLP": MLPClassifier}[model_kind]
    y1 = (ytr_lab == "I").astype(int)
    w1 = class_weights_alpha(np.where(ytr_lab == "I", "I", "nonI"), alpha_class)
    mask2 = np.isin(ytr_lab, ["Cl", "Br"])
    y2_lab = ytr_lab[mask2]
    y2 = (y2_lab == "Br").astype(int)
    w2 = class_weights_alpha(y2_lab, alpha_class)

    if model_kind == "MLP":
        scaler1 = StandardScaler()
        Xtr1 = scaler1.fit_transform(Xtr)
        Xte1 = scaler1.transform(Xte)
        m1 = cls(**params)
        try:
            m1.fit(Xtr1, y1, sample_weight=w1)
        except TypeError:
            m1.fit(Xtr1, y1)
        p_i = m1.predict_proba(Xte1)[:, 1]
        scaler2 = StandardScaler()
        Xtr2 = scaler2.fit_transform(Xtr[mask2])
        Xte2 = scaler2.transform(Xte)
        m2 = cls(**params)
        try:
            m2.fit(Xtr2, y2, sample_weight=w2)
        except TypeError:
            m2.fit(Xtr2, y2)
        p_br = m2.predict_proba(Xte2)[:, 1]
    else:
        m1 = cls(**params)
        m1.fit(Xtr, y1, sample_weight=w1)
        p_i = m1.predict_proba(Xte)[:, 1]
        m2 = cls(**params)
        m2.fit(Xtr[mask2], y2, sample_weight=w2)
        p_br = m2.predict_proba(Xte)[:, 1]
    return two_stage_combine(p_i, p_br)


def run_tune_svc(config):
    import optuna

    seed = int(config.get("seed", 7))
    alpha_class = float(config.get("alpha_class", 1.0))
    tune_splits = int(config.get("tune_splits", 5))
    tune_repeats = int(config.get("tune_repeats", 3))
    n_trials = int(config.get("n_trials_svc", 60))
    lambda_logloss = float(config.get("lambda_logloss", 0.05))
    out_root = _make_out(config["base_folder"], "SVC_TwoStage_Optuna")
    _, X, y_label, y_int = load_dataset(config["data_xlsx"])
    splits = list(RepeatedStratifiedKFold(tune_splits, tune_repeats, random_state=seed).split(X, y_int))

    def objective(trial):
        params = {
            "stage1": {"C": 3.0, "gamma": "scale"},
            "stage2": {
                "C": trial.suggest_float("C2", 0.1, 100.0, log=True),
                "gamma": trial.suggest_float("gamma2", 1e-4, 1.0, log=True),
            },
        }
        return _score_cv("SVC", params, X, y_label, y_int, splits, seed, alpha_class, lambda_logloss)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    _write_study_outputs(study, out_root, {"best_params": study.best_params, "best_score": study.best_value})
    return out_root


def run_tune_other_models(config):
    import optuna

    seed = int(config.get("seed", 7))
    alpha_class = float(config.get("alpha_class", 1.0))
    n_splits = int(config.get("tune_splits", 5))
    n_repeats = int(config.get("tune_repeats", 5))
    lambda_logloss = float(config.get("lambda_logloss", 0.05))
    out_root = _make_out(config["base_folder"], "Optuna_Tuning_3Models")
    _, X, y_label, y_int = load_dataset(config["data_xlsx"])
    splits = list(RepeatedStratifiedKFold(n_splits, n_repeats, random_state=seed).split(X, y_int))

    jobs = [
        ("RandomForest", "RF", int(config.get("n_trials_rf", 50))),
        ("XGBoost", "XGB", int(config.get("n_trials_xgb", 50))),
        ("MLP", "MLP", int(config.get("n_trials_mlp", 30))),
    ]
    for name, kind, n_trials in jobs:
        model_out = os.path.join(out_root, f"{name}_Tuning")
        os.makedirs(model_out, exist_ok=True)

        def objective(trial, model_kind=kind):
            params = _suggest_params(trial, model_kind, seed)
            return _score_cv(model_kind, params, X, y_label, y_int, splits, seed, alpha_class, lambda_logloss)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = _suggested_params_to_model_params(kind, study.best_params, seed)
        _write_study_outputs(
            study,
            model_out,
            {"best_params_raw": study.best_params, "best_model_params_used": best_params, "best_score": study.best_value},
        )
    return out_root


def _suggest_params(trial, model_kind, seed):
    if model_kind == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30, 40]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": seed,
        }
    if model_kind == "XGB":
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "n_jobs": -1,
            "random_state": seed,
        }
    return {
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes", [(64, 32), (64, 64), (128, 64), (128, 64, 32)]
        ),
        "activation": "relu",
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "max_iter": 400,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": seed,
    }


def _suggested_params_to_model_params(model_kind, params, seed):
    class DummyTrial:
        def __init__(self, values):
            self.values = values

        def suggest_int(self, name, *args, **kwargs):
            return self.values[name]

        def suggest_float(self, name, *args, **kwargs):
            return self.values[name]

        def suggest_categorical(self, name, choices):
            value = self.values[name]
            return tuple(value) if isinstance(value, list) else value

    return _suggest_params(DummyTrial(params), model_kind, seed)


def _write_study_outputs(study, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def _make_out(base_folder, prefix):
    out = os.path.join(base_folder, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    return out
