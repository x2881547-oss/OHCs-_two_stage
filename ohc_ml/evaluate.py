import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .constants import CLASSES, FEATURES
from .data import load_dataset
from .metrics import brier_mse_multiclass, prob_mae_rmse, safe_auc, stats_table
from .models import eval_fold_two_stage, model_params_used_from_json, svc_params_from_json


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_out_root(base_folder: str, prefix="Final_Eval_5x20") -> str:
    out_root = os.path.join(base_folder, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_root, exist_ok=True)
    return out_root


def run_final_evaluation(config):
    required = ["base_folder", "data_xlsx", "svc_best_json", "rf_best_json", "xgb_best_json", "mlp_best_json"]
    for key in required:
        if not config.get(key):
            raise ValueError(f"Missing config key: {key}")

    seed = int(config.get("seed", 7))
    n_splits = int(config.get("n_splits", 5))
    n_repeats = int(config.get("n_repeats", 20))
    alpha_class = float(config.get("alpha_class", 1.0))

    out_root = make_out_root(config["base_folder"])
    df, X, y_label, y_int = load_dataset(config["data_xlsx"])
    df[["formula", "label"] + FEATURES].to_excel(os.path.join(out_root, "ML_input_combined.xlsx"), index=False)

    models = [
        ("SVC_TwoStage", "SVC", svc_params_from_json(load_json(config["svc_best_json"]))),
        ("RandomForest_TwoStage", "RF", model_params_used_from_json(load_json(config["rf_best_json"]))),
        ("XGBoost_TwoStage", "XGB", model_params_used_from_json(load_json(config["xgb_best_json"]))),
        ("DNN_MLP_TwoStage", "MLP", model_params_used_from_json(load_json(config["mlp_best_json"]))),
    ]

    splits = list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X, y_int))
    mean_rows = []
    mean_std_rows = []
    fullstats = {}

    for model_name, model_kind, params in models:
        print(f"[EVAL] {model_name}")
        model_out = os.path.join(out_root, model_name)
        os.makedirs(model_out, exist_ok=True)
        fold_rows = []
        conf_sum = np.zeros((3, 3), dtype=int)

        for fold_id, (tr_idx, te_idx) in enumerate(splits, start=1):
            proba3, train_t, pred_t = eval_fold_two_stage(
                model_kind,
                params,
                X[tr_idx],
                y_label[tr_idx],
                X[te_idx],
                seed=seed,
                alpha_class=alpha_class,
            )
            y_true = y_int[te_idx]
            y_pred = np.argmax(proba3, axis=1)
            rec_each = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
            aucs = {}
            for idx, cname in enumerate(CLASSES):
                auc_k, _, _ = safe_auc((y_true == idx).astype(int), proba3[:, idx])
                aucs[cname] = auc_k
            mae, rmse = prob_mae_rmse(y_true, proba3)
            conf_sum += confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            fold_rows.append(
                {
                    "fold": fold_id,
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    "balanced_accuracy": float(recall_score(y_true, y_pred, average="macro")),
                    "recall_Cl": float(rec_each[0]),
                    "recall_Br": float(rec_each[1]),
                    "recall_I": float(rec_each[2]),
                    "auc_macro": float(np.nanmean(list(aucs.values()))),
                    "auc_Cl": float(aucs["Cl"]),
                    "auc_Br": float(aucs["Br"]),
                    "auc_I": float(aucs["I"]),
                    "log_loss": float(log_loss(y_true, proba3, labels=[0, 1, 2])),
                    "brier_mse": brier_mse_multiclass(y_true, proba3),
                    "prob_mae": mae,
                    "prob_rmse": rmse,
                    "train_time_sec": float(train_t),
                    "predict_time_sec": float(pred_t),
                    "total_time_sec": float(train_t + pred_t),
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(os.path.join(model_out, "CV_fold_metrics.csv"), index=False, encoding="utf-8-sig")
        summary_df = stats_table(fold_df)
        summary_df.to_csv(os.path.join(model_out, "CV_summary.csv"), index=False, encoding="utf-8-sig")
        fullstats[model_name] = summary_df
        pd.DataFrame(conf_sum, index=CLASSES, columns=CLASSES).to_csv(
            os.path.join(model_out, "CV_confusion_matrix_sum.csv"), encoding="utf-8-sig"
        )

        means = fold_df.drop(columns=["fold"]).mean(numeric_only=True)
        stds = fold_df.drop(columns=["fold"]).std(numeric_only=True, ddof=1)
        mean_rows.append({"Model": model_name, **{col: float(means[col]) for col in means.index}})
        mean_std_rows.append({"Model": model_name, **{col: f"{means[col]:.4f} +/- {stds[col]:.4f}" for col in means.index}})

    mean_df = pd.DataFrame(mean_rows)
    mean_std_df = pd.DataFrame(mean_std_rows)
    mean_df.to_csv(os.path.join(out_root, "Table3_like_mean.csv"), index=False, encoding="utf-8-sig")
    mean_std_df.to_csv(os.path.join(out_root, "Table3_like_mean_std.csv"), index=False, encoding="utf-8-sig")
    mean_df.to_excel(os.path.join(out_root, "Table3_like_mean.xlsx"), index=False)
    mean_std_df.to_excel(os.path.join(out_root, "Table3_like_mean_std.xlsx"), index=False)
    with pd.ExcelWriter(os.path.join(out_root, "Table3_like_fullstats.xlsx"), engine="openpyxl") as writer:
        for model_name, stats_df in fullstats.items():
            stats_df.to_excel(writer, sheet_name=model_name[:31], index=False)

    pd.DataFrame([{**config, "features": ", ".join(FEATURES)}]).to_csv(
        os.path.join(out_root, "run_config.csv"), index=False, encoding="utf-8-sig"
    )
    print("Output folder:", out_root)
    return out_root
