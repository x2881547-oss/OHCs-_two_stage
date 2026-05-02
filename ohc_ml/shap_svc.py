import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .constants import CLASSES, FEATURES
from .data import load_dataset
from .evaluate import load_json
from .metrics import class_weights_alpha
from .models import svc_params_from_json


def run_svc_shap(config):
    import shap

    seed = int(config.get("seed", 7))
    alpha_class = float(config.get("alpha_class", 1.0))
    n_bg_stage1 = int(config.get("n_bg_stage1", 120))
    n_expl_stage1 = int(config.get("n_expl_stage1", 350))
    n_bg_stage2 = int(config.get("n_bg_stage2", 120))
    n_expl_stage2 = int(config.get("n_expl_stage2", 350))
    kernel_nsamples = int(config.get("kernel_nsamples", 200))

    out_root = os.path.join(config["base_folder"], f"Final_SVC_TwoStage_SHAP_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_root, exist_ok=True)

    df, X, y_label, y_int = load_dataset(config["data_xlsx"])
    params = svc_params_from_json(load_json(config["svc_best_json"]))

    y1 = (y_label == "I").astype(int)
    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(X)
    m1 = SVC(kernel="rbf", probability=True, random_state=seed, C=params["stage1"]["C"], gamma=params["stage1"]["gamma"])
    m1.fit(X1, y1, sample_weight=class_weights_alpha(np.where(y_label == "I", "I", "nonI"), alpha_class))

    mask2 = np.isin(y_label, ["Cl", "Br"])
    X2_raw = X[mask2]
    y2_lab = y_label[mask2]
    y2 = (y2_lab == "Br").astype(int)
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(X2_raw)
    m2 = SVC(kernel="rbf", probability=True, random_state=seed, C=params["stage2"]["C"], gamma=params["stage2"]["gamma"])
    m2.fit(X2, y2, sample_weight=class_weights_alpha(y2_lab, alpha_class))

    pred1 = m1.predict(X1)
    pred2_all = m2.predict(scaler2.transform(X))
    final_pred = np.where(pred1 == 1, 2, np.where(pred2_all == 1, 1, 0))
    pd.DataFrame(confusion_matrix(y_int, final_pred, labels=[0, 1, 2]), index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_root, "training_confusion_matrix.csv"), encoding="utf-8-sig"
    )
    with open(os.path.join(out_root, "training_classification_report.txt"), "w", encoding="utf-8") as handle:
        handle.write(classification_report(y_int, final_pred, target_names=CLASSES))

    _run_kernel_shap(
        shap,
        m1.predict_proba,
        X1,
        df,
        out_root,
        "stage1_I_vs_nonI",
        class_index=1,
        n_bg=n_bg_stage1,
        n_expl=n_expl_stage1,
        nsamples=kernel_nsamples,
        seed=seed,
    )
    _run_kernel_shap(
        shap,
        m2.predict_proba,
        X2,
        df.loc[mask2].copy(),
        out_root,
        "stage2_Br_vs_Cl",
        class_index=1,
        n_bg=n_bg_stage2,
        n_expl=n_expl_stage2,
        nsamples=kernel_nsamples,
        seed=seed + 1,
    )
    print("Output folder:", out_root)
    return out_root


def _run_kernel_shap(shap, predict_proba, X_scaled, df_subset, out_root, tag, class_index, n_bg, n_expl, nsamples, seed):
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(X_scaled.shape[0], size=min(n_bg, X_scaled.shape[0]), replace=False)
    ex_idx = rng.choice(X_scaled.shape[0], size=min(n_expl, X_scaled.shape[0]), replace=False)
    explainer = shap.KernelExplainer(predict_proba, X_scaled[bg_idx])
    shap_values = explainer.shap_values(X_scaled[ex_idx], nsamples=nsamples)
    values = shap_values[class_index] if isinstance(shap_values, list) else shap_values[:, :, class_index]

    mean_abs = np.mean(np.abs(values), axis=0)
    rows = []
    raw_subset = df_subset.iloc[ex_idx]
    for idx, feat in enumerate(FEATURES):
        rho, pval = spearmanr(raw_subset[feat].values, values[:, idx], nan_policy="omit")
        rows.append({"feature": feat, "mean_abs_shap": float(mean_abs[idx]), "spearman_rho": rho, "spearman_p": pval})
    pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).to_csv(
        os.path.join(out_root, f"{tag}_shap_feature_importance.csv"), index=False, encoding="utf-8-sig"
    )
