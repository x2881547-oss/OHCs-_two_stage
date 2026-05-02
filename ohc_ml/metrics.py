import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def class_weights_alpha(labels: np.ndarray, alpha: float):
    series = pd.Series(labels)
    counts = series.value_counts()
    weight_map = {label: (1.0 / float(count)) ** alpha for label, count in counts.items()}
    weights = series.map(weight_map).values.astype(float)
    return weights / np.mean(weights)


def two_stage_combine(p_i, p_br_cond):
    p_non_i = 1.0 - p_i
    p_br = p_non_i * p_br_cond
    p_cl = p_non_i * (1.0 - p_br_cond)
    return np.vstack([p_cl, p_br, p_i]).T


def brier_mse_multiclass(y_true_int, proba3):
    y = np.zeros_like(proba3)
    y[np.arange(len(y_true_int)), y_true_int] = 1.0
    return float(np.mean(np.sum((proba3 - y) ** 2, axis=1)))


def prob_mae_rmse(y_true_int, proba3):
    y = np.zeros_like(proba3)
    y[np.arange(len(y_true_int)), y_true_int] = 1.0
    diff = proba3 - y
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


def safe_auc(y_true_bin, y_score):
    if np.unique(y_true_bin).size < 2:
        return np.nan, None, None
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    return float(auc(fpr, tpr)), fpr, tpr


def stats_table(df_folds: pd.DataFrame):
    def stats_col(series: pd.Series):
        return pd.Series(
            {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            }
        )

    return (
        df_folds.drop(columns=["fold"])
        .apply(stats_col, axis=0)
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
