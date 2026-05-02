import re

import numpy as np
import pandas as pd

from .constants import CLASSES, FEATURES, LABEL_MAP, MASS


def norm_key(value: str) -> str:
    value = str(value).strip().lower()
    return re.sub(r"[\s_\-–—/()]+", "", value)


def find_col(df: pd.DataFrame, candidates):
    colmap = {norm_key(col): col for col in df.columns}
    for candidate in candidates:
        key = norm_key(candidate)
        if key in colmap:
            return colmap[key]
    return None


def get_classified_sheets(xlsx_path: str):
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    keep = [sheet for sheet in xl.sheet_names if "summary" not in sheet.lower()]
    classified = [sheet for sheet in keep if "classified" in sheet.lower()]
    if len(classified) >= 3:
        return classified[:3]
    if len(keep) < 3:
        raise ValueError(f"Need at least 3 data sheets in {xlsx_path}. Found: {xl.sheet_names}")
    return keep[:3]


def infer_label_from_counts(row) -> str:
    has_cl = row.get("Cl", 0) > 0
    has_br = row.get("Br", 0) > 0
    has_i = row.get("I", 0) > 0
    if has_cl and not has_br and not has_i:
        return "Cl"
    if has_br and not has_cl and not has_i:
        return "Br"
    if has_i and not has_cl and not has_br:
        return "I"
    return ""


def standardize_atom_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {}
    for col in df.columns:
        key = norm_key(col)
        if key in ["c", "h", "o", "n", "s", "p"]:
            mapping[col] = key.upper()
        elif key == "cl":
            mapping[col] = "Cl"
        elif key == "br":
            mapping[col] = "Br"
        elif key == "i":
            mapping[col] = "I"
    if mapping:
        df = df.rename(columns=mapping)

    if "IUPAC MW" not in df.columns:
        alt = find_col(df, ["IUPAC MW", "IUPAC_MW", "IUPACMW", "iupac mw"])
        if alt is not None:
            df = df.rename(columns={alt: "IUPAC MW"})
    if "IUPAC MW" not in df.columns:
        df["IUPAC MW"] = np.nan

    for col in ["C", "H", "O", "N", "S", "P", "Cl", "Br", "I"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["IUPAC MW"] = pd.to_numeric(df["IUPAC MW"], errors="coerce")
    return df


def add_backbone_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    C = df["C"].astype(float)
    H = df["H"].astype(float)
    O = df["O"].astype(float)
    N = df["N"].astype(float)
    S = df["S"].astype(float)
    P = df["P"].astype(float)
    Cl = df["Cl"].astype(float)
    Br = df["Br"].astype(float)
    I = df["I"].astype(float)
    X = Cl + Br + I

    df["H_bb"] = H + X
    df["MW_bb"] = df["IUPAC MW"] - Cl * MASS["Cl"] - Br * MASS["Br"] - I * MASS["I"] + X * MASS["H"]

    with np.errstate(divide="ignore", invalid="ignore"):
        df["O_over_C"] = np.where(C > 0, O / C, np.nan)
        df["N_over_C"] = np.where(C > 0, N / C, np.nan)
        df["S_over_C"] = np.where(C > 0, S / C, np.nan)
        df["Hbb_over_C"] = np.where(C > 0, df["H_bb"].astype(float) / C, np.nan)

    df["DBE_bb"] = (2 * C + N + P - df["H_bb"].astype(float) + 2) / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        df["NOSC_bb"] = 4.0 - (4 * C + df["H_bb"].astype(float) - 2 * O - 3 * N - 2 * S) / C
        df["DBEbb_minusO_over_C"] = np.where(C > 0, (df["DBE_bb"] - O) / C, np.nan)

    num = 1 + C - 0.5 * O - S - 0.5 * df["H_bb"].astype(float)
    den = C - 0.5 * O - N - S
    with np.errstate(divide="ignore", invalid="ignore"):
        df["AImod_dd"] = np.where(den != 0, num / den, np.nan)
    return df


def load_dataset(xlsx_path: str):
    dfs = []
    for sheet in get_classified_sheets(xlsx_path):
        df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
        df = standardize_atom_cols(df)

        formula_col = find_col(df, ["formula", "Formula", "molecular_formula", "molecular formula"])
        if formula_col is not None and formula_col != "formula":
            df = df.rename(columns={formula_col: "formula"})
        if "formula" not in df.columns:
            df["formula"] = ""

        df["label"] = df.apply(infer_label_from_counts, axis=1)
        df = df[df["label"].isin(CLASSES)].copy()
        dfs.append(add_backbone_features(df))

    combined = pd.concat(dfs, ignore_index=True)
    feature_df = combined[FEATURES].replace([np.inf, -np.inf], np.nan)
    combined = combined.loc[~feature_df.isna().any(axis=1)].copy()
    X = combined[FEATURES].values.astype(float)
    y_label = combined["label"].values
    y_int = combined["label"].map(LABEL_MAP).values.astype(int)
    return combined, X, y_label, y_int
