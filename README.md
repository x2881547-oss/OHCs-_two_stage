# Two-stage ML workflow for organohalogenated compounds

This repository contains the machine-learning workflow used to classify
single-halogen organohalogenated compounds into `Cl`, `Br`, and `I` classes
from FT-ICR-MS molecular-formula descriptors.

The workflow includes:

- input table loading and formula-column normalization;
- backbone descriptor calculation after replacing halogen atoms with hydrogen;
- two-stage classification: `I` vs `non-I`, followed by `Br` vs `Cl`;
- hyperparameter tuning with Optuna;
- final repeated stratified cross-validation;
- optional SHAP analysis for the tuned SVC model.

## Repository layout

```text
OHCs_two_stage_ML/
  README.md
  requirements.txt
  config.example.json
  scripts/
    run_workflow.py
  ohc_ml/
    constants.py
    data.py
    metrics.py
    models.py
    tune.py
    evaluate.py
    shap_svc.py
```

Put the input Excel file and Optuna `best_params.json` files outside version
control if they are large or unpublished. For publication, either include a
small example dataset or describe where the dataset can be obtained.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Copy `config.example.json` to `config.json` and edit the paths:

```json
{
  "base_folder": "path/to/output/folder",
  "data_xlsx": "path/to/OHCs Top6 (frequency more than 5).xlsx",
  "svc_best_json": "path/to/SVC/best_params.json",
  "rf_best_json": "path/to/RF/best_params.json",
  "xgb_best_json": "path/to/XGB/best_params.json",
  "mlp_best_json": "path/to/MLP/best_params.json"
}
```

## Run

Tune SVC:

```bash
python scripts/run_workflow.py tune-svc --config config.json
```

Tune RandomForest, XGBoost, and DNN-MLP:

```bash
python scripts/run_workflow.py tune-other-models --config config.json
```

Final 4-model evaluation:

```bash
python scripts/run_workflow.py evaluate --config config.json
```

Final SVC SHAP analysis:

```bash
python scripts/run_workflow.py shap-svc --config config.json
```

## Method overview

### Backbone descriptors

For each molecular formula, halogen atoms are converted back to hydrogen on
the carbon backbone:

```text
H_bb = H + Cl + Br + I
MW_bb = MW - Cl_mass * Cl - Br_mass * Br - I_mass * I + H_mass * (Cl + Br + I)
```

The model then uses the following descriptors:

```text
MW_bb, C, O, N, S, P,
Hbb_over_C, O_over_C, N_over_C, S_over_C,
DBE_bb, NOSC_bb, DBEbb_minusO_over_C, AImod_dd
```

### Two-stage classification

The classifier is trained as two binary problems:

1. Stage 1: `I` vs `non-I`.
2. Stage 2: `Br` vs `Cl`, trained only on the `Cl` and `Br` subset.

The final class probabilities are composed as:

```text
P(I)  = P_stage1(I)
P(Br) = (1 - P_stage1(I)) * P_stage2(Br | non-I)
P(Cl) = (1 - P_stage1(I)) * (1 - P_stage2(Br | non-I))
```

## Notes for reproducibility

- All main workflows use the same engineered feature set and `SEED = 7` by
  default.
- The final evaluation uses repeated stratified CV, 5 folds x 20 repeats.
- SVC and MLP are scaled separately inside each stage to avoid leakage.
- Tree models use raw engineered descriptors.
- The two-stage probability definition is:
  - `P(I) = P_stage1(I)`
  - `P(Br) = (1 - P_stage1(I)) * P_stage2(Br | non-I)`
  - `P(Cl) = (1 - P_stage1(I)) * (1 - P_stage2(Br | non-I))`
