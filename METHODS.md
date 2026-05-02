# Methods

## Objective

This workflow classifies single-halogen organohalogenated compounds detected by
FT-ICR-MS into `Cl`, `Br`, and `I` classes using molecular-formula descriptors.

## Data preprocessing

The input is an Excel workbook containing molecular formula information and
element counts. The workflow:

1. reads the data sheets while ignoring summary sheets;
2. standardizes element-count column names such as `C`, `H`, `O`, `N`, `S`,
   `P`, `Cl`, `Br`, and `I`;
3. infers the class label from the halogen counts;
4. keeps only single-halogen formulas assigned to `Cl`, `Br`, or `I`;
5. calculates backbone descriptors after replacing halogen atoms with hydrogen.

## Backbone feature calculation

For a formula containing `Cl`, `Br`, and/or `I`, the total halogen count is:

```text
X = Cl + Br + I
```

The hydrogen count and molecular weight of the inferred non-halogenated
backbone are:

```text
H_bb = H + X
MW_bb = MW - Cl_mass * Cl - Br_mass * Br - I_mass * I + H_mass * X
```

The final model features are:

```text
MW_bb, C, O, N, S, P,
Hbb_over_C, O_over_C, N_over_C, S_over_C,
DBE_bb, NOSC_bb, DBEbb_minusO_over_C, AImod_dd
```

## Model design

A two-stage classification design is used:

1. Stage 1 predicts `I` vs `non-I`.
2. Stage 2 predicts `Br` vs `Cl` using only the non-iodinated training subset.

Final three-class probabilities are composed from the two binary classifiers:

```text
P(I)  = P_stage1(I)
P(Br) = (1 - P_stage1(I)) * P_stage2(Br | non-I)
P(Cl) = (1 - P_stage1(I)) * (1 - P_stage2(Br | non-I))
```

The implemented model families are support vector classification, random
forest, XGBoost, and a scikit-learn multilayer perceptron.

## Evaluation

The final evaluation uses repeated stratified cross-validation with 5 folds and
20 repeats by default. Metrics include accuracy, macro F1, balanced accuracy,
class-specific recall, one-vs-rest AUC, log-loss, multiclass Brier error,
probability MAE/RMSE, and training/prediction time.
