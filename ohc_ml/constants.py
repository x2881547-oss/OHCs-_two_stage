CLASSES = ["Cl", "Br", "I"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASSES)}

MASS = {
    "C": 12.0,
    "H": 1.00783,
    "N": 14.00307,
    "O": 15.99491,
    "P": 30.97376,
    "S": 31.97207,
    "Cl": 34.96885,
    "Br": 78.91834,
    "I": 126.90447,
}

FEATURES = [
    "MW_bb",
    "C",
    "O",
    "N",
    "S",
    "P",
    "Hbb_over_C",
    "O_over_C",
    "N_over_C",
    "S_over_C",
    "DBE_bb",
    "NOSC_bb",
    "DBEbb_minusO_over_C",
    "AImod_dd",
]
