import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ohc_ml.evaluate import run_final_evaluation
from ohc_ml.shap_svc import run_svc_shap
from ohc_ml.tune import run_tune_other_models, run_tune_svc


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Reproducible OHC two-stage ML workflow")
    parser.add_argument("command", choices=["tune-svc", "tune-other-models", "evaluate", "shap-svc"])
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "tune-svc":
        run_tune_svc(config)
    elif args.command == "tune-other-models":
        run_tune_other_models(config)
    elif args.command == "evaluate":
        run_final_evaluation(config)
    elif args.command == "shap-svc":
        run_svc_shap(config)


if __name__ == "__main__":
    main()
