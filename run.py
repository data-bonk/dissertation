"""Prepare the split, tune the three models, then write SHAP and LIME."""

from __future__ import annotations

import argparse

from explain import run_explain
from models import run_models
from prep import fit_save_artefacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prep, model tuning, and explanations")
    parser.add_argument("--quick", action="store_true", help="Skip the hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    print("prep")
    fit_save_artefacts()
    print("models")
    run_models(n_iter=args.n_iter, cv_folds=args.cv_folds, quick=args.quick)
    print("explain")
    run_explain()


if __name__ == "__main__":
    main()
