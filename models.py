"""Tune logistic regression, random forest, and XGBoost on the saved split."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from prep import (
    LOW_MAX,
    N_CLASSES,
    RANDOM_STATE,
    ROOT,
    artefact_dir,
    build_preprocessor,
    high_low_risk,
)

LABELS = list(range(1, N_CLASSES + 1))
# One thread per fit. Nested threading with the search makes the trees drift.
MODEL_JOBS = 1
SEARCH_JOBS = -1


class ResponseXGB(XGBClassifier):
    """Fit on Response-1. predict() and response_labels_ use 1..8."""

    def fit(self, X, y, **kwargs):
        y_in = np.asarray(y, dtype=int)
        kwargs.pop("sample_weight", None)
        weights = compute_sample_weight(class_weight="balanced", y=y_in)
        super().fit(X, y_in - 1, sample_weight=weights, **kwargs)
        self.response_labels_ = np.arange(1, N_CLASSES + 1)
        return self

    def predict(self, X, **kwargs):
        raw = super().predict(X, **kwargs)
        return np.asarray(raw, dtype=int) + 1


def response_labels(clf) -> list[int]:
    labels = getattr(clf, "response_labels_", None)
    if labels is None:
        labels = clf.classes_
    return [int(v) for v in labels]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_split(root: Path | str | None = None) -> dict[str, Any]:
    wp2 = artefact_dir("wp2", root)
    needed = wp2 / "X_train.parquet"
    if not needed.exists():
        raise FileNotFoundError(f"Missing {needed}. Run prep.py first.")
    return {
        "X_train": pd.read_parquet(wp2 / "X_train.parquet"),
        "X_test": pd.read_parquet(wp2 / "X_test.parquet"),
        "y_train": pd.read_parquet(wp2 / "y_train.parquet")["Response"].astype(int),
        "y_test": pd.read_parquet(wp2 / "y_test.parquet")["Response"].astype(int),
        "id_test": pd.read_parquet(wp2 / "id_test.parquet")["Id"].astype("int64"),
    }


def load_pipeline(name: str, root: Path | str | None = None):
    path = artefact_dir("wp3", root) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run models.py first.")
    return joblib.load(path)


def build_pipelines(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    linear = build_preprocessor(X_train, scale_numeric=True)
    trees = build_preprocessor(X_train, scale_numeric=False)
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", linear),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", trees),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced_subsample",
                        n_jobs=MODEL_JOBS,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "clf",
                    ResponseXGB(
                        objective="multi:softprob",
                        num_class=N_CLASSES,
                        n_estimators=200,
                        learning_rate=0.08,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        eval_metric="mlogloss",
                        n_jobs=MODEL_JOBS,
                        random_state=RANDOM_STATE,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def search_spaces() -> dict[str, dict]:
    return {
        "logistic_regression": {
            "clf__C": np.logspace(-2, 2, 8),
        },
        "random_forest": {
            "clf__n_estimators": [150, 250, 350],
            "clf__max_depth": [12, 20, 28, None],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.3],
        },
        "xgboost": {
            "clf__n_estimators": [150, 250, 350],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.05, 0.08, 0.12],
            "clf__min_child_weight": [1, 3, 5],
            "clf__subsample": [0.8, 0.9, 1.0],
        },
    }


def _align_proba(pipe: Pipeline, y_proba: np.ndarray) -> np.ndarray:
    """Columns in Response order 1..8. Missing classes stay zero."""
    labels = response_labels(pipe.named_steps["clf"])
    if labels == LABELS:
        return np.asarray(y_proba)
    aligned = np.zeros((len(y_proba), N_CLASSES), dtype=float)
    for i, lab in enumerate(labels):
        if 1 <= lab <= N_CLASSES:
            aligned[:, lab - 1] = y_proba[:, i]
    return aligned


def multiclass_metrics(y_true, y_pred, y_proba) -> dict[str, Any]:
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ovr_roc_auc_macro": float(auc),
        "ovr_gini_macro": float(2 * auc - 1),
    }


def binary_risk_metrics(y_true, y_pred, y_proba, low_max: int = LOW_MAX) -> dict[str, Any]:
    y_true_bin = high_low_risk(y_true, low_max=low_max)
    y_pred_bin = high_low_risk(y_pred, low_max=low_max)
    high_idx = [i for i, lab in enumerate(LABELS) if lab > low_max]
    y_score = y_proba[:, high_idx].sum(axis=1)
    auc = roc_auc_score(y_true_bin, y_score)
    return {
        "binary_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "binary_macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "binary_roc_auc": float(auc),
        "binary_gini": float(2 * auc - 1),
    }


def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    y_pred = np.asarray(pipe.predict(X), dtype=int)
    y_proba = _align_proba(pipe, pipe.predict_proba(X))
    if y_proba.shape[1] != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} probability columns, got {y_proba.shape[1]}")

    metrics = multiclass_metrics(y, y_pred, y_proba)
    metrics.update(binary_risk_metrics(y, y_pred, y_proba))
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred, labels=LABELS).tolist()
    metrics["classification_report"] = classification_report(
        y, y_pred, labels=LABELS, digits=3, zero_division=0
    )
    return {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}


def tune_model(
    name: str,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 8,
    cv_folds: int = 3,
) -> tuple[Pipeline, dict[str, Any]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=search_spaces()[name],
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        n_jobs=SEARCH_JOBS,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        search.fit(X_train, y_train)

    info = {
        "best_params": _jsonable(search.best_params_),
        "best_cv_macro_f1": float(search.best_score_),
        "cv_folds": cv_folds,
        "n_iter": n_iter,
    }
    return search.best_estimator_, info


def run_models(
    n_iter: int = 8,
    cv_folds: int = 3,
    quick: bool = False,
    out_dir: Path | str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir) if out_dir else artefact_dir("wp3", root)
    out_dir.mkdir(parents=True, exist_ok=True)

    arts = load_split(root)
    X_train, X_test = arts["X_train"], arts["X_test"]
    y_train, y_test = arts["y_train"], arts["y_test"]

    results: dict[str, Any] = {}
    rows = []

    for name, pipe in build_pipelines(X_train).items():
        print(f"\n=== {name} ===")
        if quick:
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            tune_info = {
                "best_params": "defaults (quick mode)",
                "best_cv_macro_f1": None,
                "cv_folds": cv_folds,
                "n_iter": 0,
            }
        else:
            best_pipe, tune_info = tune_model(
                name, pipe, X_train, y_train, n_iter=n_iter, cv_folds=cv_folds
            )

        holdout = evaluate_model(best_pipe, X_test, y_test)
        metrics = holdout["metrics"]
        joblib.dump(best_pipe, out_dir / f"{name}.joblib")

        rows.append(
            {
                "model": name,
                **{
                    k: v
                    for k, v in metrics.items()
                    if k not in ("confusion_matrix", "classification_report")
                },
                "cv_macro_f1_mean": tune_info["best_cv_macro_f1"],
                "best_params": (
                    tune_info["best_params"]
                    if isinstance(tune_info["best_params"], str)
                    else json.dumps(tune_info["best_params"])
                ),
            }
        )
        results[name] = {"pipeline": best_pipe, "tune": tune_info, "holdout": metrics}
        print(
            f"  hold-out accuracy={metrics['accuracy']:.3f}  "
            f"macro_f1={metrics['macro_f1']:.3f}  "
            f"OvR AUC={metrics['ovr_roc_auc_macro']:.3f}  "
            f"binary Gini={metrics['binary_gini']:.3f}"
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    summary = {
        "random_state": RANDOM_STATE,
        "n_iter": None if quick else n_iter,
        "cv_folds": cv_folds,
        "quick": quick,
        "low_max": LOW_MAX,
        "models": {
            name: {
                "tune": results[name]["tune"],
                "holdout": results[name]["holdout"],
            }
            for name in results
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)

    ranking = comparison.sort_values(
        ["macro_f1", "ovr_roc_auc_macro"], ascending=False
    ).reset_index(drop=True)
    best_name = str(ranking.loc[0, "model"])
    (out_dir / "best_model.txt").write_text(best_name + "\n", encoding="utf-8")

    results["comparison"] = comparison
    results["best_model"] = best_name
    results["out_dir"] = out_dir
    print(f"\nBest hold-out model (macro F1): {best_name}")
    print(f"Saved models to {out_dir.relative_to(ROOT).as_posix()}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune the three risk classifiers")
    parser.add_argument("--quick", action="store_true", help="Fit the default hyperparameters only")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    run_models(n_iter=args.n_iter, cv_folds=args.cv_folds, quick=args.quick)
