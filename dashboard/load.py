"""Load saved artefacts and score one hold-out row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from models import load_pipeline, load_split, response_labels
from prep import LOW_MAX, artefact_dir

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}

KEY_FEATURES = [
    "Ins_Age",
    "BMI",
    "Ht",
    "Wt",
    "Product_Info_2",
    "medical_keyword_count",
    "missingness_count",
]


def wp3_dir(root: Path | None = None) -> Path:
    return artefact_dir("wp3", root)


def wp4_dir(root: Path | None = None) -> Path:
    return artefact_dir("wp4", root)


def load_metrics(root: Path | None = None) -> dict[str, Any]:
    path = wp3_dir(root) / "metrics.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_comparison(root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(wp3_dir(root) / "model_comparison.csv")


def load_best_model(root: Path | None = None) -> str:
    return (wp3_dir(root) / "best_model.txt").read_text(encoding="utf-8").strip()


def load_holdout_frame(root: Path | None = None) -> pd.DataFrame:
    arts = load_split(root)
    out = arts["X_test"].copy()
    out.insert(0, "Id", arts["id_test"].to_numpy())
    out["Response"] = arts["y_test"].to_numpy()
    return out


def load_explanation_cases(root: Path | None = None) -> list[dict[str, Any]]:
    path = wp4_dir(root) / "explanation_cases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def load_lime_cases(root: Path | None = None) -> dict[str, Any]:
    path = wp4_dir(root) / "lime_cases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict_application(pipe, row: pd.DataFrame) -> dict[str, Any]:
    feat = row.drop(columns=[c for c in ("Id", "Response") if c in row.columns])
    proba = pipe.predict_proba(feat)[0]
    pred = int(pipe.predict(feat)[0])
    labels = response_labels(pipe.named_steps["clf"])
    proba_by_class = {int(lab): float(proba[i]) for i, lab in enumerate(labels)}
    p_high = sum(proba_by_class[c] for c in proba_by_class if c > LOW_MAX)
    return {
        "pred": pred,
        "proba": proba_by_class,
        "p_high": float(p_high),
        "risk_band": "High (5-8)" if pred > LOW_MAX else "Low (1-4)",
    }
