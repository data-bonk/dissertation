"""Load the labelled Prudential file, build the model matrix, and save the split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

RANDOM_STATE = 42
LOW_MAX = 4  # Response 1..4 is low risk; 5..8 is high risk
N_CLASSES = 8

# Dropped by name. These four are mostly empty; a missingness threshold would
# also catch columns we want to keep an indicator for.
DROP_SPARSE = [
    "Medical_History_10",
    "Medical_History_32",
    "Medical_History_24",
    "Medical_History_15",
]

MISSINGNESS_PREFIXES = (
    "Employment_Info_",
    "Insurance_History_",
    "Family_Hist_",
    "Medical_History_",
)


def artefact_dir(stage: str, root: Path | str | None = None) -> Path:
    base = Path(root) if root else ROOT
    return base / "artefacts" / stage


def load_labelled(data_dir: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_dir or DATA_DIR) / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    if df["Id"].duplicated().any():
        raise ValueError("Duplicate Id values in train.csv")
    return df


def high_low_risk(y: pd.Series | np.ndarray, low_max: int = LOW_MAX) -> np.ndarray:
    """1 when Response is above the low-risk band, else 0."""
    return (np.asarray(y) > low_max).astype(int)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    kw_cols = [c for c in out.columns if c.startswith("Medical_Keyword_")]
    if kw_cols:
        out["medical_keyword_count"] = out[kw_cols].sum(axis=1)

    miss_cols = [
        c
        for c in out.columns
        if c != "Response" and any(c.startswith(p) for p in MISSINGNESS_PREFIXES)
    ]
    if miss_cols:
        out["missingness_count"] = out[miss_cols].isna().sum(axis=1)

    # Raw product. Median imputation happens later, inside each model pipeline.
    if {"Ins_Age", "BMI"}.issubset(out.columns):
        out["age_x_bmi"] = out["Ins_Age"] * out["BMI"]
    return out


def add_missing_indicators(
    df: pd.DataFrame,
    columns: Iterable[str],
    suffix: str = "_was_missing",
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns and out[col].isna().any():
            out[f"{col}{suffix}"] = out[col].isna().astype(int)
    return out


def prepare_xy(
    df: pd.DataFrame,
    drop_sparse: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Id, features, Response. Indicators are chosen on this whole frame."""
    drop_sparse = DROP_SPARSE if drop_sparse is None else drop_sparse
    work = df.drop(columns=[c for c in drop_sparse if c in df.columns])

    ids = work["Id"]
    y = work["Response"].astype(int)
    X = work.drop(columns=["Id", "Response"])

    candidate_miss = [
        c
        for c in X.columns
        if any(c.startswith(p) for p in MISSINGNESS_PREFIXES) and X[c].isna().any()
    ]
    X = add_missing_indicators(X, candidate_miss)
    X = add_engineered_features(X)
    return X, y, ids


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """Median impute, optional scaling, one-hot Product_Info_2 at the end."""
    cat_cols = [
        c
        for c in X.columns
        if (
            pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or isinstance(X[c].dtype, pd.CategoricalDtype)
            or c == "Product_Info_2"
        )
    ]
    cat_cols = list(dict.fromkeys(cat_cols))
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple] = [("num", Pipeline(numeric_steps), num_cols)]
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    return train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_save_artefacts(
    data_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Write the untransformed split. Each model pipeline fits its own preprocessor."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    out_dir = Path(out_dir) if out_dir else artefact_dir("wp2")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_labelled(data_dir)
    X, y, ids = prepare_xy(df)
    X_train, X_test, y_train, y_test, id_train, id_test = stratified_split(
        X, y, ids, test_size=test_size, random_state=random_state
    )

    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    X_test.to_parquet(out_dir / "X_test.parquet", index=False)
    y_train.to_frame("Response").to_parquet(out_dir / "y_train.parquet", index=False)
    y_test.to_frame("Response").to_parquet(out_dir / "y_test.parquet", index=False)
    id_train.to_frame("Id").to_parquet(out_dir / "id_train.parquet", index=False)
    id_test.to_frame("Id").to_parquet(out_dir / "id_test.parquet", index=False)

    meta = {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "response_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_response_counts": {
            int(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "holdout_response_counts": {
            int(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "dropped_columns": list(DROP_SPARSE),
        "random_state": random_state,
        "test_size": test_size,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "out_dir": out_dir,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "meta": meta,
    }


if __name__ == "__main__":
    result = fit_save_artefacts()
    print(f"Saved split to {result['out_dir'].relative_to(ROOT).as_posix()}")
    print(result["meta"])
