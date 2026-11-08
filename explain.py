"""SHAP and LIME for the three saved pipelines. Case choice uses XGBoost."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from models import load_pipeline, load_split, response_labels
from prep import LOW_MAX, RANDOM_STATE, ROOT, artefact_dir

N_GLOBAL = 1000
N_BACKGROUND = 200
BATCH_SIZE = 200
MODEL_NAMES = ("xgboost", "logistic_regression", "random_forest")
PREFIX = {"xgboost": "xgb", "random_forest": "rf", "logistic_regression": "lr"}


def clean_feature_name(name: str) -> str:
    if name.startswith("num__"):
        return name[len("num__") :]
    if name.startswith("cat__"):
        rest = name[len("cat__") :]
        if rest.startswith("Product_Info_2_"):
            return "Product_Info_2=" + rest[len("Product_Info_2_") :]
        return rest
    return name


def transformed_matrix(pipe, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocess = pipe.named_steps["preprocess"]
    matrix = np.asarray(preprocess.transform(X), dtype=float)
    names = [clean_feature_name(n) for n in preprocess.get_feature_names_out()]
    return matrix, names


def sample_indices(n: int, k: int, seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, n)
    return np.sort(rng.choice(n, size=k, replace=False))


def mean_abs_shap_multiclass(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=-1)
        return stacked.mean(axis=(0, -1))
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=(0, -1))
    return np.abs(arr).mean(axis=0)


def predicted_class_shap(shap_values: Any, class_index: np.ndarray) -> np.ndarray:
    if isinstance(shap_values, list):
        n, f = shap_values[0].shape
        out = np.zeros((n, f))
        for i, c in enumerate(class_index):
            out[i] = shap_values[int(c)][i]
        return out
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        n, f, _ = arr.shape
        out = np.zeros((n, f))
        for i, c in enumerate(class_index):
            out[i] = arr[i, :, int(c)]
        return out
    return arr


def top_features_from_importance(
    importance: np.ndarray, feature_names: list[str], k: int = 20
) -> list[dict[str, Any]]:
    order = np.argsort(importance)[::-1][:k]
    return [
        {"feature": feature_names[i], "mean_abs_shap": float(importance[i])}
        for i in order
    ]


def save_bar_plot(
    importance: np.ndarray,
    feature_names: list[str],
    path: Path,
    title: str,
    top_k: int = 20,
) -> None:
    order = np.argsort(importance)[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = importance[order][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_tree_shap_batched(explainer, X: np.ndarray, batch_size: int = BATCH_SIZE):
    parts = []
    for start in range(0, len(X), batch_size):
        parts.append(explainer.shap_values(X[start : start + batch_size]))
    if isinstance(parts[0], list):
        n_classes = len(parts[0])
        return [np.vstack([p[c] for p in parts]) for c in range(n_classes)]
    arr0 = np.asarray(parts[0])
    if arr0.ndim == 3:
        return np.concatenate(parts, axis=0)
    return np.vstack(parts)


def class_index_for(clf, pred_label: int) -> int:
    return response_labels(clf).index(int(pred_label))


def select_explanation_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    ids: np.ndarray,
    low_max: int = LOW_MAX,
) -> list[dict[str, Any]]:
    """Six hold-out rows. High-risk probability is columns for classes 5..8."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ids = np.asarray(ids)
    high_cols = list(range(low_max, y_proba.shape[1]))
    p_high = y_proba[:, high_cols].sum(axis=1)
    p_low = 1.0 - p_high
    true_high = y_true > low_max
    pred_high = y_pred > low_max
    cases: list[dict[str, Any]] = []

    def pick(mask: np.ndarray, score: np.ndarray, label: str) -> None:
        chosen_idxs = {c["index"] for c in cases}
        idx = np.array([i for i in np.where(mask)[0] if i not in chosen_idxs])
        if len(idx) == 0:
            return
        chosen = int(idx[np.argmax(score[idx])])
        cases.append(
            {
                "label": label,
                "index": chosen,
                "Id": int(ids[chosen]),
                "y_true": int(y_true[chosen]),
                "y_pred": int(y_pred[chosen]),
                "p_high": float(p_high[chosen]),
            }
        )

    pick(true_high & pred_high, p_high, "correct_high_risk")
    pick(~true_high & ~pred_high, p_low, "correct_low_risk")
    pick(true_high & ~pred_high, p_high, "misclassified_high_to_low")
    pick(~true_high & pred_high, p_low, "misclassified_low_to_high")

    taken = {c["index"] for c in cases}
    rare_idx = np.array([i for i in np.where(np.isin(y_true, [3, 4]))[0] if i not in taken])
    if len(rare_idx):
        rng = np.random.default_rng(RANDOM_STATE)
        chosen = int(rng.choice(rare_idx))
        cases.append(
            {
                "label": "rare_class_3_or_4",
                "index": chosen,
                "Id": int(ids[chosen]),
                "y_true": int(y_true[chosen]),
                "y_pred": int(y_pred[chosen]),
                "p_high": float(p_high[chosen]),
            }
        )

    pick((y_true == 8) & (y_pred == 8), p_high, "typical_class_8")

    seen = set()
    unique = []
    for case in cases:
        if case["index"] in seen:
            continue
        seen.add(case["index"])
        unique.append(case)

    fill_i = 0
    while len(unique) < 6 and fill_i < 100:
        remaining = np.array([i for i in range(len(y_true)) if i not in seen])
        if len(remaining) == 0:
            break
        if fill_i % 2 == 0:
            chosen = int(remaining[np.argmax(p_high[remaining])])
            label = "extra_high_confidence"
        else:
            chosen = int(remaining[np.argmin(np.abs(p_high[remaining] - 0.5))])
            label = "extra_borderline"
        seen.add(chosen)
        unique.append(
            {
                "label": label,
                "index": chosen,
                "Id": int(ids[chosen]),
                "y_true": int(y_true[chosen]),
                "y_pred": int(y_pred[chosen]),
                "p_high": float(p_high[chosen]),
            }
        )
        fill_i += 1

    return unique[:6]


def _lime_training_matrix(X_train: pd.DataFrame):
    levels = sorted(X_train["Product_Info_2"].astype(str).unique())
    mapping = {level: i for i, level in enumerate(levels)}
    mapped = X_train.copy()
    mapped["Product_Info_2"] = mapped["Product_Info_2"].astype(str).map(mapping)
    medians = mapped.median(numeric_only=True)
    matrix = mapped.fillna(medians).to_numpy(dtype=float)
    inverse = {i: level for level, i in mapping.items()}
    return matrix, list(X_train.columns), levels, medians, inverse


def _lime_row(X: pd.DataFrame, levels: list[str], medians: pd.Series) -> np.ndarray:
    # Same code book as the training matrix. A map built on this row alone
    # would send every product level to 0, which LIME then reads back as A1.
    mapping = {level: i for i, level in enumerate(levels)}
    frame = X.copy()
    frame["Product_Info_2"] = frame["Product_Info_2"].astype(str).map(mapping)
    return frame.fillna(medians).to_numpy(dtype=float)


def _lime_predict_fn(pipe, columns: list[str], inverse: dict[int, str]):
    def predict_fn(arr: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(arr, columns=columns)
        codes = frame["Product_Info_2"].round().astype(int).clip(0, max(inverse))
        frame["Product_Info_2"] = codes.map(inverse).fillna(inverse[0])
        return pipe.predict_proba(frame)

    return predict_fn


def run_global_shap_tree(
    model_name: str,
    pipe,
    X_test: pd.DataFrame,
    out_dir: Path,
    with_beeswarm: bool = False,
    n_global: int | None = None,
) -> dict[str, Any]:
    clf = pipe.named_steps["clf"]
    X_test_t, feature_names = transformed_matrix(pipe, X_test)

    n_use = n_global if n_global is not None else N_GLOBAL
    if model_name == "random_forest" and n_global is None:
        n_use = 100

    idx = sample_indices(len(X_test_t), n_use)
    X_sample = X_test_t[idx]
    pred = np.asarray(clf.predict(X_sample), dtype=int)
    labels = response_labels(clf)
    class_index = np.array([labels.index(int(v)) for v in pred])

    print(f"  TreeExplainer SHAP for {model_name} on {len(X_sample)} rows...")
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    shap_values = compute_tree_shap_batched(explainer, X_sample)

    importance = mean_abs_shap_multiclass(shap_values)
    top = top_features_from_importance(importance, feature_names, 20)
    prefix = PREFIX[model_name]
    save_bar_plot(
        importance,
        feature_names,
        out_dir / f"{prefix}_shap_global_bar.png",
        f"{model_name}: mean |SHAP| (top 20)",
    )
    with open(out_dir / f"{prefix}_top_features.json", "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2)

    if with_beeswarm:
        local_for_pred = predicted_class_shap(shap_values, class_index)
        top_idx = np.argsort(importance)[::-1][:15]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            local_for_pred[:, top_idx],
            X_sample[:, top_idx],
            feature_names=[feature_names[i] for i in top_idx],
            show=False,
            max_display=15,
        )
        plt.title(f"{model_name}: SHAP beeswarm (predicted class)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    return {"feature_names": feature_names, "top_features": top, "explainer": explainer}


def run_global_shap_linear(pipe, X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: Path):
    clf = pipe.named_steps["clf"]
    X_train_t, feature_names = transformed_matrix(pipe, X_train)
    X_test_t, _ = transformed_matrix(pipe, X_test)
    X_bg = X_train_t[sample_indices(len(X_train_t), N_BACKGROUND)]
    X_sample = X_test_t[sample_indices(len(X_test_t), N_GLOBAL)]

    print(f"  LinearExplainer SHAP on {len(X_sample)} rows...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
        explainer = shap.LinearExplainer(clf, masker)
        shap_values = explainer.shap_values(X_sample)

    importance = mean_abs_shap_multiclass(shap_values)
    top = top_features_from_importance(importance, feature_names, 20)
    save_bar_plot(
        importance,
        feature_names,
        out_dir / "lr_shap_global_bar.png",
        "logistic_regression: mean |SHAP| (top 20)",
    )
    with open(out_dir / "lr_top_features.json", "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2)
    return {"feature_names": feature_names, "top_features": top, "explainer": explainer}


def _shap_vector(explainer, row: np.ndarray, class_index: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(row.reshape(1, -1))

    if isinstance(sv, list):
        values = np.asarray(sv[class_index][0])
        base = explainer.expected_value
        base_val = float(np.asarray(base)[class_index]) if np.ndim(base) else float(base)
        return values, base_val

    arr = np.asarray(sv)
    if arr.ndim == 3:
        base = explainer.expected_value
        return arr[0, :, class_index], float(np.asarray(base)[class_index])
    return arr[0], float(np.asarray(explainer.expected_value).reshape(-1)[0])


def save_waterfall(explainer, row: np.ndarray, feature_names: list[str], class_index: int, path: Path, title: str):
    values, base_val = _shap_vector(explainer, row, class_index)
    explanation = shap.Explanation(
        values=values,
        base_values=base_val,
        data=row,
        feature_names=feature_names,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def run_lime(pipes: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, cases: list[dict], out_dir: Path):
    X_lime, columns, levels, medians, inverse = _lime_training_matrix(X_train)
    cat_idx = [columns.index("Product_Info_2")]
    store: dict[str, Any] = {"cases": []}

    for model_name, pipe in pipes.items():
        labels = response_labels(pipe.named_steps["clf"])
        predict_fn = _lime_predict_fn(pipe, columns, inverse)
        explainer = LimeTabularExplainer(
            training_data=X_lime,
            feature_names=columns,
            class_names=[str(i) for i in labels],
            categorical_features=cat_idx,
            mode="classification",
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )
        for case_i, case in enumerate(cases):
            row_df = X_test.iloc[[case["index"]]]
            row_vec = _lime_row(row_df, levels, medians)[0]
            pred = int(pipe.predict(row_df)[0])
            label_idx = labels.index(pred)
            print(f"  LIME {model_name} case {case_i} ({case['label']})...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp = explainer.explain_instance(
                    row_vec,
                    predict_fn,
                    num_features=10,
                    labels=(label_idx,),
                )
            fig = exp.as_pyplot_figure(label=label_idx)
            fig.suptitle(
                f"LIME {model_name} | {case['label']} | Id={case['Id']} | pred={pred}"
            )
            out_png = out_dir / f"lime_{model_name}_case{case_i}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            store["cases"].append(
                {
                    "model": model_name,
                    "case_index": case_i,
                    "label": case["label"],
                    "Id": case["Id"],
                    "y_true": case["y_true"],
                    "y_pred": pred,
                    "weights": [
                        {"feature": feat, "weight": float(w)}
                        for feat, w in exp.as_list(label=label_idx)
                    ],
                    "image": out_png.name,
                }
            )

    with open(out_dir / "lime_cases.json", "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    return store


def run_explain(root: Path | str | None = None) -> dict[str, Any]:
    out_dir = artefact_dir("wp4", root)
    out_dir.mkdir(parents=True, exist_ok=True)

    arts = load_split(root)
    X_train, X_test = arts["X_train"], arts["X_test"]
    y_test = arts["y_test"].to_numpy()
    ids = arts["id_test"].to_numpy()
    pipes = {name: load_pipeline(name, root) for name in MODEL_NAMES}

    xgb = pipes["xgboost"]
    y_pred = np.asarray(xgb.predict(X_test), dtype=int)
    y_proba = xgb.predict_proba(X_test)
    # ResponseXGB keeps booster columns in 0..7 order, which is Response 1..8.
    cases = select_explanation_cases(y_test, y_pred, y_proba, ids)
    with open(out_dir / "explanation_cases.json", "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "low_max": LOW_MAX}, f, indent=2)
    print(f"Selected {len(cases)} explanation cases")

    print("Global SHAP: XGBoost")
    xgb_shap = run_global_shap_tree("xgboost", xgb, X_test, out_dir, with_beeswarm=True)
    print("Global SHAP: logistic regression")
    lr_shap = run_global_shap_linear(pipes["logistic_regression"], X_train, X_test, out_dir)
    print("Global SHAP: random forest")
    rf_shap = run_global_shap_tree(
        "random_forest", pipes["random_forest"], X_test, out_dir, with_beeswarm=False
    )

    print("Local SHAP waterfalls")
    X_test_t_xgb, feat_xgb = transformed_matrix(xgb, X_test)
    X_test_t_lr, feat_lr = transformed_matrix(pipes["logistic_regression"], X_test)
    xgb_clf = xgb.named_steps["clf"]
    lr_pipe = pipes["logistic_regression"]
    lr_clf = lr_pipe.named_steps["clf"]

    for case_i, case in enumerate(cases):
        idx = case["index"]
        save_waterfall(
            xgb_shap["explainer"],
            X_test_t_xgb[idx],
            feat_xgb,
            class_index_for(xgb_clf, int(y_pred[idx])),
            out_dir / f"xgb_shap_waterfall_case{case_i}.png",
            f"SHAP waterfall XGB | {case['label']} | Id={case['Id']}",
        )
        if case_i < 2:
            lr_pred = int(lr_pipe.predict(X_test.iloc[[idx]])[0])
            save_waterfall(
                lr_shap["explainer"],
                X_test_t_lr[idx],
                feat_lr,
                class_index_for(lr_clf, lr_pred),
                out_dir / f"lr_shap_waterfall_case{case_i}.png",
                f"SHAP waterfall LR | {case['label']} | Id={case['Id']}",
            )

    print("LIME")
    lime_store = run_lime(pipes, X_train, X_test, cases, out_dir)

    summary = {
        "n_global": N_GLOBAL,
        "n_background": N_BACKGROUND,
        "n_cases": len(cases),
        "cases": cases,
        "xgb_top_features": xgb_shap["top_features"],
        "lr_top_features": lr_shap["top_features"],
        "rf_top_features": rf_shap["top_features"],
    }
    with open(out_dir / "wp4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved explanations to {out_dir.relative_to(ROOT).as_posix()}")
    return {"out_dir": out_dir, "cases": cases, "summary": summary, "lime": lime_store}


if __name__ == "__main__":
    run_explain()
