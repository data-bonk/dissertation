"""
Streamlit dashboard for the insurance risk models.
From the project folder:
  python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.load import (  # noqa: E402
    KEY_FEATURES,
    MODEL_LABELS,
    load_best_model,
    load_comparison,
    load_explanation_cases,
    load_holdout_frame,
    load_lime_cases,
    load_metrics,
    load_pipeline,
    predict_application,
    wp4_dir,
)
from prep import ROOT as PROJECT_ROOT  # noqa: E402

st.set_page_config(page_title="XAI Insurance Risk", layout="wide")
st.html(
    """
    <style>
    [data-testid="stImageContainer"] img {
        max-height: calc(100vh - 80px); /* Display height - 80px due to the header */
        height: auto;
        object-fit: contain;
    }
    </style>
    """
)


@st.cache_resource
def cached_pipeline(model_name: str):
    return load_pipeline(model_name)


@st.cache_data
def cached_metrics():
    return load_metrics()


@st.cache_data
def cached_comparison():
    return load_comparison()


@st.cache_data
def cached_holdout():
    return load_holdout_frame()


@st.cache_data
def cached_cases():
    return load_explanation_cases()


@st.cache_data
def cached_lime():
    return load_lime_cases()


def page_overview() -> None:
    st.header("Project overview")
    st.markdown(
        """
**Research question:** How effective are explainable ML techniques (SHAP, LIME)
at improving interpretability of insurance risk prediction **without** significantly
reducing predictive performance?

**Data:** Public anonymised Prudential Life Insurance Assessment data (Kaggle).
Features are normalised or coded. Only the group meanings are known.

**Pipeline**
1. `prep.py` — labelled `train.csv`, stratified 80/20 split (`artefacts/wp2`)
2. `models.py` — logistic regression, random forest, XGBoost (`artefacts/wp3`)
3. `explain.py` — SHAP and LIME (`artefacts/wp4`)
4. this dashboard

Metrics use the labelled hold-out from `train.csv`. Unlabelled `test.csv` is not scored.
"""
    )
    best = load_best_model()
    st.info(f"Best hold-out model by macro F1: **{MODEL_LABELS.get(best, best)}**.")
    st.caption("Local SHAP and LIME figures are for XGBoost, which is the model those plots were built on.")


def score_bars(frame: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> alt.Chart:
    """Grouped bars. Scores share a 0-1 axis so models can be compared on each metric."""
    keep = [c for c in columns if c in frame.columns]
    long = (
        frame[keep]
        .rename(columns=labels)
        .reset_index()
        .melt(id_vars="model", var_name="metric", value_name="score")
    )
    long["model"] = long["model"].map(lambda name: MODEL_LABELS.get(name, name))
    model_order = [MODEL_LABELS.get(name, name) for name in frame.index]
    metric_order = [labels[c] for c in keep]
    color = alt.Color("model:N", sort=model_order, title=None)
    base = alt.Chart(long).encode(
        x=alt.X("metric:N", sort=metric_order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=model_order),
        y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
    )
    bars = base.mark_bar().encode(
        color=color,
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("score:Q", title="Score", format=".3f"),
        ],
    )
    tags = base.mark_text(dy=-8, fontSize=11).encode(
        text=alt.Text("score:Q", format=".3f"),
        color=alt.Color("model:N", sort=model_order, legend=None),
    )
    return (bars + tags).resolve_scale(color="shared").properties(height=340).interactive()


def page_performance() -> None:
    st.header("Model performance")
    comparison = cached_comparison()
    best = load_best_model()
    st.info(f"Best hold-out model (macro F1): **{MODEL_LABELS.get(best, best)}**")

    plot_df = comparison.set_index("model")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Multiclass metrics")
        st.altair_chart(
            score_bars(
                plot_df,
                ["accuracy", "macro_f1", "ovr_roc_auc_macro"],
                {
                    "accuracy": "Accuracy",
                    "macro_f1": "Macro F1",
                    "ovr_roc_auc_macro": "OvR ROC-AUC",
                },
            ),
            width="stretch",
        )
    with c2:
        st.subheader("High vs low risk")
        st.altair_chart(
            score_bars(
                plot_df,
                ["binary_roc_auc", "binary_gini"],
                {"binary_roc_auc": "ROC-AUC", "binary_gini": "Gini"},
            ),
            width="stretch",
        )

    st.subheader("Full comparison table")
    show = comparison.drop(columns=["best_params"], errors="ignore")
    st.dataframe(show, width="stretch")

    st.subheader("Confusion matrix")
    names = list(MODEL_LABELS.keys())
    model_choice = st.selectbox(
        "Model",
        options=names,
        format_func=lambda m: MODEL_LABELS[m],
        index=names.index(best) if best in names else 0,
    )
    cm = cached_metrics()["models"][model_choice]["holdout"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(range(1, 9))
    ax.set_yticklabels(range(1, 9))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(MODEL_LABELS[model_choice])
    for i in range(8):
        for j in range(8):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig)
    plt.close(fig)


def page_score() -> None:
    st.header("Score an application")
    holdout = cached_holdout()
    ids = holdout["Id"].tolist()

    mode = st.radio("Row source", ["Select Id", "Random hold-out row"], horizontal=True)
    if mode == "Select Id":
        selected_id = st.selectbox("Hold-out Id", options=ids)
    else:
        if st.button("Draw random row") or "random_id" not in st.session_state:
            st.session_state["random_id"] = int(holdout.sample(1)["Id"].iloc[0])
        selected_id = st.session_state["random_id"]
        st.write(f"Random Id: **{selected_id}**")

    row = holdout.loc[holdout["Id"] == selected_id].iloc[[0]]
    key_cols = [c for c in KEY_FEATURES if c in row.columns]
    st.subheader("Key features")
    st.dataframe(row[key_cols], width="stretch")

    model_name = st.selectbox(
        "Model",
        options=["xgboost", "random_forest", "logistic_regression"],
        format_func=lambda m: MODEL_LABELS[m],
        key="score_model",
    )
    result = predict_application(cached_pipeline(model_name), row)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Response", result["pred"])
    m2.metric("Risk band", result["risk_band"])
    m3.metric("P(high risk)", f"{result['p_high']:.3f}")
    st.caption(f"True hold-out Response: {int(row['Response'].iloc[0])}")
    st.caption("Risk band follows the predicted class. P(high risk) can sit on the other side of 0.5.")

    proba_df = pd.DataFrame(
        {"Response": list(result["proba"].keys()), "probability": list(result["proba"].values())}
    ).set_index("Response")
    st.subheader("Class probabilities")
    st.bar_chart(proba_df)


def page_explanations() -> None:
    st.header("Explanations (SHAP & LIME)")
    st.caption("Local explanations are a review aid. They do not change the hold-out metrics.")
    wp4 = wp4_dir()

    st.subheader("Global SHAP")
    global_choice = st.radio(
        "Global model",
        ["XGBoost", "Logistic Regression", "Random Forest"],
        horizontal=True,
    )
    global_map = {
        "XGBoost": "xgb_shap_global_bar.png",
        "Logistic Regression": "lr_shap_global_bar.png",
        "Random Forest": "rf_shap_global_bar.png",
    }
    img_path = wp4 / global_map[global_choice]
    if img_path.exists():
        st.image(str(img_path), width="stretch")
    else:
        st.warning(f"Missing {img_path.name}. Run python explain.py.")

    if global_choice == "XGBoost" and (wp4 / "xgb_shap_summary.png").exists():
        with st.expander("XGBoost SHAP beeswarm"):
            st.image(str(wp4 / "xgb_shap_summary.png"), width="stretch")

    cases_path = wp4 / "explanation_cases.json"
    if not cases_path.exists():
        st.warning("No explanation cases yet. Run python explain.py.")
        return

    st.subheader("Local explanations")
    cases = cached_cases()
    lime = cached_lime()
    labels = [
        f"{c['label']} (Id={c['Id']}, true={c['y_true']}, pred={c['y_pred']})" for c in cases
    ]
    choice = st.selectbox("Case", options=range(len(cases)), format_func=lambda i: labels[i])
    case = cases[choice]
    st.write(
        f"**{case['label']}** — Id `{case['Id']}`, true={case['y_true']}, "
        f"pred={case['y_pred']}, P(high)={case['p_high']:.3f}"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SHAP waterfall (XGBoost)**")
        shap_img = wp4 / f"xgb_shap_waterfall_case{choice}.png"
        if shap_img.exists():
            st.image(str(shap_img), width="stretch")
        else:
            st.warning("SHAP waterfall image missing.")
    with c2:
        st.markdown("**LIME (XGBoost)**")
        lime_img = wp4 / f"lime_xgboost_case{choice}.png"
        if lime_img.exists():
            st.image(str(lime_img), width="stretch")
        else:
            st.warning("LIME image missing.")

    xgb_lime = [
        e
        for e in lime.get("cases", [])
        if e["model"] == "xgboost" and e["case_index"] == choice
    ]
    if xgb_lime:
        st.markdown("**LIME feature weights (XGBoost)**")
        st.dataframe(pd.DataFrame(xgb_lime[0]["weights"]), width="stretch")

    if choice < 2:
        lr_shap = wp4 / f"lr_shap_waterfall_case{choice}.png"
        if lr_shap.exists():
            with st.expander("Logistic regression SHAP waterfall"):
                st.image(str(lr_shap), width="stretch")


def main() -> None:
    st.title("XAI Insurance Risk Prediction")
    needed = PROJECT_ROOT / "artefacts" / "wp3" / "metrics.json"
    if not needed.exists():
        st.error("No model artefacts yet. From the project folder run: python run.py")
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Section",
        [
            "Project overview",
            "Model performance",
            "Score an application",
            "Explanations",
        ],
    )
    if page == "Project overview":
        page_overview()
    elif page == "Model performance":
        page_performance()
    elif page == "Score an application":
        page_score()
    else:
        page_explanations()


if __name__ == "__main__":
    main()
