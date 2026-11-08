# XAI insurance risk prediction

Predict Prudential underwriting class `Response` (1–8) with logistic regression, a random forest, and XGBoost, then explain the fits with SHAP and LIME. The Streamlit app reads the saved files. It does not retrain.

Evaluation is a stratified 80/20 hold-out from labelled `data/train.csv`. `data/test.csv` is not scored.

Python 3.12. Package versions are pinned in `requirements.txt`. Tree models fit on one thread so a second run with this environment stays on the same hyperparameters and the same trees. The search still spreads candidates across cores.

## Setup

From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe run.py
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

`run.py` writes the split, tunes the three models (`n_iter=8`, 3-fold stratified search on macro F1), then writes the SHAP and LIME figures. `--quick` fits the default hyperparameters and skips the search. `--n-iter` and `--cv-folds` change the search.

Each stage can also be run on its own, in order: `prep.py`, `models.py`, `explain.py`.

Notebooks live in `notebooks/`. Start Jupyter from this folder so the import path resolves.

## Layout

| Path | Role |
| --- | --- |
| `data/train.csv` | Labelled applications |
| `prep.py` | Drop four sparse medical columns, add missingness indicators and three engineered features, stratified split |
| `models.py` | Three pipelines, search, hold-out metrics |
| `explain.py` | Six hold-out cases, SHAP, LIME |
| `dashboard/app.py` | Overview, metrics, score one hold-out row, saved figures |
| `artefacts/` | Split, fitted pipelines, figures |

`best_model.txt` is whichever model won macro F1, then OvR AUC. The local explanation figures are XGBoost either way. The risk band on the score page follows the predicted class, not whether P(high risk) is above 0.5.

Imbalance is handled differently in each model: `class_weight="balanced"` for logistic regression, `balanced_subsample` for the forest, and balanced sample weights inside the XGBoost fit. XGBoost trains on labels 0–7 and returns Response 1–8.
