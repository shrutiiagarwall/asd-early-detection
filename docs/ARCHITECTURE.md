# System Architecture

## Overview

The pipeline is designed around three principles:

1. **Separation of concerns** — data, features, models, evaluation, and UI are fully decoupled
2. **IP protection** — novel feature engineering is abstracted without breaking the public pipeline
3. **Deployment readiness** — the Gradio app degrades gracefully when weights are absent

---

## Two-Stage Modelling Architecture

The core research contribution is a **two-stage comparison**:

```
Stage 1 — Demographic Baseline
  Input  : Age, Sex, Jaundice, Family_ASD, Bio_Risk, Age_Band_Enc
  Purpose: Establish pre-questionnaire ASD risk
  AUC    : ≈ 0.63 (honest — demographics are weak ASD predictors alone)

Stage 2 — Full AQ-10 Screening Model
  Input  : A1–A10 + all Stage 1 features
  Purpose: Primary screening prediction
  AUC    : ≈ 0.99 (expected — AQ-10 was designed to predict this label)

ΔAUC = +0.34 → quantifies clinical value of the AQ-10 instrument
```

This framing resolves the "1.0 AUC problem" that appears in prior work: the
UCI ASD dataset assigns labels using the AQ-10 cutoff, so near-perfect
discrimination is a **property of the dataset**, not a leakage artefact.
Reporting both stages makes this transparent and publishable.

---

## Data Flow

```
data/raw/*.arff               (UCI originals, not in git)
       │
       ▼
notebooks/00_data_preparation.ipynb
       │  merge, clean, age-band assignment
       ▼
data/processed/ASD_Combined_Enhanced_Final.csv   (not in git)
data/sample/ASD_sample_public.csv                (50 rows, in git)
       │
       ▼
src/features/pipeline.py :: engineer_features()
       │  encode_demographics()
       │  compute_aq_domains()
       │  _novel_risk_scoring()   ← ABSTRACTED
       │  compute_interaction_features()
       ▼
build_feature_matrix()
       │  DEMO_FEATURES  (7 cols)
       │  FULL_FEATURES  (17 cols)
       ▼
StandardScaler + SMOTETomek
       │
       ▼
src/models/train.py
       │  get_models()            LR | RF | GB | SVM
       │  train_with_cv()         5-fold StratifiedKFold
       │  tune_random_forest()    GridSearchCV (54 configs)
       │  build_stacking_ensemble()  RF + GB + SVM → LR meta
       │  optimize_threshold()    recall ≥ 0.88 sweep
       ▼
src/evaluation/metrics.py
       │  evaluate_model()        accuracy, AUC, PR-AUC, clinical 4-metric
       │  monte_carlo_uncertainty()  tree variance → CI
       │  run_shap_analysis()     TreeExplainer global + local
       │  monitor_drift()         PSI per feature
       ▼
models/  (gitignored)
       │  asd_full_aq10_model.pkl
       │  asd_demographic_model.pkl
       │  model_card.json
       ▼
app/app.py  (Gradio)
       │  loads weights if present, else DEMO MODE
       │  shows both stage predictions + uncertainty
       ▼
http://localhost:7860
```

---

## IP Protection Design

```
PUBLIC (fully visible)                 PROTECTED (abstracted)
───────────────────────────────────    ──────────────────────────────────
encode_demographics()                  _novel_risk_scoring()
compute_aq_domains()                      ↑
compute_interaction_features()         coefficient values, nonlinear terms,
build_feature_matrix()                 calibration offsets — NOT published
get_preprocessor()
get_models()
train_with_cv()
tune_random_forest()
build_stacking_ensemble()
optimize_threshold()
evaluate_model()
monte_carlo_uncertainty()
run_shap_analysis()
monitor_drift()
app/app.py
```

The `_novel_risk_scoring()` stub returns the pre-computed `Composite_Risk`
column when it is present in the dataframe (as in all distributed datasets),
so the pipeline runs correctly without the formula being visible.

---

## Research Gaps Addressed

See README.md § "Research Gaps Addressed" for the full table.
This architecture was specifically designed to address 7 gaps in prior ASD
screening ML literature identified through a systematic review of papers using
the same UCI datasets (2017–2024).
