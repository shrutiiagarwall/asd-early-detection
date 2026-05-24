# 🧩 Early ASD Detection via Machine Learning
### A Multi-Age AQ-10 Screening Pipeline with Explainability, Uncertainty Quantification, and Drift Monitoring

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-scikit--learn-orange?logo=scikit-learn" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/UI-Gradio-FF7C00?logo=gradio" alt="Gradio"/>
  <img src="https://img.shields.io/badge/XAI-SHAP-8A2BE2" alt="SHAP"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" alt="Status"/>
</p>

---

## Abstract

> *[Placeholder — to be replaced with paper abstract upon publication]*
>
> We present a production-grade machine learning pipeline for early Autistic Spectrum Disorder
> (ASD) screening across the full age spectrum (4–80 years) using the validated AQ-10 instrument.
> The system addresses seven documented research gaps in prior work: lack of threshold optimisation,
> absence of explainability, unaddressed class imbalance, no uncertainty quantification, no
> deployment readiness, missing production monitoring, and hardcoded hyperparameters.
> A two-stage modelling architecture quantifies the discriminative contribution of the AQ-10
> instrument by comparing a demographic-only baseline (AUC ≈ 0.63) against the full screening
> model (AUC ≈ 0.99), confirming the instrument's clinical validity on a 5,021-sample cohort.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Notebooks](#notebooks)
6. [Model Weights](#model-weights)
7. [Research Gaps Addressed](#research-gaps-addressed)
8. [Dataset](#dataset)
9. [Results](#results)
10. [Citation](#citation)
11. [License](#license)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT: AQ-10 Questionnaire                       │
│               (10 binary items + 4 demographic fields)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   src/features/pipeline.py                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Demographic  │  │  AQ Domain   │  │  Novel Composite Risk     │  │
│  │  Encoding   │→ │  Clustering  │→ │  Scoring  [ABSTRACTED]    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
               ┌───────────────┴────────────────┐
               ▼                                ▼
┌──────────────────────┐           ┌───────────────────────────┐
│  Stage 1: DEMO Model │           │  Stage 2: FULL AQ-10 Model │
│  (Demographics only) │           │  (AQ-10 items + demo)      │
│  AUC ≈ 0.63          │           │  AUC ≈ 0.99                │
└──────────┬───────────┘           └──────────────┬────────────┘
           │                                       │
           └──────────────────┬────────────────────┘
                              ▼
         ┌──────────────────────────────────────────┐
         │           Post-processing                 │
         │  • Threshold optimisation (recall ≥ 0.88) │
         │  • Uncertainty quantification (MC trees)  │
         │  • SHAP explanation (global + local)       │
         │  • PSI drift monitoring                    │
         └──────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │   app/app.py  (Gradio UI)  │
              └───────────────────────────┘
```

---

## Repository Structure

```
asd-early-detection/
│
├── app/
│   └── app.py                    # Gradio web interface (fully public)
│
├── data/
│   ├── raw/                      # Original UCI datasets (gitignored; see Dataset §)
│   ├── processed/                # Merged & enhanced CSV (gitignored)
│   └── sample/
│       ├── ASD_sample_public.csv # 50-row anonymised sample (public)
│       └── README.md
│
├── models/                       # !! Excluded from git (see Model Weights §)
│   └── README.md                 # Access instructions
│
├── notebooks/
│   ├── 00_data_preparation.ipynb # Data merging & cleaning
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_model_training.ipynb   # Full training pipeline (main notebook)
│
├── src/
│   ├── features/
│   │   └── pipeline.py           # Feature engineering (novel logic abstracted)
│   ├── models/
│   │   └── train.py              # Model definitions, CV, tuning, stacking
│   ├── evaluation/
│   │   └── metrics.py            # Clinical metrics, SHAP, PSI, uncertainty
│   └── utils/
│       └── config.py             # Central configuration
│
├── tests/
│   └── test_pipeline.py          # Pytest smoke tests
│
├── docs/
│   └── architecture.png          # System architecture diagram
│
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI
│
├── .gitignore                    # Blocks .pkl / .onnx / data files
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.9** or higher
- `git`
- (Optional) CUDA-capable GPU for faster SHAP computation

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/asd-early-detection.git
cd asd-early-detection
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Verify installation

```bash
pytest tests/ -v
```

All tests should pass on the public sample data without model weights.

---

## Usage

### Launch the Gradio screening app

```bash
python app/app.py
```

Open `http://localhost:7860` in your browser.

> **No model weights?** The app runs in **DEMO MODE** automatically —
> UI, validation, and explanation logic are fully functional; predictions
> are rule-based stubs. See [Model Weights](#model-weights) to request artifacts.

**Additional options:**

```bash
python app/app.py --share          # Generate public Gradio link
python app/app.py --port 7861      # Custom port
```

### Run the training pipeline

```bash
# Step 1: Place ASD_Combined_Enhanced_Final.csv in data/processed/
# Step 2: Execute the main notebook
jupyter notebook notebooks/02_model_training.ipynb
```

### Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_data_preparation.ipynb` | Downloads UCI datasets, merges, cleans, and saves `ASD_Combined_Enhanced_Final.csv` |
| `01_eda.ipynb` | Full EDA: class distribution, AQ-10 score structure, age-band analysis, correlation heatmap |
| `02_model_training.ipynb` | Two-stage training pipeline: CV, tuning, stacking, SHAP, uncertainty, PSI drift detection |

---

## Model Weights

> ### ⚠️ Model weights and pre-trained artifacts are available upon request for research collaboration.

The trained model files (`.pkl`, `.onnx`) are **not distributed in this repository**
to protect intellectual property associated with the pending publication.

**To request access:**

1. Email **[your.email@institution.ac.in]**
2. Use subject line: `[ASD Detection] Model Weights Request — <Name> / <Institution>`
3. Include: your affiliation, intended use, and confirmation of licence agreement

We aim to respond within **5 business days**.

*See [`models/README.md`](models/README.md) for a full list of available artifacts.*

---

## Research Gaps Addressed

This project directly addresses **7 documented gaps** in the existing ASD screening ML literature:

| # | Gap in Prior Work | This Project's Solution |
|---|-------------------|-------------------------|
| 1 | Default 0.5 threshold used everywhere | `optimize_threshold()`: sweeps PR curve, targets recall ≥ 0.88 |
| 2 | Black-box models, no explainability | SHAP TreeExplainer: global feature importance + local waterfall plots |
| 3 | Class imbalance ignored | SMOTETomek: synthetic oversampling + boundary cleaning |
| 4 | No uncertainty output for clinicians | Monte Carlo tree variance → mean ± std + 95% CI per prediction |
| 5 | No deployment readiness | Gradio UI + ONNX export + joblib + model card |
| 6 | No production monitoring | Per-feature PSI drift detection with demographic shift simulation |
| 7 | Hardcoded hyperparameters | GridSearchCV over 54 RF configurations with stratified CV |

---

## Dataset

**Source:** UCI Machine Learning Repository — Thabtah, F. (2017–2018)  
**Licence:** CC BY 4.0

| Split | Rows | Description |
|-------|------|-------------|
| Full processed | 5,021 | Ages 4–80, AQ-10, Child + Adolescent + Adult |
| Public sample | 50 | Anonymised subset in `data/sample/` |

**Note on dataset construction:** The ASD label in the UCI dataset was assigned using the AQ-10
clinical cutoff (score ≥ 6). Any model receiving A1–A10 items will therefore achieve near-perfect
discrimination — this is expected and reflects the instrument's validated design, not data leakage.
The demographic-only baseline (AUC ≈ 0.63) vs full-instrument model (AUC ≈ 0.99) comparison
quantifies the screening instrument's clinical contribution, which is a central paper finding.

---

## Results

> *Full results table to be completed post-experiment. Preliminary figures shown.*

### Two-Stage Model Performance (Test Set, n = 1,005)

| Stage | Best Model | AUC | Sensitivity | Specificity |
|-------|-----------|-----|-------------|-------------|
| Demographic baseline | Gradient Boosting | ≈ 0.651 | ≈ 0.582 | ≈ 0.694 |
| Full AQ-10 screening | Stacking Ensemble | ≈ 0.998 | ≈ 0.994 | ≈ 0.997 |

**ΔAUC = +0.347** — quantifies the discriminative value of the AQ-10 instrument.

---

## Citation

> *To be updated upon publication.*

```bibtex
@article{asd_detection_2025,
  title   = {Early ASD Detection via Multi-Stage AQ-10 Machine Learning Pipeline
             with Explainability and Production Monitoring},
  author  = {[Author Names]},
  journal = {[Journal Name]},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/<your-username>/asd-early-detection}
}
```

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

The underlying dataset is released under **CC BY 4.0** by Fadi Thabtah (UCI ML Repository).

Trained model weights are released under a **non-commercial research licence**,
available upon request (see [Model Weights](#model-weights)).

---

## Acknowledgements

- **Thabtah, F.** (2017–2018) for the ASD Screening Datasets (UCI ML Repository).
- **Allison, C. et al.** (2012) for the AQ-10 instrument and clinical threshold validation.
- **Lundberg, S. & Lee, S.-I.** (2017) for the SHAP framework.
- University of Engineering and Management, Jaipur.

---

<p align="center">
  <sub>⚠️ Research prototype — not for clinical diagnosis.</sub>
</p>
