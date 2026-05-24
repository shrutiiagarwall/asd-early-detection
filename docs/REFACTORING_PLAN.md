# Refactoring Plan: Monolithic Notebook → Production Repository

This document explains exactly how `ASD_Detection_v4.ipynb` maps to the
repository file structure and how the IP-protection abstraction works.

---

## 1. Cell-to-file mapping

| Notebook cells | → Repository file | Notes |
|----------------|-------------------|-------|
| Install cell | `requirements.txt` | Pinned versions |
| Imports | `src/utils/config.py` + each module | Constants extracted |
| `load_and_prepare()` | `src/features/pipeline.py :: engineer_features()` | Path uses config |
| `encode_demographics()` | `src/features/pipeline.py :: encode_demographics()` | Public |
| `compute_aq_domains()` | `src/features/pipeline.py :: compute_aq_domains()` | Public |
| Novel composite scoring | `src/features/pipeline.py :: _novel_risk_scoring()` | **ABSTRACTED** (stub + pre-computed col) |
| `build_preprocessor()` | `src/features/pipeline.py :: get_preprocessor()` | Public |
| `build_feature_matrix()` | `src/features/pipeline.py :: build_feature_matrix()` | Public |
| `prepare_split()` | `notebooks/02_model_training.ipynb` (inline) | Split logic stays in notebook |
| `get_models()` | `src/models/train.py :: get_models()` | Public |
| `train_with_cv()` | `src/models/train.py :: train_with_cv()` | Public |
| `tune_random_forest()` | `src/models/train.py :: tune_random_forest()` | Public |
| `build_stacking_ensemble()` | `src/models/train.py :: build_stacking_ensemble()` | Public |
| `optimize_threshold()` | `src/models/train.py :: optimize_threshold()` | Public |
| `evaluate_model()` | `src/evaluation/metrics.py :: evaluate_model()` | Public |
| `monte_carlo_uncertainty()` | `src/evaluation/metrics.py :: monte_carlo_uncertainty()` | Public |
| `compute_psi()` / `monitor_drift()` | `src/evaluation/metrics.py` | Public |
| `run_shap_analysis()` | `src/evaluation/metrics.py :: run_shap_analysis()` | Public |
| `export_model()` | inline in notebook | Saves to `models/` (gitignored) |
| `build_gradio_app()` | `app/app.py` | Rewritten as production app |
| EDA plots | `notebooks/01_eda.ipynb` | Separated for clarity |
| Summary cell | `notebooks/02_model_training.ipynb` footer | |

---

## 2. IP protection: novel feature engineering

### The problem

The `_novel_risk_scoring()` function contains domain-weighted linear
aggregation coefficients derived from a held-out clinical validation cohort.
These coefficients represent the primary novel contribution of the paper.

### The solution: three-layer abstraction

**Layer 1 — Stub function (public)**
```python
def _novel_risk_scoring(df):
    """Proprietary. Pre-computed column used if available."""
    if "Composite_Risk" in df.columns:
        return df["Composite_Risk"]   # ← uses distributed pre-computed values
    # fallback: unweighted sum (different from published system)
    return df["Social_Score"] + df["Attention_Score"] + ...
```

**Layer 2 — Pre-computed column in distributed data**
The `Composite_Risk` column ships in `ASD_sample_public.csv` and in the
full dataset given to collaborators. The pipeline runs correctly without
the formula being visible.

**Layer 3 — README disclosure**
The `pipeline.py` docstring explains *what* the feature does (domain-weighted
composite with biological risk amplifier and age calibration) but not *how*
(the exact coefficients). This is standard practice in academic software.

### What a recruiter / reviewer sees

- The pipeline is complete and runs end-to-end ✓
- The abstraction point is clearly documented ✓
- The stub fallback produces reasonable (if suboptimal) results ✓
- The proprietary detail is protected ✓

---

## 3. Splitting the single notebook into three

```
ASD_Detection_v4.ipynb
       │
       ├── Cells 0-9  (load + EDA)          → notebooks/01_eda.ipynb
       │
       ├── Cells 10-13 (feature eng + split) → imports from src/features/
       │                                       (cells kept thin in notebook)
       │
       └── Cells 14-38 (training + eval)     → notebooks/02_model_training.ipynb
                                                imports from src/models/
                                                imports from src/evaluation/
```

The notebooks become **orchestrators** — they call `src/` functions rather
than defining them inline.  This means:

- Researchers can read the full logic in `src/`
- The notebooks stay clean and readable
- The `_novel_risk_scoring` stub is the only hidden piece

---

## 4. Running the abstracted pipeline end-to-end

```python
# In 02_model_training.ipynb — this is all the "glue" needed:

from src.features.pipeline import engineer_features, build_feature_matrix, get_preprocessor
from src.models.train import get_models, train_with_cv, tune_random_forest, build_stacking_ensemble, optimize_threshold
from src.evaluation.metrics import evaluate_model, monte_carlo_uncertainty, monitor_drift, run_shap_analysis

df = pd.read_csv("data/processed/ASD_Combined_Enhanced_Final.csv")
df = engineer_features(df)          # Composite_Risk pre-computed, stub not needed
X, y, feat_names = build_feature_matrix(df, feature_set="full")
# ... rest of pipeline unchanged
```
