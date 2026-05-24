"""
src/features/pipeline.py
------------------------
Public feature engineering pipeline.

Architecture note (IP protection)
----------------------------------
The high-level feature *categories* (demographic encoding, AQ domain clustering,
biological risk aggregation, interaction terms) are fully visible here.

The exact mathematical formulation of the **novel composite risk scoring**
(domain weighting coefficients, nonlinear interaction terms, calibration
offsets) is implemented in ``_novel_risk_scoring()``, which is intentionally
abstracted.  The pipeline is fully reproducible using the pre-computed
``Composite_Risk`` column that ships with the public sample dataset.

For research collaboration or replication, contact the authors —
see README.md § "Model Weights & Proprietary Features".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    AQ10_ITEMS,
    AGE_BAND_MAP,
    AGE_BAND_BINS,
    AGE_BAND_LABELS,
    AQ10_CLINICAL_THRESHOLD,
)


# ── Public helper: demographic encoding ──────────────────────────────────────

def encode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode raw demographic columns to numeric.

    Maps
    ----
    Sex          : {'m','male'}→1, {'f','female'}→0
    Jaundice     : 'yes'→1, 'no'→0
    Family_ASD   : 'yes'→1, 'no'→0
    Age_Band     : ordinal encoding via AGE_BAND_MAP
    """
    df = df.copy()

    if df["Sex"].dtype == object:
        df["Sex"] = df["Sex"].str.lower().map(
            {"m": 1, "male": 1, "f": 0, "female": 0}
        )
    for col in ["Jaundice", "Family_ASD"]:
        if df[col].dtype == object:
            df[col] = df[col].str.lower().map({"yes": 1, "no": 0})

    df["Age_Band_Enc"] = df["Age_Band"].map(AGE_BAND_MAP)
    return df


# ── Public helper: AQ-10 domain aggregation ───────────────────────────────────

def compute_aq_domains(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AQ-10 domain sub-scores.

    Domain groupings follow the published AQ-10 validation literature
    (Allison et al., 2012; Booth et al., 2006).

    Domains
    -------
    Social_Score      : items capturing social awareness / empathy (A1, A7, A8, A10)
    Attention_Score   : attention switching items (A2, A4)
    Comm_Score        : communication items (A5, A6)
    Imagination_Score : imagination / flexibility items (A3, A9)
    Total_Score       : sum of all 10 items
    Above_Threshold   : 1 if Total_Score >= clinical cutoff (default 6)
    """
    df = df.copy()
    df["Total_Score"]       = df[AQ10_ITEMS].sum(axis=1)
    df["Above_Threshold"]   = (df["Total_Score"] >= AQ10_CLINICAL_THRESHOLD).astype(int)
    df["Social_Score"]      = df[["A1", "A7", "A8", "A10"]].sum(axis=1)
    df["Attention_Score"]   = df[["A2", "A4"]].sum(axis=1)
    df["Comm_Score"]        = df[["A5", "A6"]].sum(axis=1)
    df["Imagination_Score"] = df[["A3", "A9"]].sum(axis=1)
    return df


# ── Novel composite risk scoring — ABSTRACTED ─────────────────────────────────

def _novel_risk_scoring(df: pd.DataFrame) -> pd.Series:
    """
    **Proprietary composite risk score.**

    This function is intentionally left as a stub in the public repository.
    The pre-computed ``Composite_Risk`` column is included in the distributed
    dataset to allow full pipeline reproducibility without exposing the
    underlying formulation.

    The method involves:
      - Domain-weighted linear aggregation (weights derived from a held-out
        clinical validation cohort — not disclosed publicly)
      - A non-linear biological risk amplifier
      - An age-band calibration offset

    For research collaboration, replication, or licensing enquiries,
    please contact the authors (see README.md).

    Returns
    -------
    pd.Series
        Composite risk score for each sample.  If ``Composite_Risk`` already
        exists in *df* (as in the distributed dataset), it is returned
        directly without recomputation.
    """
    if "Composite_Risk" in df.columns:
        return df["Composite_Risk"]

    # ── Stub: falls back to a simple unweighted sum so the pipeline
    # ── remains runnable on new data without the proprietary weights.
    # ── Results will differ from the published system.
    stub_score = (
        df["Social_Score"] * 1.0
        + df["Attention_Score"] * 1.0
        + df["Comm_Score"] * 1.0
        + df["Imagination_Score"] * 1.0
    )
    return stub_score


# ── Public: interaction features ─────────────────────────────────────────────

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute non-leaking interaction features from demographic and
    biological risk columns only.  No AQ item scores are used here to
    avoid target leakage through score aggregation.
    """
    df = df.copy()
    df["Bio_Risk"]          = df["Jaundice"] + df["Family_ASD"]
    df["Bio_Combined"]      = df["Jaundice"] * df["Family_ASD"]
    df["FamASD_x_Threshold"] = df["Family_ASD"] * df["Above_Threshold"]
    df["Risk_x_AgeBand"]    = df["Bio_Risk"] * df["Age_Band_Enc"]
    return df


# ── Main public entry point ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps
    -----
    1. Demographic encoding
    2. AQ-10 domain aggregation (public)
    3. Novel composite risk scoring (pre-computed column used if available)
    4. Interaction feature construction

    Parameters
    ----------
    df : pd.DataFrame
        Raw or minimally pre-processed dataframe containing at minimum:
        A1–A10, Age, Sex, Jaundice, Family_ASD, Age_Band.

    Returns
    -------
    pd.DataFrame
        Feature-enriched dataframe ready for ``build_feature_matrix()``.
    """
    df = encode_demographics(df)
    df = compute_aq_domains(df)
    df["Composite_Risk"] = _novel_risk_scoring(df)
    df = compute_interaction_features(df)

    # Score severity tier (ordinal)
    df["Score_Tier_Enc"] = pd.cut(
        df["Total_Score"],
        bins=[-1, 2, 5, 7, 10],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return df


# ── Preprocessor construction ─────────────────────────────────────────────────

# Feature sets are split into two published model stages (see paper §3.2):
#   DEMO  — demographic pre-screening (no AQ items)
#   FULL  — complete AQ-10 screening model

DEMO_FEATURES = [
    "Age", "Sex", "Jaundice", "Family_ASD",
    "Bio_Risk", "Bio_Combined", "Age_Band_Enc",
]

FULL_FEATURES = AQ10_ITEMS + [
    "Age", "Sex", "Jaundice", "Family_ASD",
    "Bio_Risk", "Bio_Combined", "Age_Band_Enc",
]

# NOTE: Composite_Risk, Total_Score, sub-scores, Above_Threshold, Score_Tier_Enc,
# FamASD_x_Threshold are deliberately excluded from model inputs —
# they encode the target signal and would cause evaluation inflation.


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "full",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract feature matrix and target vector from an engineered dataframe.

    Parameters
    ----------
    df          : Feature-engineered dataframe (output of engineer_features).
    feature_set : ``"demo"`` for demographic-only or ``"full"`` for AQ-10 model.

    Returns
    -------
    X           : Feature matrix (numpy array)
    y           : Target vector (numpy array, int)
    feat_names  : List of feature names in column order
    """
    cols = DEMO_FEATURES if feature_set == "demo" else FULL_FEATURES
    cols = [c for c in cols if c in df.columns]

    X = df[cols].values.astype(float)
    y = df["Class"].values.astype(int) if "Class" in df.columns else None

    return X, y, cols


def get_preprocessor(feature_set: str = "full") -> ColumnTransformer:
    """
    Return a fitted-ready ColumnTransformer for the chosen feature set.

    Numeric columns (Age, Bio_Risk, Risk_x_AgeBand) are StandardScaled.
    Binary / ordinal columns are passed through unchanged.
    """
    cols = DEMO_FEATURES if feature_set == "demo" else FULL_FEATURES

    numeric  = ["Age", "Bio_Risk", "Risk_x_AgeBand"]
    numeric  = [c for c in numeric if c in cols]
    passthru = [c for c in cols if c not in numeric]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough",   passthru),
        ],
        remainder="drop",
    )
