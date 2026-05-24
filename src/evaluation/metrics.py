"""
src/evaluation/metrics.py
-------------------------
Evaluation helpers: clinical metrics, PSI drift detection,
uncertainty quantification, and SHAP explainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.utils.config import UNCERTAINTY_STD_THRESHOLD


# ── Full evaluation report ────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    y_pred_opt: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute accuracy, ROC-AUC, PR-AUC, and four clinical metrics.

    Clinical metrics
    ----------------
    Sensitivity (Recall) — fraction of true ASD cases correctly detected
    Specificity          — fraction of true non-ASD cases correctly cleared
    PPV (Precision)      — positive predictive value
    NPV                  — negative predictive value
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = y_pred_opt if y_pred_opt is not None else (y_prob >= threshold).astype(int)

    cm           = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity  = tp / (tp + fn + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)
    ppv          = tp / (tp + fp + 1e-8)
    npv          = tn / (tn + fn + 1e-8)
    f1           = 2 * ppv * sensitivity / (ppv + sensitivity + 1e-8)

    print(f"\n{'='*58}\n  {model_name}\n{'='*58}")
    print(f"  Threshold    : {threshold:.3f}")
    print(f"  Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC      : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC       : {average_precision_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=["No ASD", "ASD Traits"]))
    print(f"  Sensitivity  : {sensitivity:.4f}    Specificity : {specificity:.4f}")
    print(f"  PPV          : {ppv:.4f}    NPV         : {npv:.4f}")

    return {
        "accuracy"   : accuracy_score(y_test, y_pred),
        "roc_auc"    : roc_auc_score(y_test, y_prob),
        "pr_auc"     : average_precision_score(y_test, y_prob),
        "f1"         : f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv"        : ppv,
        "npv"        : npv,
        "threshold"  : threshold,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "cm"         : cm,
    }


# ── Uncertainty quantification ────────────────────────────────────────────────

def monte_carlo_uncertainty(
    model,
    X_test: np.ndarray,
) -> pd.DataFrame:
    """
    Estimate epistemic uncertainty via Random Forest tree variance.

    Returns a DataFrame with mean probability, std, 95% CI,
    and a flag for samples exceeding UNCERTAINTY_STD_THRESHOLD.
    """
    if not hasattr(model, "estimators_"):
        raise TypeError("monte_carlo_uncertainty requires a tree-ensemble model.")

    tree_preds = np.array([t.predict_proba(X_test)[:, 1] for t in model.estimators_])
    mean_p = tree_preds.mean(axis=0)
    std_p  = tree_preds.std(axis=0)
    ci_lo  = np.percentile(tree_preds, 2.5,  axis=0)
    ci_hi  = np.percentile(tree_preds, 97.5, axis=0)

    udf = pd.DataFrame({
        "mean_probability": mean_p,
        "uncertainty_std" : std_p,
        "CI_lower_95"     : ci_lo,
        "CI_upper_95"     : ci_hi,
        "uncertain_flag"  : std_p > UNCERTAINTY_STD_THRESHOLD,
    })
    flagged = udf["uncertain_flag"].sum()
    print(
        f"[Uncertainty] Flagged for review (std>{UNCERTAINTY_STD_THRESHOLD}): "
        f"{flagged}/{len(udf)} ({flagged/len(udf)*100:.1f}%)"
    )
    return udf


# ── PSI drift detection ───────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between a reference distribution and a new batch.

    PSI < 0.10  → Stable       (no action)
    PSI < 0.20  → Monitor      (flag for review)
    PSI >= 0.20 → Significant  (retrain recommended)
    """
    bp = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    bp[0] -= 1e-8
    bp[-1] += 1e-8
    if len(np.unique(np.round(bp, 8))) < 2:
        return 0.0

    exp_pct = np.histogram(expected, bins=bp)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=bp)[0] / len(actual)
    exp_pct = np.clip(exp_pct, 1e-4, None)
    act_pct = np.clip(act_pct, 1e-4, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def monitor_drift(
    X_train: np.ndarray,
    X_new: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute per-feature PSI and return a sorted drift report DataFrame."""
    results = {}
    for i, feat in enumerate(feature_names):
        psi    = compute_psi(X_train[:, i], X_new[:, i])
        status = "✅ STABLE" if psi < 0.1 else ("⚠️ MONITOR" if psi < 0.2 else "🔴 RETRAIN")
        clean  = feat.replace("num__", "").replace("bin__", "")
        results[clean] = {"PSI": psi, "Status": status}

    drift_df = pd.DataFrame(results).T.sort_values("PSI", ascending=False)
    drift_df["PSI"] = drift_df["PSI"].astype(float)
    return drift_df


# ── SHAP explainability ───────────────────────────────────────────────────────

def run_shap_analysis(model, X_train, X_test, feature_names, label=""):
    """
    Compute SHAP values using TreeExplainer (tree models) or
    KernelExplainer fallback (any model).

    Returns the shap_values array for the positive class.
    Plots are generated inside the notebook (see notebooks/02_model_training.ipynb).
    """
    try:
        import shap  # optional dependency — already in requirements.txt
    except ImportError:
        print("[SHAP] shap not installed. Run: pip install shap")
        return None

    print(f"[SHAP] Computing for {label}...")
    clean_names = [f.replace("num__", "").replace("bin__", "") for f in feature_names]

    if hasattr(model, "estimators_") and hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        sv_raw    = explainer.shap_values(X_test)
        if isinstance(sv_raw, list):
            sv = sv_raw[1]
            ev = explainer.expected_value[1]
        elif sv_raw.ndim == 3:
            sv = sv_raw[:, :, 1]
            ev = (explainer.expected_value[1]
                  if hasattr(explainer.expected_value, "__len__")
                  else explainer.expected_value)
        else:
            sv = sv_raw
            ev = explainer.expected_value
    else:
        background = shap.kmeans(X_train, 50)
        explainer  = shap.KernelExplainer(model.predict_proba, background)
        sv_raw     = explainer.shap_values(X_test[:200])
        sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        ev = (explainer.expected_value[1]
              if hasattr(explainer.expected_value, "__len__")
              else explainer.expected_value)

    return sv, float(ev), clean_names
