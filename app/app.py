"""
app/app.py
----------
ASD Early Detection — Gradio Screening Interface
================================================
Instrument : AQ-10 (Allison et al. 2012) | Ages 4–80
Model      : Stacking Ensemble (RF + GB + SVM) — Full AQ-10 Pipeline

Usage
-----
    python app/app.py              # local launch
    python app/app.py --share      # public Gradio link
    python app/app.py --port 7861  # custom port

Deployment note
---------------
If model weights are absent (see README — "Model Weights" section),
the app runs in DEMO MODE with a clearly labelled stub predictor.
The UI, validation, and explanation logic remain fully functional.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import (
    AQ10_QUESTIONS,
    AQ10_ITEMS,
    AQ10_CLINICAL_THRESHOLD,
    MODELS_DIR,
    UNCERTAINTY_STD_THRESHOLD,
)
from src.features.pipeline import DEMO_FEATURES, FULL_FEATURES, engineer_features

# ── Gradio ────────────────────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print("[ERROR] Gradio not installed. Run: pip install gradio")
    sys.exit(1)


# ── Model loader — graceful degradation ──────────────────────────────────────

def _load_artifacts() -> tuple:
    """
    Attempt to load trained model and scaler from MODELS_DIR.
    Returns (model_full, scaler_full, model_demo, scaler_demo, demo_mode).

    If artifacts are absent, returns (None, None, None, None, True)
    and the app renders in DEMO MODE with a stub predictor.
    """
    import joblib

    full_model_path  = MODELS_DIR / "asd_full_aq10_model.pkl"
    full_scaler_path = MODELS_DIR / "asd_full_aq10_scaler.pkl"
    demo_model_path  = MODELS_DIR / "asd_demographic_model.pkl"
    demo_scaler_path = MODELS_DIR / "asd_demographic_scaler.pkl"

    required = [full_model_path, full_scaler_path, demo_model_path, demo_scaler_path]
    if not all(p.exists() for p in required):
        missing = [p.name for p in required if not p.exists()]
        print(
            f"[INFO] Model weights not found ({missing}).\n"
            "       Running in DEMO MODE — predictions are illustrative only.\n"
            "       See README.md § 'Model Weights' to request trained artifacts."
        )
        return None, None, None, None, True

    return (
        joblib.load(full_model_path),
        joblib.load(full_scaler_path),
        joblib.load(demo_model_path),
        joblib.load(demo_scaler_path),
        False,
    )


model_full, scaler_full, model_demo, scaler_demo, DEMO_MODE = _load_artifacts()


# ── Stub predictor for demo mode ──────────────────────────────────────────────

def _stub_predict(aq_score: int, age: int, bio_risk: int) -> tuple[float, float]:
    """
    Rule-based stub that mimics plausible ASD screening probabilities
    without any trained model.  Used only when weights are absent.
    Logic mirrors the AQ-10 clinical heuristic, not the trained model.
    """
    base      = aq_score / 10
    demo_base = min(0.35 + bio_risk * 0.08, 0.65)
    noise     = np.random.default_rng(42).uniform(-0.02, 0.02)
    return float(np.clip(base + noise, 0.01, 0.99)), float(np.clip(demo_base + noise, 0.01, 0.99))


# ── Core prediction logic ─────────────────────────────────────────────────────

def predict(
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
    age, sex, jaundice, family_asd,
):
    """Called by Gradio on every button click."""

    # ── Build raw feature row ────────────────────────────────────────────────
    sex_e  = 1 if sex == "Male"  else 0
    jau_e  = 1 if jaundice == "Yes" else 0
    fam_e  = 1 if family_asd == "Yes" else 0
    band_e = (0 if age <= 11 else (1 if age <= 17 else (2 if age <= 40 else 3)))
    bio    = jau_e + fam_e
    total  = a1+a2+a3+a4+a5+a6+a7+a8+a9+a10

    above_thresh = int(total >= AQ10_CLINICAL_THRESHOLD)

    if DEMO_MODE:
        full_p, demo_p = _stub_predict(total, age, bio)
        full_std = 0.0
        mode_banner = (
            "⚠️  **DEMO MODE** — Model weights not loaded.  "
            "Predictions are illustrative only (rule-based stub).\n"
            "See README § *Model Weights* to request trained artifacts.\n\n"
        )
    else:
        # ── Demographic model ────────────────────────────────────────────────
        demo_row = np.array([[age, sex_e, jau_e, fam_e, bio, jau_e*fam_e, band_e]])
        demo_sc  = scaler_demo.transform(demo_row)
        demo_p   = float(model_demo.predict_proba(demo_sc)[0][1])

        # ── Full AQ-10 model ─────────────────────────────────────────────────
        full_row = np.array([[
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
            age, sex_e, jau_e, fam_e, bio, jau_e*fam_e, band_e,
        ]])
        full_sc  = scaler_full.transform(full_row)
        full_p   = float(model_full.predict_proba(full_sc)[0][1])
        full_std = (
            float(np.std([t.predict_proba(full_sc)[0][1] for t in model_full.estimators_]))
            if hasattr(model_full, "estimators_") else 0.0
        )
        mode_banner = ""

    # ── Build output ─────────────────────────────────────────────────────────
    risk_icon  = "🔴" if full_p >= 0.5 else "🟢"
    risk_label = "HIGH RISK — Recommend clinical evaluation" if full_p >= 0.5 else "LOW RISK — Routine monitoring advised"
    aq_note    = f"Above AQ-10 threshold (≥{AQ10_CLINICAL_THRESHOLD})" if total >= AQ10_CLINICAL_THRESHOLD else f"Below AQ-10 threshold (<{AQ10_CLINICAL_THRESHOLD})"
    conf_note  = (
        "⚠️  High uncertainty — clinical referral recommended regardless of score"
        if full_std > UNCERTAINTY_STD_THRESHOLD else "✓  Confident prediction"
    )

    result = (
        f"{mode_banner}"
        f"**{risk_icon}  {risk_label}**\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Full AQ-10 model probability | **{full_p:.1%}** (±{full_std:.1%}) |\n"
        f"| Demographic baseline probability | {demo_p:.1%} |\n"
        f"| AQ-10 Score | {total}/10 — {aq_note} |\n"
        f"| Model confidence | {conf_note} |\n\n"
        f"---\n"
        f"⚠️  **Disclaimer:** This tool is a research-grade screening aid only.  "
        f"It does **not** constitute a clinical diagnosis.  "
        f"A formal ASD assessment must be conducted by a licensed clinician."
    )
    return result


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    demo_banner_md = (
        "> **DEMO MODE** — Model weights are not loaded. "
        "Screening results are illustrative only.\n"
        "> To request trained model artifacts, see the project README."
        if DEMO_MODE else ""
    )

    with gr.Blocks(
        title="ASD Early Detection — AQ-10 Screening Tool",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css="""
        .disclaimer { font-size: 0.82rem; color: #888; }
        .result-box textarea { font-size: 1rem !important; }
        footer { visibility: hidden; }
        """,
    ) as app:

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🧩 ASD Early Detection Screening Tool
            **Instrument:** AQ-10 (Allison et al., 2012) &nbsp;|&nbsp;
            **Age range:** 4 – 80 years &nbsp;|&nbsp;
            **Model:** Stacking Ensemble (RF + GB + SVM)

            > This tool supports early identification of Autistic Spectrum Disorder traits
            > using the validated 10-item Autism Spectrum Quotient (AQ-10) screening instrument.
            > It is intended for **research use only** and does not replace clinical diagnosis.
            """
        )

        if DEMO_MODE:
            gr.Markdown(
                f"> ⚠️ **DEMO MODE** — Model weights are not present locally.  "
                f"Predictions are rule-based stubs for UI demonstration purposes only.  \n"
                f"> Contact the authors (see [README](../README.md)) to request trained artifacts.",
                elem_classes="disclaimer",
            )

        gr.Markdown("---")

        # ── Input columns ────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    "### AQ-10 Questionnaire\n"
                    "For each statement, select **1** (Agree / Slightly Agree) "
                    "or **0** (Disagree / Slightly Disagree)."
                )
                aq_inputs = []
                for key, question in AQ10_QUESTIONS.items():
                    aq_inputs.append(
                        gr.Slider(
                            minimum=0, maximum=1, step=1, value=0,
                            label=f"{key} — {question}",
                        )
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Personal Information")
                age_in  = gr.Slider(4, 80, value=25, step=1, label="Age (years)")
                sex_in  = gr.Radio(["Male", "Female"], label="Sex", value="Male")
                jau_in  = gr.Radio(["Yes", "No"], label="Born with Jaundice?", value="No")
                fam_in  = gr.Radio(
                    ["Yes", "No"],
                    label="Family member with ASD diagnosis?",
                    value="No",
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**AQ-10 Scoring Guide**\n\n"
                    "Score ≥ 6 → clinical referral recommended  \n"
                    "Score < 6 → below threshold  \n\n"
                    "_Source: Allison et al. (2012), PLoS ONE_",
                    elem_classes="disclaimer",
                )

        # ── Action + output ───────────────────────────────────────────────────
        gr.Markdown("---")
        run_btn = gr.Button("▶  Run Screening", variant="primary", size="lg")

        output_md = gr.Markdown(
            value="_Results will appear here after screening._",
            label="Screening Result",
        )

        gr.Markdown("---")
        with gr.Accordion("ℹ️  About this tool", open=False):
            gr.Markdown(
                """
                **Model architecture**

                This tool runs a two-stage screening pipeline:

                1. **Demographic baseline** (Age, Sex, Jaundice, Family History) —
                   provides a pre-questionnaire risk estimate (AUC ≈ 0.63).
                2. **Full AQ-10 model** (items A1–A10 + demographics) —
                   the primary screening model (AUC ≈ 0.99 on held-out test set).

                The improvement from stage 1 → stage 2 quantifies the
                discriminative value of the AQ-10 instrument itself
                (ΔAUC ≈ +0.36), which is a central finding of the associated paper.

                **References**
                - Allison C. et al. (2012). *Autism Spectrum Quotient-10 (AQ-10)*.
                  PLoS ONE 7(9).
                - Thabtah F. (2017–2018). *ASD Screening Datasets*. UCI ML Repository.

                **Explainability:** SHAP TreeExplainer is available in the
                research notebook (`notebooks/02_model_training.ipynb`).

                **Model weights:** Not distributed publicly.
                Available upon request for research collaboration — see README.
                """
            )

        # ── Wire up ───────────────────────────────────────────────────────────
        run_btn.click(
            fn=predict,
            inputs=aq_inputs + [age_in, sex_in, jau_in, fam_in],
            outputs=output_md,
        )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown(
            "<div class='disclaimer' style='text-align:center; padding-top:12px;'>"
            "ASD Early Detection Pipeline — Research Prototype | "
            "University of Engineering and Management, Jaipur | "
            "Not for clinical use"
            "</div>"
        )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Launch ASD Screening Gradio app")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",  type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--host",  type=str, default="0.0.0.0", help="Host")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app  = build_ui()
    app.launch(
        share      = args.share,
        server_port= args.port,
        server_name= args.host,
        show_error = True,
    )
