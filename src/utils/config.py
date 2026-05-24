"""
src/utils/config.py
-------------------
Central configuration for the ASD Early Detection pipeline.
All tuneable constants live here — no magic numbers in source files.
"""

from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"          # excluded from git (see .gitignore)
REPORTS_DIR= ROOT_DIR / "reports"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_FILENAME  = "ASD_Combined_Enhanced_Final.csv"
SAMPLE_FILENAME= "ASD_sample_public.csv"

# ── AQ-10 instrument metadata ─────────────────────────────────────────────────
AQ10_ITEMS = [f"A{i}" for i in range(1, 11)]

AQ10_QUESTIONS = {
    "A1":  "I often notice small sounds when others do not.",
    "A2":  "I usually concentrate more on the whole picture than the details.",
    "A3":  "I find it easy to do more than one thing at once.",
    "A4":  "If there is an interruption, I can switch back to what I was doing.",
    "A5":  "I find it easy to read between the lines when someone talks to me.",
    "A6":  "I know how to tell if someone listening is getting bored.",
    "A7":  "When reading a story, I find it difficult to work out characters' intentions.",
    "A8":  "I like to collect information about categories of things.",
    "A9":  "I find it easy to work out what someone is thinking or feeling.",
    "A10": "I find it difficult to work out people's intentions.",
}

# Scoring note: items A1, A2, A3, A4, A5, A6, A9 score 1 for "disagree/slightly disagree"
# Items A7, A8, A10 score 1 for "agree/slightly agree"
# For this dataset, A1-A10 are already binarised (0/1) in the pre-processed CSV.

AQ10_CLINICAL_THRESHOLD = 6   # Score >= 6 → refer for formal assessment (Allison et al. 2012)

DEMOGRAPHIC_COLS = ["Age", "Sex", "Jaundice", "Family_ASD"]

# ── Model split ───────────────────────────────────────────────────────────────
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
CV_FOLDS      = 5

# ── Age bands ─────────────────────────────────────────────────────────────────
AGE_BAND_MAP   = {"Child": 0, "Adolescent": 1, "Adult": 2, "Older_Adult": 3}
AGE_BAND_BINS  = [3, 11, 17, 40, 100]
AGE_BAND_LABELS= ["Child", "Adolescent", "Adult", "Older_Adult"]

# ── Sensitivity target for threshold optimisation ────────────────────────────
MIN_RECALL_TARGET = 0.88   # clinical requirement: catch >= 88% of true ASD cases

# ── Uncertainty flag ─────────────────────────────────────────────────────────
UNCERTAINTY_STD_THRESHOLD = 0.15
