# data/sample/

This directory contains a **50-row anonymised sample** of the processed
ASD screening dataset, provided for UI demonstration and pipeline smoke-testing.

## File: `ASD_sample_public.csv`

| Column | Description |
|--------|-------------|
| A1–A10 | Binarised AQ-10 item responses (0 or 1) |
| Age | Age in years (4–80) |
| Sex | 1 = Male, 0 = Female |
| Jaundice | 1 = Yes, 0 = No |
| Family_ASD | 1 = Yes, 0 = No |
| Age_Band | Child / Adolescent / Adult / Older_Adult |
| Total_Score | Sum of A1–A10 (for reference only, not used as model input) |
| Class | 0 = No ASD traits, 1 = ASD traits |

## Full dataset

The full dataset (`ASD_Combined_Enhanced_Final.csv`, 5,021 rows) is derived
from the UCI ASD Screening datasets (Thabtah, 2017–2018, CC BY 4.0).

Download the originals from:
- https://archive.ics.uci.edu/dataset/419 (Children)
- https://archive.ics.uci.edu/dataset/426 (Adults)

Then run `notebooks/00_data_preparation.ipynb` to reproduce the merged dataset.
