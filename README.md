# ICU Length of Stay Classification

This capstone project now predicts ICU length-of-stay category from information available at ICU admission through the first 24 hours only.

Target classes:

| Class | Definition |
| --- | --- |
| `0` | ICU LOS `< 2` days |
| `1` | ICU LOS `2` through `7` days, inclusive |
| `2` | ICU LOS `> 7` days |

The previous survival-analysis and log-LOS regression work is preserved under `legacy_survival/`.

## Repository Structure

| Path | Description |
| --- | --- |
| `src/data/` | Target creation, patient-level splitting, first-24-hour validation, and leakage checks. |
| `src/features/` | Feature builders for admission-time and first-24-hour ICU data. |
| `src/models/` | Scikit-learn preprocessing and classifier pipeline. |
| `src/evaluation/` | Macro/weighted F1, balanced accuracy, per-class metrics, confusion matrix, and ROC AUC helpers. |
| `scripts/train_best_model.py` | Main training CLI for the three-class classifier. |
| `tests/` | Lightweight sanity checks for target labels, split integrity, leakage columns, first-24-hour timestamps, and preprocessing column consistency. |
| `notebooks/project.ipynb` | Fast public demo that loads synthetic sample data and the saved model artifact. |
| `data/sample/` | Synthetic demo data. No restricted patient records are included. |
| `models/` | Saved model artifacts and metadata. |
| `reports/` | Classification metrics, confusion matrices, split files, and predictions. |
| `legacy_survival/` | Archived survival/log-LOS notebooks, script, and prior model artifacts. |

## Leakage Controls

The active pipeline explicitly:

- Splits by `subject_id`, so all ICU stays for a patient are assigned to exactly one of train, validation, or test.
- Runs an assertion that no `subject_id` appears in more than one split.
- Excludes outcome/future-information columns from features, including `los`, `outtime`, `dischtime`, `deathtime`, `dod`, `hospital_expire_flag`, `event_observed`, `duration`, and `last_careunit`.
- Uses `first_careunit`, not `last_careunit`.
- Uses admission-time values and first-24-hour aggregate features only.
- Fits imputation, scaling, one-hot encoding, and model parameters only on the training split.
- Handles imbalance with balanced sample weights computed on the training labels only.

## Feature Sources

The training script can reuse existing notebook-generated processed feature tables when `data/processed/modeling_dataset.parquet` exists. Those features are filtered to remove known leakage columns and prior survival/regression targets.

The raw-data feature path uses:

- `icustays`: ICU identifiers, `intime`, `first_careunit`, and LOS target creation.
- `admissions`: admission type/location, insurance, language, marital status, race, and hospital-to-ICU timing from `admittime`.
- `patients`: gender and anchor age.
- Optional first-24-hour numeric summaries from `chartevents` and `labevents` when present.

All event summaries are filtered to ICU hour `0` through `24` before aggregation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Train The Classifier

Use existing processed first-24-hour feature artifacts when available:

```bash
python scripts/train_best_model.py
```

Build features directly from `data/raw/` instead:

```bash
python scripts/train_best_model.py --from-raw
```

Create a tiny synthetic demo model artifact:

```bash
python scripts/train_best_model.py --sample
```

The active model artifact is saved to:

```text
models/icu_los_classifier.joblib
models/icu_los_classifier_metadata.json
```

Reports are saved under:

```text
reports/classification/
```

Metrics include macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1/support, confusion matrix, and multiclass ROC AUC when probabilities are available.

## Run The Demo Notebook

The public demo uses only synthetic data and should run in under 1 minute:

```bash
jupyter notebook notebooks/project.ipynb
```

The demo loads:

```text
data/sample/icu_los_classification_sample.csv
models/icu_los_classifier_sample.joblib
```

If the model artifact is missing, generate it with:

```bash
python scripts/train_best_model.py --sample
```

## Restricted MIMIC-IV Data

The full project requires restricted MIMIC-IV ICU source tables. They are not included in this repository and must be obtained externally through the official PhysioNet credentialing and data-use process. After approval, place source files in `data/raw/` using names such as:

```text
admissions.csv
chartevents.csv.gz
d_items.csv
d_labitems.csv
icustays.csv
inputevents.csv
labevents.csv.gz
outputevents.csv.gz
patients.csv
prescriptions.csv.gz
procedureevents.csv.gz
radiology.csv.gz
```

Only add new sources if the resulting features can be computed from ICU admission through hour 24 and do not reveal discharge timing, death timing, final LOS, or events after hour 24.
