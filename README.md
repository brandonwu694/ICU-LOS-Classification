# ICU Length of Stay Classification

This capstone project predicts an ICU length-of-stay category using information available at ICU admission and during the first 24 ICU hours. The goal is to make the prediction early enough that it could support planning, while avoiding information from later in the stay.

Target classes:

| Class | Definition |
| --- | --- |
| `0` | ICU LOS `< 2` days |
| `1` | ICU LOS `2` through `7` days, inclusive |
| `2` | ICU LOS `> 7` days |

## Repository Structure

```text
.
├── src/
│   ├── data/          # target labels, patient-level splits, leakage checks
│   ├── features/      # first-24-hour raw and processed feature builders
│   ├── models/        # preprocessing, classifier, and baseline pipelines
│   └── evaluation/    # metrics and report writers
├── scripts/
│   └── train_best_model.py
├── tests/
│   └── test_classification_pipeline.py
├── notebooks/
│   ├── 01_los_category_validation.ipynb
│   ├── 02_model_performance_checks.ipynb
│   └── 03_project.ipynb
├── data/
│   └── sample/        # synthetic demo data only
├── models/            # saved model artifacts
└── reports/           # metrics, confusion matrices, splits, predictions
```

## Leakage Controls

The pipeline splits data by `subject_id`, so all ICU stays for the same patient stay in the same train, validation, or test split. It also checks that no patient appears in more than one split.

Features use only admission-time information and ICU hour 0-24 summaries. Fields that reveal the outcome or future care, such as LOS, discharge/death timestamps, ICU `outtime`, and `last_careunit`, are removed before modeling. Preprocessing, encoding, scaling, model fitting, and class-imbalance weighting are fit only on the training split.

## Feature Sources

The default path can reuse `data/processed/modeling_dataset.parquet` when it exists and then filters known leakage columns. The raw rebuild path creates these first-24-hour feature groups:

| Feature group | Examples | First-24-hour handling |
| --- | --- | --- |
| Admission context and demographics | age, sex, admission type, insurance, race, first ICU care unit, hospital-to-ICU time | Uses information known at or before ICU admission. |
| Vitals from chart events | heart rate, blood pressure, respiratory rate, oxygen saturation, temperature-like charted measurements | Uses all numeric measurements from ICU hour 0-24, then summarizes each common item with count, mean, minimum, and maximum. |
| Labs | creatinine, white blood cell count, hemoglobin, electrolytes, lactate-like lab measurements | Uses all numeric lab results charted from ICU hour 0-24, then summarizes each common item with count, mean, minimum, and maximum. |
| Inputs | total input volume, input event count, unique input items, common input categories | Keeps input events started within ICU hour 0-24 and aggregates volumes/counts by stay. |
| Outputs | total output volume, urine-related output, output event count, unique output items | Keeps output events charted within ICU hour 0-24 and aggregates volumes/counts by stay. |
| Procedures | procedure count, unique procedures, common procedure indicators | Keeps procedures started within ICU hour 0-24 and creates count/binary indicators. |
| Prescriptions | medication count, unique drugs, routes, broad medication categories | Keeps prescriptions started within ICU hour 0-24 and creates count/binary indicators. |
| Radiology | note count, modality/body-region indicators, simple timestamp-safe keywords | Keeps radiology notes charted within ICU hour 0-24 and uses only simple indicators from those notes. |

All time-stamped raw rows are joined to ICU stays and filtered between ICU `intime` and `intime + 24 hours` before aggregation. Outcome and future-information fields are not used as predictors.

## Model Performance

Held-out test performance from `reports/classification/model_comparison.csv`:

| Model | Macro F1 | Weighted F1 | Balanced Accuracy | ROC AUC Macro |
| --- | ---: | ---: | ---: | ---: |
| HistGradientBoosting tuned | 0.599 | 0.648 | 0.625 | 0.816 |
| Random Forest | 0.579 | 0.649 | 0.566 | 0.807 |
| Logistic Regression | 0.561 | 0.608 | 0.604 | 0.785 |
| Dummy Majority | 0.225 | 0.342 | 0.333 | 0.500 |

The selected HistGradientBoosting model was tuned with patient-group-aware cross-validation. Best parameters from the tuned run:

```text
model__learning_rate: 0.02
model__max_iter: 500
model__max_leaf_nodes: 45
model__min_samples_leaf: 50
model__l2_regularization: 0.01
model__max_bins: 128
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The saved joblib model was created with scikit-learn `1.8.0`, so `requirements.txt` pins that version for reproducible loading.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Train The Classifier

Train from the processed first-24-hour modeling dataset when it is available:

```bash
python scripts/train_best_model.py
```

Rebuild features directly from `data/raw/` instead:

```bash
python scripts/train_best_model.py --from-raw
```

Optionally tune the HistGradientBoosting model with patient-group-aware cross-validation on the training split:

```bash
python scripts/train_best_model.py --tune-hgb --tuning-iterations 20
```

Create a small synthetic demo model artifact:

```bash
python scripts/train_best_model.py --sample
```

The active model artifact is saved to:

```text
models/icu_los_classifier.joblib
models/icu_los_classifier_metadata.json
```

Generated comparison model artifacts can stay local to avoid pushing large files. The selected `icu_los_classifier.joblib` artifact is the model to include when sharing a runnable demo or submission.

The training script also fits comparison models on the same patient-level split:

- `dummy_most_frequent`: majority-class baseline that ignores predictors.
- `logistic_regression`: regularized class-weighted logistic regression using the same preprocessing pipeline.
- `random_forest`: class-weighted random forest comparison model.

Baseline artifacts are saved to:

```text
models/icu_los_classifier_dummy_most_frequent.joblib
models/icu_los_classifier_logistic_regression.joblib
models/icu_los_classifier_random_forest.joblib
```

Reports are saved under:

```text
reports/classification/
```

Reports include macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1/support, a confusion matrix, and multiclass ROC AUC when probabilities are available. The model-level comparison is written to:

```text
reports/classification/model_comparison.csv
```

When `--tune-hgb` is used, tuning results are written to:

```text
reports/classification/hgb_tuning_results.csv
```

## Run The Demo Notebook

The demo uses only synthetic data and should run in under 1 minute:

```bash
jupyter notebook notebooks/03_project.ipynb
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

## Review Model Performance

Run the performance and assumption-check notebook:

```bash
jupyter notebook notebooks/02_model_performance_checks.ipynb
```

This notebook reviews saved model reports and checks patient-level split integrity, target labels, leakage-prone feature names, feature compatibility with the saved model, and prediction output shape.

To compare the selected LOS categories with an alternative `<=7`, `7-14`, `>14` day definition, run:

```bash
jupyter notebook notebooks/01_los_category_validation.ipynb
```

## Restricted MIMIC-IV Data

The full project uses restricted MIMIC-IV data from PhysioNet. These data are not included in this repository. They must be obtained through PhysioNet credentialing and the required data-use agreement. After approval, place the source files locally under `data/raw/`.
