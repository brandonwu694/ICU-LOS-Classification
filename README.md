# ICU Length of Stay Classification

This capstone project now predicts ICU length-of-stay category from information available at ICU admission through the first 24 hours only.

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
│   └── project.ipynb
├── data/
│   └── sample/        # synthetic demo data only
├── models/            # saved model artifacts
└── reports/           # metrics, confusion matrices, splits, predictions
```

## Leakage Controls

The pipeline splits by `subject_id` and asserts that no patient appears in more than one split. Features are restricted to admission-time values and ICU hour 0-24 aggregates. Outcome and future-information fields such as LOS, discharge/death timestamps, ICU `outtime`, and `last_careunit` are excluded from the feature matrix.

Preprocessing, encoding, scaling, model fitting, and class-imbalance weighting are fit only on the training split.

## Feature Sources

The default path can reuse `data/processed/modeling_dataset.parquet` when it exists, after filtering known leakage columns. The raw rebuild path creates admission context, demographics, first-24-hour vitals/labs, input and output summaries, procedure indicators, prescription summaries, and timestamp-safe radiology-note indicators.

All time-stamped raw features are joined to ICU stays and filtered between ICU `intime` and `intime + 24 hours` before aggregation.

## Model Performance

Held-out test performance from `reports/classification/model_comparison.csv`:

| Model | Macro F1 | Weighted F1 | Balanced Accuracy | ROC AUC Macro |
| --- | ---: | ---: | ---: | ---: |
| HistGradientBoosting | 0.590 | 0.640 | 0.617 | 0.814 |
| Logistic Regression | 0.561 | 0.608 | 0.604 | 0.785 |
| Dummy Majority | 0.225 | 0.342 | 0.333 | 0.500 |

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

The training script also fits two comparison baselines on the same patient-level split:

- `dummy_most_frequent`: majority-class baseline that ignores predictors.
- `logistic_regression`: regularized class-weighted logistic regression using the same preprocessing pipeline.

Baseline artifacts are saved to:

```text
models/icu_los_classifier_dummy_most_frequent.joblib
models/icu_los_classifier_logistic_regression.joblib
```

Reports are saved under:

```text
reports/classification/
```

Metrics include macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1/support, confusion matrix, and multiclass ROC AUC when probabilities are available. The model-level comparison is written to:

```text
reports/classification/model_comparison.csv
```

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

The full project requires restricted MIMIC-IV data from PhysioNet. These data are not included in this repository and must be obtained externally through PhysioNet credentialing and the required data-use agreement. After approval, place the source files locally under `data/raw/`.

Only add new sources if the resulting features can be computed from ICU admission through hour 24 and do not reveal discharge timing, death timing, final LOS, or events after hour 24.
