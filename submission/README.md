# ICU LOS Classification Submission

This package demonstrates a saved ICU length-of-stay classifier using synthetic example data. The model artifact was trained offline on restricted MIMIC-IV-derived data from PhysioNet. Those restricted data are not included here.

## Files

| Path | Description |
| --- | --- |
| `notebooks/project.ipynb` | Runnable demo notebook. It loads the saved model, reads the synthetic sample data, generates predictions, and reports example metrics. |
| `project.html` | HTML export of the demo notebook with outputs shown, named for the submission requirement. |
| `notebooks/project.html` | Copy of the same HTML export stored next to the notebook. |
| `models/icu_los_classifier.joblib` | Saved real-data-trained HistGradientBoosting classification pipeline. |
| `models/icu_los_classifier_metadata.json` | Metadata for the saved model, including the target, feature counts, feature window, and saved test metrics when available. |
| `data/sample/icu_los_classification_sample.csv` | Small synthetic dataset that matches the model feature schema. It is only for demo inference. |
| `data/sample/README.md` | Note describing the synthetic sample data. |
| `src/data/target.py` | LOS category creation logic. |
| `src/data/splitting.py` | Patient-level train/validation/test split utilities. |
| `src/data/validation.py` | Leakage-column and split-integrity validation helpers. |
| `src/models/pipeline.py` | Preprocessing and classifier pipeline definitions, including the selected HistGradientBoosting model. |
| `requirements.txt` | Python packages needed to run the notebook. |


## Saved Model Summary

The included `icu_los_classifier.joblib` file is the tuned HistGradientBoosting model. It was trained on restricted MIMIC-IV-derived data and evaluated on a held-out patient-level test split.

| Metric | Held-out test value |
| --- | ---: |
| Macro F1 | 0.599 |
| Weighted F1 | 0.648 |
| Balanced accuracy | 0.625 |
| ROC AUC macro | 0.816 |

Best tuned HistGradientBoosting parameters:

```text
model__learning_rate: 0.02
model__max_iter: 500
model__max_leaf_nodes: 45
model__min_samples_leaf: 50
model__l2_regularization: 0.01
model__max_bins: 128
```

## Target Classes

| Class | Definition |
| --- | --- |
| `0` | ICU LOS `< 2` days |
| `1` | ICU LOS `2` through `7` days, inclusive |
| `2` | ICU LOS `> 7` days |

## Run The Demo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/project.ipynb
```

The notebook should run in under 1 minute and does not require access to restricted MIMIC-IV data. The saved joblib model was created with scikit-learn `1.8.0`, so `requirements.txt` pins that version for reproducible loading.

The `src/` files are included so the viewer can inspect the target definition, patient-level splitting, leakage checks, preprocessing, and model choice. The demo notebook does not depend on these files. Full retraining still requires the restricted source data.

## Restricted Data Note

The full training data are restricted MIMIC-IV data from PhysioNet and require credentialing plus a data-use agreement. They are intentionally excluded from this submission. The included CSV is synthetic and is only used to show model inference.
