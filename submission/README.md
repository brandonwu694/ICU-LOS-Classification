# ICU LOS Classification Submission

This package demonstrates a saved ICU length-of-stay classifier on synthetic example data. The model artifact was trained offline on restricted MIMIC-IV-derived data from PhysioNet; those restricted data are not included here.

## Files

| Path | Description |
| --- | --- |
| `notebooks/project.ipynb` | Runnable demo notebook. Loads the saved model, reads the synthetic sample data, generates predictions, and reports example metrics. |
| `notebooks/project.html` | HTML export of the demo notebook with outputs shown. |
| `models/icu_los_classifier.joblib` | Saved real-data-trained HistGradientBoosting classification pipeline. |
| `models/icu_los_classifier_metadata.json` | Metadata describing the target, feature counts, feature window, and saved test metrics when available. |
| `data/sample/icu_los_classification_sample.csv` | Small synthetic dataset that matches the model feature schema. It is for demo inference only. |
| `data/sample/README.md` | Note describing the synthetic sample data. |
| `requirements.txt` | Python packages needed to run the notebook. |

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

The notebook should run in under 1 minute. It does not require access to restricted MIMIC-IV data.

## Restricted Data Note

The full training data are restricted MIMIC-IV data from PhysioNet and require credentialing plus a data-use agreement. They are intentionally excluded from this submission. The included CSV is synthetic and is only used to illustrate model inference.
