# ICU Length of Stay Survival Analysis

This capstone project predicts ICU length of stay from early ICU information, including demographics, admission context, first-day vital signs, first-day lab measurements, and first-day input events. The project compares survival-style accelerated failure time models with a log-transformed ICU length-of-stay gradient boosting model.

The original analysis uses restricted ICU data, so the public repository includes a small synthetic sample under `data/sample/` that can run the project demo notebook in under 1 minute.

## Repository Contents

| Path | Description |
| --- | --- |
| `README.md` | Project overview, file descriptions, setup instructions, and run instructions. |
| `requirements.txt` | Python packages needed to run the demo notebook and analysis notebooks. |
| `config.py` | Shared path configuration for local raw data, processed data, and model output directories. |
| `project.html` | Static HTML version of the project demo with rendered outputs. |
| `data/sample/README.md` | Description of the synthetic sample data files. |
| `data/sample/demo_predictions.csv` | Synthetic patient-level examples with observed ICU LOS, predicted ICU LOS, and prediction error. |
| `data/sample/demo_model_metrics.csv` | Representative final-model metrics used by the demo notebook. |
| `data/sample/demo_feature_importance.csv` | High-level feature group importance summary used for interpretation. |
| `data/sample/demo_subgroup_error.csv` | Care-unit subgroup error summary generated from the synthetic demo predictions. |
| `notebooks/project_demo.ipynb` | Fast public demo notebook. It imports the synthetic sample data and demonstrates model metrics, example predictions, error analysis, and interpretation. |
| `notebooks/01_icustays_data_cleaning.ipynb` | Cleans ICU stay, admission, and patient tables. |
| `notebooks/02_icustays_feature_engineering.ipynb` | Creates ICU stay demographic, admission, timing, and care-unit features. |
| `notebooks/03_chartevents_data_cleaning.ipynb` | Cleans chart event vital-sign data. |
| `notebooks/04_chartevents_feature_engineering.ipynb` | Creates first-24-hour vital-sign summary features. |
| `notebooks/05_labevents_data_cleaning.ipynb` | Cleans lab event data. |
| `notebooks/06_labevents_feature_engineering.ipynb` | Creates first-24-hour lab summary and missingness features. |
| `notebooks/07_inputevents_data_cleaning.ipynb` | Cleans ICU input event data. |
| `notebooks/08_inputevents_feature_engineering.ipynb` | Creates first-24-hour medication, fluid, duration, and dose features. |
| `notebooks/09_merge_modeling_dataset.ipynb` | Merges feature tables into the final modeling dataset. |
| `notebooks/10_model_readiness_assumptions.ipynb` | Checks missingness, correlations, predictor readiness, and train/test split assumptions. |
| `notebooks/11_survival_modeling.ipynb` | Fits and compares survival-style AFT models. |
| `notebooks/12_lognormal_aft_modeling.ipynb` | Fits the selected log-normal AFT model and exports metrics/predictions. |
| `notebooks/13_log_los_gradient_boosting.ipynb` | Fits the log-transformed ICU LOS gradient boosting model and exports metrics/predictions. |
| `notebooks/14_data_sanity_model_diagnostics.ipynb` | Runs final sanity checks, residual diagnostics, tail checks, and subgroup error analysis. |
| `notebooks/15_original_data_eda.ipynb` | Summarizes the original ICU cohort, length-of-stay distribution, and cohort composition. |

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Public Demo

The public demo uses only the synthetic files in `data/sample/` and should run in under 1 minute:

```bash
jupyter notebook notebooks/project_demo.ipynb
```

Run all cells from top to bottom. The same rendered results are also saved in:

```text
project.html
```

## Full Pipeline With Restricted Data

The full analysis requires access to the original ICU source tables. These files are intentionally not included in the repository.

Place the restricted source files in `data/raw/` with these names:

```text
admissions.csv
chartevents.csv.gz
d_items.csv
d_labitems.csv
icustays.csv
inputevents.csv
labevents.csv.gz
patients.csv
```

Then run the notebooks in numeric order:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15
```

The cleaning and feature notebooks write intermediate files to `data/processed/`. The modeling notebooks write final metrics, predictions, and diagnostics to `data/processed/model_outputs/`. Because the source data are restricted and the processed files can be large, `data/` is ignored by git except for the small synthetic demo sample.

## Final Result

The final comparison showed that the log-transformed ICU length-of-stay gradient boosting model performed best among the evaluated models, with stronger discrimination and slightly lower mean absolute error than the log-normal AFT model. Long ICU stays remained the hardest cases to predict from first-day data alone, so the final analysis emphasizes error distributions, subgroup diagnostics, and prediction-tail checks rather than only average performance.

## Notes On Data Privacy

The included `data/sample/` files are synthetic. They are designed to demonstrate the notebook workflow and expected outputs without exposing restricted patient-level ICU records.
