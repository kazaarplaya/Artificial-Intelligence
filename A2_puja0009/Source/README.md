# Artifact 2 Artificial Intelligence Capstone Project

This is the working repository for the A2 submission for Group P, COMP3742 Artificial Intelligence.

**Group Members:**

| **Name**       | **FAN**  |
|----------------|----------|
| Oliver Wuttke  | WUTT0019 |
| Hans Pujalte   | PUJA0009 |
| Shivansh Pant  | PANT0108 |
| Matilda Alford | ALFO0043 |

Each Python function and method contains an author credit with the name and FAN of the credited group member, including files with a single author. Prolog predicates have equivalent author comments. Shared pipeline functions credit all four group members; combined Prolog assessment predicates credit Group P, with integration by Hans Pujalte.

## Layout

A2 contains only `Source`, `Reports`, and `Misc` at its top level.

- `Source`: Python code, Prolog rules, requirements and the local `.env` file.
- `Misc/data/raw` and `Misc/data/processed`: downloaded and prepared datasets.
- `Misc/saved_models`: all fitted models and network-stress metadata.
- `Misc/notebooks`: network-stress training and historical inference notebooks.
- `Misc/data_pipeline`: cleaning, transformation and exploratory modelling reports.
- `Misc/price`: price evaluation reports and charts.
- `Reports`: submission documents, including the self-assessment PDF. Other subsystems currently print their results, and demand also displays a chart.

`config.py` resolves default locations relative to this source directory, independently
of the working directory. Explicit `--input` paths are relative to your current directory.

## Setup

Use Python 3.11–3.13 and an environment with the packages from `requirements.txt`.
Install SWI-Prolog separately so PySwip can load its shared library.
From the repository root, install Python requirements into your active environment:

```bash
python -m pip install -r A2_puja0009/Source/requirements.txt
```

Create `Source/.env` containing your OpenElectricity API key:

```text
OPENELECTRICITY_API_KEY=<your API key>
```

This file is excluded from Git. An existing environment variable takes precedence.
Historical data access requires an appropriate OpenElectricity account.

## Prepare data

From the repository root:

```bash
python A2_puja0009/Source/A1_data_pipeline/run_pipeline.py
```

This fetches the configured historical period, cleans and transforms the data,
and runs A1's exploratory modelling. Data is saved under `Misc/data` and reports
under `Misc/data_pipeline`. The date constants are in that script.

## Machine-learning training

Run training separately after preparing the data:

```bash
python A2_puja0009/Source/training_pipeline.py
```

The default input is `Misc/data/processed/electricity_transformed.parquet`.
To use another dataset:

```bash
python A2_puja0009/Source/training_pipeline.py --input /absolute/path/to/dataset.parquet
```

Training loads the shared dataset once and trains demand, price spike, network stress
and renewable generation in sequence. Close the demand plot window to continue if
it blocks. The dataset must cover the models' configured training and evaluation
periods: price uses 2024 validation and 2025-onwards testing; demand starts testing
on 24 September 2025 with a preceding 30-day training window.

Models are saved under `Misc/saved_models`:

- `demand_gp.npz`
- `price_spike_model.joblib`
- `network_stress_model.joblib`
- `network_stress_model_metadata.json`
- `renewable_gb.npz`

The pre-existing network-stress artifact and metadata have been moved here.
Train all four models before running the complete inference pipeline. Existing
artifacts may need retraining if their dependency versions differ from your environment.

## Inference

Run inference separately:

```bash
python A2_puja0009/Source/inference.py
```

This fetches recent history once and runs all four subsystems, renewable coverage,
and the combined system assessment. It prints results to the terminal.
It does not trigger training or historical data preparation.

From inside `Source`, the equivalent commands are
`python A1_data_pipeline/run_pipeline.py`, `python training_pipeline.py`, and
`python inference.py`. There is no combined entry point.

The notebooks in `Misc/notebooks` use the same dataset and model locations.
They support launching from the repository root, A2, or within its subdirectories.

## Combined assessment

`prolog/system_assessment.pl` coordinates subsystem evidence.
`prolog/SystemAssessment.py` validates aligned one-hour forecast times and records
subsystem failures. Missing or invalid evidence prevents an all-clear. Renewable
coverage uses miniKanren and requires matching demand and renewable forecasts.

The combined layer is advisory: it does not execute actions or certify grid safety.
A renewable-share shortfall is a target issue, not proof of insufficient total supply.
Subsystem failures are displayed; inspect the assessment and errors rather than
assuming a successful process exit means every subsystem succeeded.
