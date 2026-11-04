"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: run_pipeline.py
Date: 22-09-2026
Description: Runs the automated data ingestion pipeline from start to finish.
"""

# Imports
import os
import json

import pandas as pd

# Support both direct execution and importing as a package from Source.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from A1_data_pipeline.DataLoader import DataLoader
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR as A2_REPORTS_DIR, ENV_PATH
from dotenv import load_dotenv
from datetime import datetime

# Resolve paths from this file, independently of the launch directory.
load_dotenv(ENV_PATH)

# Pipeline constants
API_KEY: str | None = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE: datetime = datetime(2020, 1, 1)
END_DATE: datetime = datetime(2026, 9, 24)

# Directory
RAW_OUTPUT_PATH: Path = RAW_DATA_DIR / "electricity_raw.parquet"

CLEAN_DATA_DIR = PROCESSED_DATA_DIR
CLEAN_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_cleaned.parquet"
TRANSFORMED_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_transformed.parquet"
REPORTS_DIR = A2_REPORTS_DIR / "data_pipeline"
MODELLED_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_modelled.parquet"


def main():
    """Author: Oliver Wuttke - WUTT0019; Hans Pujalte - PUJA0009; Shivansh Pant - PANT0108; Matilda Alford - ALFO0043"""
    if not API_KEY:
        raise ValueError("Set OPENELECTRICITY_API_KEY in the environment or Source/.env.")
    etl: DataLoader = DataLoader(START_DATE, END_DATE, API_KEY)
    etl.fetch()

    # Access and/or assign underlying dataframe object
    df: pd.DataFrame = etl.data_set
    print(df.head())
    print(df.describe())

    # Save to parquet
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df)} rows and {len(df.columns)} columns to {RAW_OUTPUT_PATH}")

    # Clean data
    df_clean: pd.DataFrame = etl.clean()
    print(df_clean.head())
    print(df_clean.describe())

    # Save cleaned df as parquet
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_clean)} cleaned rows and {len(df_clean.columns)} columns to {CLEAN_OUTPUT_PATH}")

    # Store a report of what was done during cleaning.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cleaning_report.json").write_text(
        json.dumps(etl.cleaning_report, indent=2) + "\n"
    )

    # Transform cleaned data into features
    df_transformed: pd.DataFrame = etl.transform()
    print(df_transformed.head())
    print(df_transformed.describe())

    # Save transformed df as parquet
    df_transformed.to_parquet(TRANSFORMED_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_transformed)} transformed rows and {len(df_transformed.columns)} columns to {TRANSFORMED_OUTPUT_PATH}")

    # Store a report of what the transform did.
    (REPORTS_DIR / "transform_report.json").write_text(
        json.dumps(etl.transform_report, indent=2) + "\n"
    )

    # Perform exploratory modelling
    df_modelled: pd.DataFrame = etl.exploratory_modelling()

    # Save modelled df as parquet
    df_modelled.to_parquet(MODELLED_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_modelled)} modelled rows and {len(df_modelled.columns)} columns to {MODELLED_OUTPUT_PATH}")

    # Store a report of what the modelling did.
    (REPORTS_DIR / "modelling_report.json").write_text(
        json.dumps(etl.modelling_report, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()
