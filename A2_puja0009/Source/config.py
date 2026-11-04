"""Shared A2 locations, resolved independently of the working directory."""
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
A2_DIR = SOURCE_DIR.parent
MISC_DIR = A2_DIR / "Misc"
DATA_DIR = MISC_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_PATH = PROCESSED_DATA_DIR / "electricity_transformed.parquet"
SAVED_MODELS_DIR = MISC_DIR / "saved_models"
REPORTS_DIR = A2_DIR / "Reports"
DATA_PIPELINE_REPORT_DIR = MISC_DIR / "data_pipeline"
PRICE_REPORT_DIR = MISC_DIR / "price"
RULES_DIR = SOURCE_DIR / "prolog"
ENV_PATH = SOURCE_DIR / ".env"
