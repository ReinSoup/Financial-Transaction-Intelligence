import logging

from src.ingestion import load_csv
from src.schema_mapper import map_columns
from src.schema_validation import validate_schema

logging.basicConfig(level = logging.INFO)

file_path = r"data\raw\synthetic_transactions_messy.csv"

df = load_csv(file_path)

if df is not None:
    mapped_df = map_columns(df)

    if mapped_df is not None:
        print("Mapped columns:", mapped_df.columns.tolist())
        is_valid = validate_schema(mapped_df)

        print("Schema Valid", is_valid)
