import logging
import pandas as pd

from src.schema_mapper import map_columns
from src.ingestion import load_csv

logging.basicConfig(level = logging.INFO)

file_path = r"data\raw\synthetic_transactions_messy.csv"

df = load_csv(file_path)

if df is not None:
    mapped_df = map_columns(df)

    if mapped_df is not None:
        print(df.columns.tolist())
        print(mapped_df.columns.tolist())
        print(mapped_df.head())
        print(df.columns.tolist())

        
print("\nTesting mapping conflict:")

conflict_df = pd.DataFrame({
    "user_id": ["C001"],
    "customer_id": ["C002"],
    "amount": [500]
})

conflict_result = map_columns(conflict_df)

if conflict_result is None:
    print("PASS: Mapping conflict detected.")
else:
    print("FAIL: Mapping conflict was not detected.")
