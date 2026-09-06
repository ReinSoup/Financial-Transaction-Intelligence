from src.ingestion.pipeline import run_ingestion_pipeline

df = run_ingestion_pipeline(
    "data/raw/sample_transactions.csv"
)

print(df)