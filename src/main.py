import logging
from ingestion import load_csv
logging.basicConfig(level=logging.INFO)
file_path = "data/raw/sample_transactions.csv"

df = load_csv(file_path)

