import shutil
import tempfile
from pathlib import Path

from src.data_enrichment import enrich_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import customer_features
from src.ingestion import load_csv
from src.schema_mapper import map_columns
from src.schema_validation import validate_schema
from src.data_validation import validate_data


def test_ingestion_pipeline_writes_outputs_to_temporary_directory(monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    raw_file = repository_root / "data/raw/synthetic_transactions_messy.csv"
    temporary_parent = repository_root / ".pytest_cache" / "pipeline_test"
    temporary_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temporary_parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_raw_file = temporary_root / "data/raw/synthetic_transactions_messy.csv"
        temporary_processed_directory = temporary_root / "data/processed"
        temporary_raw_file.parent.mkdir(parents=True)
        temporary_processed_directory.mkdir(parents=True)
        shutil.copy2(raw_file, temporary_raw_file)

        with monkeypatch.context() as patcher:
            patcher.chdir(temporary_root)
            raw_data = load_csv("data/raw/synthetic_transactions_messy.csv")
            assert raw_data is not None

            mapped_data = map_columns(raw_data)
            assert mapped_data is not None
            assert validate_schema(mapped_data)

            validation_result = validate_data(mapped_data)
            preprocessing_result = preprocess_data(validation_result["accepted"])
            cleaned_data = preprocessing_result["cleaned"]
            enrichment_result = enrich_data(cleaned_data)
            enriched_data = enrichment_result["enriched"]
            customer_feature_data = customer_features(enriched_data)

            enriched_path = temporary_processed_directory / "enriched_transactions.csv"
            customer_features_path = temporary_processed_directory / "customer_features.csv"
            enriched_data.to_csv(enriched_path, index=False)
            customer_feature_data.to_csv(customer_features_path, index=False)

            assert len(enriched_data) == len(cleaned_data)
            assert enriched_path.is_file()
            assert customer_features_path.is_file()
            assert customer_feature_data["customer_id"].is_unique
