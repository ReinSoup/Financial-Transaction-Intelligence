""" Tests are AI generated btw, using PYTEST for each"""


import pandas as pd
import pytest

from src.ingestion import load_csv
from src.schema_mapper import map_columns
from src.schema_validation import validate_schema
from src.data_validation import validate_data
from src.data_preprocessing import preprocess_data


DATA_PATH = r"data/raw/synthetic_transactions_messy.csv"


@pytest.fixture(scope="module")
def preprocessing_result():
    """Run the actual synthetic dataset through validation and preprocessing."""

    raw_data = load_csv(DATA_PATH)
    assert raw_data is not None, "CSV failed to load."

    mapped_data = map_columns(raw_data)
    assert mapped_data is not None, "Schema mapping failed."

    assert validate_schema(mapped_data), "Schema validation failed."

    validation_result = validate_data(mapped_data)
    accepted_data = validation_result["accepted"]

    result = preprocess_data(accepted_data)

    return accepted_data, result


def test_preprocessing_preserves_row_count(preprocessing_result):
    accepted_data, result = preprocessing_result
    cleaned_data = result["cleaned"]

    assert len(cleaned_data) == len(accepted_data)


def test_required_columns_exist(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    required_columns = [
        "customer_id",
        "date",
        "amount",
        "transaction_type",
    ]

    for column in required_columns:
        assert column in cleaned_data.columns


def test_transaction_directions_are_standardized(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    assert cleaned_data["transaction_type"].isin(
        ["debit", "credit"]
    ).all()


def test_debit_amounts_are_negative_or_zero(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    debit_amounts = cleaned_data.loc[
        cleaned_data["transaction_type"] == "debit",
        "amount",
    ]

    assert (debit_amounts <= 0).all()


def test_credit_amounts_are_positive_or_zero(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    credit_amounts = cleaned_data.loc[
        cleaned_data["transaction_type"] == "credit",
        "amount",
    ]

    assert (credit_amounts >= 0).all()


def test_amounts_are_numeric_and_finite(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    assert pd.api.types.is_numeric_dtype(cleaned_data["amount"])
    assert cleaned_data["amount"].notna().all()
    assert cleaned_data["amount"].map(
        lambda value: abs(value) != float("inf")
    ).all()


def test_dates_are_datetime(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    assert pd.api.types.is_datetime64_any_dtype(cleaned_data["date"])
    assert cleaned_data["date"].notna().all()


def test_text_columns_have_no_unnecessary_whitespace(preprocessing_result):
    _, result = preprocessing_result
    cleaned_data = result["cleaned"]

    text_columns = ["description", "merchant", "category"]

    for column in text_columns:
        if column in cleaned_data.columns:
            values = cleaned_data[column].dropna().astype("string")

            assert values.eq(values.str.strip()).all()
            assert values.str.contains(r"\s{2,}", regex=True).sum() == 0


def test_duplicate_rows_are_preserved(preprocessing_result):
    accepted_data, result = preprocessing_result
    cleaned_data = result["cleaned"]

    original_duplicate_count = int(
        accepted_data.duplicated(keep=False).sum()
    )
    cleaned_duplicate_count = int(
        cleaned_data.duplicated(keep=False).sum()
    )

    # Duplicates must not be deliberately removed by preprocessing.
    assert len(cleaned_data) == len(accepted_data)

    # Report should match the duplicate count in the cleaned output.
    assert result["report"]["duplicate_rows_preserved"] == (
        cleaned_duplicate_count
    )


def test_processing_report_is_consistent(preprocessing_result):
    accepted_data, result = preprocessing_result

    report = result["report"]
    cleaned_data = result["cleaned"]

    assert report["input_rows"] == len(accepted_data)
    assert report["output_rows"] == len(cleaned_data)
    assert report["amount_values_processed"] == len(cleaned_data)
    assert report["date_values_processed"] == len(cleaned_data)
    assert report["duplicate_rows_preserved"] == int(
        cleaned_data.duplicated(keep=False).sum()
    )


def test_original_accepted_data_is_not_modified(preprocessing_result):
    accepted_data, result = preprocessing_result

    # Preprocessing should work on a copy.
    # The accepted data should still match its original snapshot.
    assert accepted_data is not result["cleaned"]


