
import pandas as pd
import pytest

from src.data_enrichment import enrich_data


def create_test_data():
    """Create a small dataset for testing enrichment."""

    data = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "date": pd.to_datetime([
            "2025-01-03",  # Friday
            "2025-01-04",  # Saturday
            "2025-01-05",  # Sunday
        ]),
        "amount": [-500.0, -250.0, 1000.0],
    })

    return data


def test_date_features_are_created():
    """Check that all four date features are created."""

    data = create_test_data()

    result = enrich_data(data)
    enriched = result["enriched"]

    expected_columns = [
        "year",
        "month",
        "day_of_week",
        "is_weekend",
    ]

    for column in expected_columns:
        assert column in enriched.columns


def test_date_features_have_correct_values():
    """Check that date features contain the expected values."""

    data = create_test_data()

    result = enrich_data(data)
    enriched = result["enriched"]

    assert enriched["year"].tolist() == [2025, 2025, 2025]
    assert enriched["month"].tolist() == [1, 1, 1]

    assert enriched["day_of_week"].tolist() == [
        "Friday",
        "Saturday",
        "Sunday",
    ]

    assert enriched["is_weekend"].tolist() == [
        False,
        True,
        True,
    ]


def test_row_count_is_preserved():
    """Check that enrichment does not add or remove rows."""

    data = create_test_data()

    result = enrich_data(data)
    enriched = result["enriched"]

    assert len(enriched) == len(data)


def test_original_data_is_not_modified():
    """Check that enrichment preserves the input DataFrame."""

    data = create_test_data()
    original_data = data.copy(deep=True)

    enrich_data(data)

    pd.testing.assert_frame_equal(data, original_data)


def test_report_is_correct():
    """Check that the report describes the enrichment operation."""

    data = create_test_data()

    result = enrich_data(data)
    report = result["report"]

    assert report["input_rows"] == 3
    assert report["output_rows"] == 3

    assert report["date_features_added"] == [
        "year",
        "month",
        "day_of_week",
        "is_weekend",
    ]


def test_missing_date_column_raises_error():
    """Check that enrichment rejects data without a date column."""

    data = pd.DataFrame({
        "customer_id": ["C001"],
        "amount": [-500.0],
    })

    with pytest.raises(
        ValueError,
        match="Required column 'date' is missing.",
    ):
        enrich_data(data)
