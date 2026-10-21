
import pandas as pd
import pytest

from src.data_validation import validate_data


def create_test_data():
    """Create a small dataset for testing validation rules."""

    data = {
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "date": [
            "2026-01-10",
            "2026-01-11",
            "not-a-date",
            "2026-01-13",
            "2026-01-14",
        ],
        "amount": ["500", "invalid", "200", "300", "400"],
        "description": [
            "Groceries",
            "Transport",
            "Salary",
            "",
            "Shopping",
        ],
        "category": [
            "Food",
            "Travel",
            "Income",
            "Bills",
            "",
        ],
        "transaction_type": [
            "debit",
            "debit",
            "credit",
            "debit",
            "unknown",
        ],
    }

    return pd.DataFrame(data)


def test_invalid_amount_is_rejected():
    """Rows with unparseable amounts should be rejected."""

    df = create_test_data()
    report = validate_data(df)

    rejected = report["rejected"]

    assert "C002" in rejected["customer_id"].values
    assert "Missing or invalid amount" in (
        rejected.loc[
            rejected["customer_id"] == "C002",
            "validation_issues",
        ].iloc[0]
    )


def test_invalid_date_is_rejected():
    """Rows with unparseable dates should be rejected."""

    df = create_test_data()
    report = validate_data(df)

    rejected = report["rejected"]

    assert "C003" in rejected["customer_id"].values


def test_missing_description_is_flagged():
    """Missing descriptions should be flagged, not rejected."""

    df = create_test_data()
    report = validate_data(df)

    flagged = report["flagged"]

    assert "C004" in flagged["customer_id"].values
    assert "C004" in report["accepted"]["customer_id"].values


def test_missing_category_is_flagged():
    """Missing categories should be flagged, not rejected."""

    df = create_test_data()
    report = validate_data(df)

    flagged = report["flagged"]

    assert "C005" in flagged["customer_id"].values
    assert "Missing category" in (
        flagged.loc[
            flagged["customer_id"] == "C005",
            "validation_issues",
        ].iloc[0]
    )


def test_duplicate_indices_do_not_mix_up_rows():
    """Repeated index labels must not cause incorrect row selection."""

    df = create_test_data().iloc[:2].copy()
    df.index = [0, 0]

    report = validate_data(df)

    assert len(report["accepted"]) == 1
    assert len(report["rejected"]) == 1

    assert report["accepted"]["customer_id"].iloc[0] == "C001"
    assert report["rejected"]["customer_id"].iloc[0] == "C002"

    # The original DataFrame's index should remain unchanged.
    assert df.index.tolist() == [0, 0]


def test_missing_required_column_raises_error():
    """Missing essential columns should produce a clear error."""

    df = create_test_data()
    df = df.drop(columns=["customer_id"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df)


def test_summary_row_counts_are_consistent():
    """Accepted and rejected rows should account for every input row."""

    df = create_test_data()
    report = validate_data(df)

    summary = report["summary"]

    assert summary["input_rows"] == len(df)

    assert summary["input_rows"] == (
        summary["accepted_rows"] + summary["rejected_rows"]
    )

    assert summary["flagged_rows"] <= summary["input_rows"]
