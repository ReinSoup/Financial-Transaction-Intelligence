import pandas as pd
import pytest

from src.statistical_analysis import (
    build_customer_debit_summary,
    compare_customer_groups,
    descriptive_statistics,
    iqr_outlier_bounds,
)


def test_descriptive_statistics_return_known_values():
    summary = descriptive_statistics([1, 2, 3, 4, 5]).iloc[0]

    assert summary["count"] == 5
    assert summary["mean"] == 3
    assert summary["median"] == 3
    assert summary["minimum"] == 1
    assert summary["maximum"] == 5
    assert summary["interquartile_range"] == 2


def test_iqr_bounds_identify_known_outlier():
    result = iqr_outlier_bounds([1, 2, 3, 4, 5, 100])

    assert result["lower_bound"] == -1.5
    assert result["upper_bound"] == 8.5
    assert result["outlier_count"] == 1


def test_customer_debit_summary_uses_one_row_per_customer():
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2"],
        "amount": [-100, -300, 2000, -50],
        "transaction_type": ["debit", "debit", "credit", "debit"],
    })
    summary = build_customer_debit_summary(transactions).set_index("customer_id")

    assert summary.loc["C1", "debit_transaction_count"] == 2
    assert summary.loc["C1", "average_debit_amount"] == 200
    assert summary.loc["C1", "total_debit_amount"] == 400
    assert summary.loc["C2", "median_debit_amount"] == 50


def test_welch_comparison_uses_customer_as_unit_and_returns_effect_interval():
    customer_summary = pd.DataFrame({
        "customer_id": ["S1", "S2", "S3", "P1", "P2", "P3"],
        "average_debit_amount": [100, 110, 120, 200, 210, 220],
    })
    customer_reference = pd.DataFrame({
        "customer_id": ["S1", "S2", "S3", "P1", "P2", "P3"],
        "customer_profile": ["student", "student", "student", "salaried_individual", "salaried_individual", "salaried_individual"],
    })
    result = compare_customer_groups(customer_summary, customer_reference).iloc[0]

    assert result["unit_of_analysis"] == "customer"
    assert result["first_group_count"] == 3
    assert result["second_group_count"] == 3
    assert result["mean_difference_first_minus_second"] == -100
    assert result["difference_ci_95_lower"] < result["difference_ci_95_upper"]
    assert 0 <= result["p_value"] <= 1


def test_empty_or_insufficient_inputs_raise_errors():
    with pytest.raises(ValueError, match="At least one"):
        descriptive_statistics([])

    with pytest.raises(ValueError, match="multiplier"):
        iqr_outlier_bounds([1, 2, 3], multiplier=0)

    with pytest.raises(ValueError, match="at least two valid customers"):
        compare_customer_groups(
            pd.DataFrame({"customer_id": ["S1"], "average_debit_amount": [100]}),
            pd.DataFrame({"customer_id": ["S1"], "customer_profile": ["student"]}),
        )
