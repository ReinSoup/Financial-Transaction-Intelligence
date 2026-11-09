import pandas as pd
import pytest

from src.insight_generation import generate_insights


def test_spending_and_cluster_insights_match_source_values():
    customer_features = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "share_Housing": [40.0, 10.0],
        "share_Food": [60.0, 90.0],
        "debit_transaction_count": [20, 10],
    })
    assignments = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "behavioral_cluster": [0, 1],
    })
    anomalies = pd.DataFrame({
        "transaction_id": ["T1"],
        "customer_id": ["C1"],
        "category": ["Housing"],
        "anomaly_flag": ["normal"],
    })

    insights = generate_insights(customer_features, assignments, anomalies)
    housing_insight = insights.loc[
        insights["insight_type"] == "concentrated_category_share"
    ].set_index("customer_id")

    assert housing_insight.loc["C1", "category"] == "Food"
    assert housing_insight.loc["C1", "value"] == 60
    assert set(insights["insight_type"]) == {
        "concentrated_category_share",
        "cluster_transaction_frequency",
    }


def test_anomaly_insight_keeps_transaction_id_and_explanation():
    customer_features = pd.DataFrame({
        "customer_id": ["C1"],
        "share_Food": [100.0],
        "debit_transaction_count": [5],
    })
    assignments = pd.DataFrame({
        "customer_id": ["C1"],
        "behavioral_cluster": [0],
    })
    anomalies = pd.DataFrame({
        "transaction_id": ["T9"],
        "customer_id": ["C1"],
        "category": ["Food"],
        "anomaly_flag": ["anomaly"],
        "explanation": ["Amount was 5 times the prior customer median."],
    })

    insights = generate_insights(customer_features, assignments, anomalies)
    anomaly_insight = insights.loc[insights["insight_type"] == "transaction_anomaly"].iloc[0]

    assert anomaly_insight["transaction_id"] == "T9"
    assert anomaly_insight["insight_text"] == "Amount was 5 times the prior customer median."


def test_invalid_threshold_and_empty_inputs_are_handled():
    with pytest.raises(ValueError, match="share_threshold"):
        generate_insights(
            pd.DataFrame({"customer_id": ["C1"], "share_Food": [100]}),
            pd.DataFrame({"customer_id": ["C1"], "behavioral_cluster": [0]}),
            pd.DataFrame({"transaction_id": ["T1"], "anomaly_flag": ["normal"]}),
            share_threshold=101,
        )

    insights = generate_insights(
        pd.DataFrame({"customer_id": ["C1"], "share_Food": [10], "debit_transaction_count": [1]}),
        pd.DataFrame({"customer_id": ["C1"], "behavioral_cluster": [0]}),
        pd.DataFrame({"transaction_id": ["T1"], "anomaly_flag": ["normal"]}),
    )
    assert "transaction_anomaly" not in insights["insight_type"].tolist()


def test_empty_results_create_no_insights_and_duplicate_customer_ids_are_rejected():
    empty_insights = generate_insights(
        pd.DataFrame(columns=["customer_id", "share_Food", "debit_transaction_count"]),
        pd.DataFrame(columns=["customer_id", "behavioral_cluster"]),
        pd.DataFrame(columns=["transaction_id", "anomaly_flag"]),
    )

    assert empty_insights.empty

    duplicate_features = pd.DataFrame({
        "customer_id": ["C1", "C1"],
        "share_Food": [50, 50],
        "debit_transaction_count": [2, 2],
    })
    with pytest.raises(ValueError, match="present and unique"):
        generate_insights(
            duplicate_features,
            pd.DataFrame({"customer_id": ["C1"], "behavioral_cluster": [0]}),
            pd.DataFrame({"transaction_id": ["T1"], "anomaly_flag": ["normal"]}),
        )
