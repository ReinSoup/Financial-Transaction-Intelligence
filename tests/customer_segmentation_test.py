import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation import (
    build_customer_behavior_features,
    evaluate_cluster_counts,
    get_behavior_feature_columns,
    segment_customers,
)


def create_transactions():
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(8)],
        "customer_id": ["C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3"],
        "amount": [-80, -20, 1000, -10, -90, -40, -30, -30],
        "category": ["Food", "Travel", "Income", "Food", "Travel", "Food", "Food", "Travel"],
        "transaction_type": ["debit", "debit", "credit", "debit", "debit", "debit", "debit", "debit"],
        "is_weekend": [False, True, False, False, False, True, False, True],
        "is_injected_anomaly": [0, 0, 0, 0, 0, 0, 1, 0],
    })


def test_category_shares_sum_to_one_hundred_and_align_to_customers():
    features = build_customer_behavior_features(create_transactions())
    share_columns = [column for column in features.columns if column.startswith("share_")]

    assert features["customer_id"].tolist() == ["C1", "C2", "C3"]
    assert np.allclose(features[share_columns].sum(axis=1), 100)
    assert features["debit_transaction_count"].tolist() == [2, 2, 3]


def test_customer_without_debit_transactions_gets_zero_features():
    transactions = create_transactions()
    transactions.loc[transactions["customer_id"] == "C1", "transaction_type"] = "credit"

    features = build_customer_behavior_features(transactions).set_index("customer_id")

    assert features.loc["C1", "debit_transaction_count"] == 0
    assert features.loc["C1", "average_debit_amount"] == 0
    assert features.loc["C1", "weekend_transaction_share"] == 0
    assert features.loc["C1", [column for column in features if column.startswith("share_")]].sum() == 0


def test_model_feature_list_excludes_identifiers_and_anomaly_labels():
    features = build_customer_behavior_features(create_transactions())
    columns = get_behavior_feature_columns(features)

    assert "customer_id" not in columns
    assert "is_injected_anomaly" not in columns
    assert set(columns).issubset(features.columns)


def test_segmentation_is_reproducible_and_keeps_customer_alignment():
    features = build_customer_behavior_features(create_transactions())
    first = segment_customers(features, n_clusters=2)
    second = segment_customers(features, n_clusters=2)

    pd.testing.assert_frame_equal(first["assignments"], second["assignments"])
    assert first["assignments"]["customer_id"].tolist() == features["customer_id"].tolist()
    assert first["profile"].shape[0] == 2


def test_cluster_evaluation_rejects_invalid_cluster_counts():
    features = build_customer_behavior_features(create_transactions())

    with pytest.raises(ValueError, match="cluster count"):
        evaluate_cluster_counts(features, cluster_counts=[3])


def test_missing_columns_and_invalid_amounts_raise_errors():
    transactions = create_transactions()

    with pytest.raises(ValueError, match="Missing required columns"):
        build_customer_behavior_features(transactions.drop(columns=["category"]))

    transactions["amount"] = transactions["amount"].astype(float)
    transactions.loc[0, "amount"] = np.inf
    with pytest.raises(ValueError, match="finite numeric"):
        build_customer_behavior_features(transactions)
