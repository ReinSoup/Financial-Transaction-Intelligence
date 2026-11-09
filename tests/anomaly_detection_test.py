import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    ANOMALY_FEATURE_COLUMNS,
    build_historical_anomaly_features,
    detect_transaction_anomalies,
    evaluate_anomaly_predictions,
)


def create_transactions(day_count=16):
    rows = []
    transaction_number = 1
    for day in range(day_count):
        for customer_id in ["C1", "C2"]:
            for transaction_type, category, amount in [
                ("debit", "Food", -100),
                ("credit", "Income", 1000),
            ]:
                if day == day_count - 1 and customer_id == "C1" and transaction_type == "debit":
                    amount = -900
                rows.append({
                    "transaction_id": f"T{transaction_number}",
                    "customer_id": customer_id,
                    "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                    "amount": amount,
                    "category": category,
                    "transaction_type": transaction_type,
                    "is_injected_anomaly": int(day == day_count - 1 and customer_id == "C1" and transaction_type == "debit"),
                })
                transaction_number += 1
    return pd.DataFrame(rows)


def test_historical_features_exclude_current_and_future_transactions():
    transactions = create_transactions()
    features = build_historical_anomaly_features(transactions, minimum_history=5)
    target = features.loc[features["transaction_id"] == "T61"].iloc[0]
    changed_future = transactions.copy()
    changed_future.loc[changed_future["transaction_id"] == "T61", "amount"] = -900000
    changed_features = build_historical_anomaly_features(changed_future, minimum_history=5)
    unchanged_prior = changed_features.loc[
        changed_features["transaction_id"] == "T21",
        "customer_prior_median",
    ].iloc[0]

    assert target["customer_prior_median"] == 100
    assert target["historical_amount_ratio"] == 9
    assert target["is_eligible"]
    assert unchanged_prior == 100


def test_sparse_customer_uses_category_history_and_early_rows_stay_unscored():
    transactions = create_transactions()
    new_customer_transaction = transactions.loc[transactions["transaction_id"] == "T1"].copy()
    new_customer_transaction["transaction_id"] = "NEW1"
    new_customer_transaction["customer_id"] = "C3"
    new_customer_transaction["date"] = pd.Timestamp("2025-01-08")
    new_customer_transaction["is_injected_anomaly"] = 0
    transactions = pd.concat([transactions, new_customer_transaction], ignore_index=True)

    features = build_historical_anomaly_features(transactions, minimum_history=5)
    sparse_row = features.loc[features["transaction_id"] == "NEW1"].iloc[0]
    early_row = features.loc[features["transaction_id"] == "T1"].iloc[0]

    assert pd.isna(sparse_row["customer_prior_median"])
    assert sparse_row["baseline_source"] == "category_history"
    assert sparse_row["is_eligible"]
    assert not early_row["is_eligible"]


def test_prediction_ids_are_preserved_and_labels_are_not_model_features():
    transactions = create_transactions()
    result = detect_transaction_anomalies(transactions, random_state=42)

    assert result["predictions"]["transaction_id"].is_unique
    assert set(result["predictions"]["transaction_id"]) == set(transactions["transaction_id"])
    assert "is_injected_anomaly" not in result["features"].columns
    assert "is_injected_anomaly" not in result["feature_columns"]
    assert result["feature_columns"] == ANOMALY_FEATURE_COLUMNS
    assert result["predictions"]["evaluation_period"].eq("test").any()
    normal = result["predictions"].loc[
        result["predictions"]["anomaly_prediction"] == 1
    ]
    unscored = result["predictions"].loc[
        result["predictions"]["anomaly_prediction"].isna()
    ]
    assert normal["explanation"].eq("Not flagged by Isolation Forest.").all()
    assert unscored["explanation"].eq("No historical baseline was available.").all()


def test_anomaly_predictions_are_reproducible():
    transactions = create_transactions()
    first = detect_transaction_anomalies(transactions, random_state=17)
    second = detect_transaction_anomalies(transactions, random_state=17)

    pd.testing.assert_series_equal(
        first["predictions"]["anomaly_prediction"],
        second["predictions"]["anomaly_prediction"],
    )
    pd.testing.assert_series_equal(
        first["predictions"]["anomaly_score"],
        second["predictions"]["anomaly_score"],
    )


def test_labels_join_only_for_held_out_evaluation_and_metrics_are_returned():
    transactions = create_transactions()
    result = detect_transaction_anomalies(transactions)
    labels = transactions[["transaction_id", "is_injected_anomaly"]]
    evaluation = evaluate_anomaly_predictions(result["predictions"], labels)

    assert len(evaluation["metrics"]) == 1
    assert set(evaluation["confusion_matrix"].index) == {"actual_normal", "actual_anomaly"}
    assert not evaluation["precision_recall_curve"].empty
    assert set(evaluation["evaluation_rows"]["evaluation_period"]) == {"test"}


def test_invalid_inputs_and_duplicate_transaction_ids_raise_errors():
    transactions = create_transactions()

    with pytest.raises(ValueError, match="unique"):
        build_historical_anomaly_features(pd.concat([transactions, transactions.iloc[[0]]]))

    with pytest.raises(ValueError, match="contamination"):
        detect_transaction_anomalies(transactions, contamination=0.8)

    transactions["amount"] = transactions["amount"].astype(float)
    transactions.loc[0, "amount"] = np.inf
    with pytest.raises(ValueError, match="finite numeric"):
        build_historical_anomaly_features(transactions)
