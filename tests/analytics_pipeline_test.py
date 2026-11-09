import pandas as pd

from src.analytics_pipeline import run_analytical_pipeline


def create_transactions():
    rows = []
    transaction_number = 1
    for day in range(20):
        for customer_number in range(8):
            customer_id = f"C{customer_number:02d}"
            debit_amount = -100 - customer_number * 30
            category = "Housing" if customer_number < 4 else "Food"
            rows.append({
                "transaction_id": f"T{transaction_number}",
                "customer_id": customer_id,
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                "amount": debit_amount,
                "category": category,
                "transaction_type": "debit",
                "is_weekend": day % 7 in [5, 6],
                "is_injected_anomaly": int(day == 19 and customer_number == 0),
            })
            transaction_number += 1
            rows.append({
                "transaction_id": f"T{transaction_number}",
                "customer_id": customer_id,
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                "amount": 2000 + customer_number * 100,
                "category": "Income",
                "transaction_type": "credit",
                "is_weekend": day % 7 in [5, 6],
                "is_injected_anomaly": 0,
            })
            transaction_number += 1
    return pd.DataFrame(rows)


def test_pipeline_returns_separate_customer_and_transaction_outputs():
    transactions = create_transactions()
    labels = transactions[["transaction_id", "is_injected_anomaly"]]
    customer_reference = pd.DataFrame({
        "customer_id": [f"C{number:02d}" for number in range(8)],
        "customer_profile": ["student"] * 4 + ["salaried_individual"] * 4,
    })

    result = run_analytical_pipeline(
        transactions,
        anomaly_labels=labels,
        customer_reference=customer_reference,
    )

    assert result["customer_features"]["customer_id"].nunique() == 8
    assert result["customer_segments"]["customer_id"].nunique() == 8
    assert result["anomaly_predictions"]["transaction_id"].is_unique
    assert len(result["anomaly_predictions"]) == len(transactions)
    assert result["anomaly_evaluation"]["metrics"].loc[0, "evaluated_transactions"] > 0
    assert len(result["statistical_findings"]["customer_group_comparison"]) == 1
    assert not result["insights"].empty
