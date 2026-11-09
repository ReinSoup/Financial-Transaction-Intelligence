import pandas as pd

from src.anomaly_detection import (
    detect_transaction_anomalies,
    evaluate_anomaly_predictions,
)
from src.customer_segmentation import (
    build_customer_behavior_features,
    evaluate_cluster_counts,
    segment_customers,
)
from src.insight_generation import generate_insights
from src.statistical_analysis import (
    build_customer_debit_summary,
    compare_customer_groups,
    descriptive_statistics,
    iqr_outlier_bounds,
)


def run_analytical_pipeline(
    processed_transactions,
    anomaly_labels=None,
    customer_reference=None,
    contamination=0.01,
    random_state=42,
):
    if processed_transactions.empty:
        raise ValueError("Processed transaction data is empty.")

    transaction_rows = processed_transactions.copy()
    customer_features = build_customer_behavior_features(transaction_rows)

    if len(customer_features) < 3:
        raise ValueError("At least three customers are required for cluster evaluation.")

    cluster_counts = range(2, min(11, len(customer_features)))
    cluster_evaluation = evaluate_cluster_counts(
        customer_features,
        cluster_counts=cluster_counts,
        random_state=random_state,
    )
    selected_k = int(
        cluster_evaluation.loc[
            cluster_evaluation["silhouette_score"].idxmax(),
            "k",
        ]
    )
    segmentation = segment_customers(
        customer_features,
        n_clusters=selected_k,
        random_state=random_state,
    )
    customer_segments = customer_features.merge(
        segmentation["assignments"],
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    anomaly_result = detect_transaction_anomalies(
        transaction_rows,
        contamination=contamination,
        random_state=random_state,
    )
    anomaly_evaluation = None
    if anomaly_labels is not None:
        anomaly_evaluation = evaluate_anomaly_predictions(
            anomaly_result["predictions"],
            anomaly_labels,
        )

    debit_amounts = transaction_rows.loc[
        transaction_rows["transaction_type"].astype("string").str.lower() == "debit",
        "amount",
    ].abs()
    customer_debit_summary = build_customer_debit_summary(transaction_rows)
    group_comparison = None
    if customer_reference is not None:
        group_comparison = compare_customer_groups(
            customer_debit_summary,
            customer_reference,
        )

    statistical_findings = {
        "debit_amount_descriptive_statistics": descriptive_statistics(debit_amounts),
        "debit_amount_iqr_bounds": iqr_outlier_bounds(debit_amounts),
        "customer_debit_summary": customer_debit_summary,
        "customer_group_comparison": group_comparison,
    }
    insights = generate_insights(
        customer_features,
        segmentation["assignments"],
        anomaly_result["predictions"],
    )

    return {
        "customer_features": customer_features,
        "customer_segments": customer_segments,
        "cluster_evaluation": cluster_evaluation,
        "cluster_profiles": segmentation["profile"],
        "selected_cluster_count": selected_k,
        "segmentation_model": segmentation["model"],
        "segmentation_scaler": segmentation["scaler"],
        "anomaly_features": anomaly_result["features"],
        "anomaly_predictions": anomaly_result["predictions"],
        "anomaly_models": anomaly_result["models"],
        "anomaly_training_cutoff": anomaly_result["training_cutoff"],
        "anomaly_evaluation": anomaly_evaluation,
        "statistical_findings": statistical_findings,
        "insights": insights,
    }
