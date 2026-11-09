import numpy as np
import pandas as pd


INSIGHT_COLUMNS = [
    "insight_type",
    "customer_id",
    "transaction_id",
    "cluster",
    "category",
    "metric",
    "value",
    "comparison_value",
    "source_explanation",
    "insight_text",
]


def format_insight_text(insight):
    if insight["insight_type"] == "concentrated_category_share":
        return (
            f"{insight['category']} was {insight['value']:.1f}% of this customer's recorded debit spending."
        )

    if insight["insight_type"] == "cluster_transaction_frequency":
        return (
            f"Cluster {insight['cluster']} averaged {insight['value']:.1f} debit transactions per customer; "
            f"the full sample average was {insight['comparison_value']:.1f}."
        )

    if insight["insight_type"] == "transaction_anomaly":
        explanation = insight["source_explanation"]
        if isinstance(explanation, str) and explanation.strip():
            return explanation
        return "Flagged by the Isolation Forest model."

    return ""


def generate_insights(
    customer_features,
    cluster_assignments,
    anomaly_predictions,
    share_threshold=25.0,
):
    if share_threshold < 0 or share_threshold > 100:
        raise ValueError("share_threshold must be between 0 and 100.")

    if "customer_id" not in customer_features.columns:
        raise ValueError("Customer features must include customer_id.")
    if customer_features["customer_id"].isna().any() or customer_features["customer_id"].duplicated().any():
        raise ValueError("Customer feature IDs must be present and unique.")
    if "customer_id" not in cluster_assignments.columns or "behavioral_cluster" not in cluster_assignments.columns:
        raise ValueError("Cluster assignments must include customer_id and behavioral_cluster.")
    if cluster_assignments["customer_id"].isna().any() or cluster_assignments["customer_id"].duplicated().any():
        raise ValueError("Cluster assignment IDs must be present and unique.")
    if "transaction_id" not in anomaly_predictions.columns or "anomaly_flag" not in anomaly_predictions.columns:
        raise ValueError("Anomaly predictions must include transaction_id and anomaly_flag.")
    if anomaly_predictions["transaction_id"].isna().any() or anomaly_predictions["transaction_id"].duplicated().any():
        raise ValueError("Anomaly prediction IDs must be present and unique.")

    share_columns = [
        column for column in customer_features.columns
        if column.startswith("share_")
    ]
    insights = []

    if share_columns:
        customer_shares = customer_features[["customer_id"] + share_columns].copy()
        customer_shares[share_columns] = customer_shares[share_columns].astype(float)
        if not np.isfinite(customer_shares[share_columns].to_numpy()).all():
            raise ValueError("Category shares must be finite numeric values.")

        for _, row in customer_shares.iterrows():
            top_column = row[share_columns].idxmax()
            top_share = float(row[top_column])
            if top_share < share_threshold:
                continue

            category = top_column.replace("share_", "", 1)
            insights.append({
                "insight_type": "concentrated_category_share",
                "customer_id": row["customer_id"],
                "transaction_id": None,
                "cluster": None,
                "category": category,
                "metric": "category_spending_share_percent",
                "value": top_share,
                "comparison_value": None,
                "source_explanation": None,
            })

    feature_lookup = customer_features.set_index("customer_id")
    assignments = cluster_assignments[["customer_id", "behavioral_cluster"]].copy()
    assignments = assignments.drop_duplicates("customer_id").set_index("customer_id")
    cluster_data = assignments.join(feature_lookup, how="inner", validate="one_to_one")

    if "debit_transaction_count" in cluster_data.columns and not cluster_data.empty:
        overall_average_count = cluster_data["debit_transaction_count"].mean()
        cluster_average_count = cluster_data.groupby("behavioral_cluster")[
            "debit_transaction_count"
        ].mean()

        for cluster, average_count in cluster_average_count.items():
            insights.append({
                "insight_type": "cluster_transaction_frequency",
                "customer_id": None,
                "transaction_id": None,
                "cluster": cluster,
                "category": None,
                "metric": "mean_debit_transaction_count",
                "value": float(average_count),
                "comparison_value": float(overall_average_count),
                "source_explanation": None,
            })

    anomaly_columns = ["transaction_id", "anomaly_flag"]
    for column in ["customer_id", "category", "explanation"]:
        if column in anomaly_predictions.columns:
            anomaly_columns.append(column)
    flagged_transactions = anomaly_predictions.loc[
        anomaly_predictions["anomaly_flag"] == "anomaly",
        anomaly_columns,
    ]

    for _, row in flagged_transactions.iterrows():
        explanation = row.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "Flagged by the Isolation Forest model."
        insights.append({
            "insight_type": "transaction_anomaly",
            "customer_id": row.get("customer_id"),
            "transaction_id": row["transaction_id"],
            "cluster": None,
            "category": row.get("category"),
            "metric": "anomaly_flag",
            "value": 1,
            "comparison_value": None,
            "source_explanation": explanation,
        })

    insight_results = pd.DataFrame(insights, columns=INSIGHT_COLUMNS)
    insight_results["insight_text"] = insight_results.apply(format_insight_text, axis=1)

    return insight_results
