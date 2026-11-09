from pathlib import Path

import pandas as pd

from src.analytics_pipeline import run_analytical_pipeline


repository_root = Path(__file__).resolve().parent
transactions = pd.read_csv(repository_root / "data/processed/enriched_transactions.csv")
anomaly_labels = pd.read_csv(repository_root / "data/reference/synthetic_anomaly_labels.csv")
customer_reference = pd.read_csv(repository_root / "data/reference/synthetic_customers.csv")

analysis = run_analytical_pipeline(
    transactions,
    anomaly_labels=anomaly_labels,
    customer_reference=customer_reference,
    contamination=0.01,
    random_state=42,
)

print("Customer feature rows:", len(analysis["customer_features"]))
print("Behavioral clusters:", analysis["selected_cluster_count"])
print("Cluster evaluation:")
print(analysis["cluster_evaluation"].round(4).to_string(index=False))
print("\nTransaction prediction rows:", len(analysis["anomaly_predictions"]))
print("Anomaly evaluation:")
print(analysis["anomaly_evaluation"]["metrics"].round(4).to_string(index=False))
print("Confusion matrix:")
print(analysis["anomaly_evaluation"]["confusion_matrix"].to_string())
print("\nCustomer group comparison:")
customer_group_comparison = analysis["statistical_findings"]["customer_group_comparison"].round(4)
customer_group_comparison["p_value"] = analysis["statistical_findings"]["customer_group_comparison"]["p_value"].map(
    lambda value: f"{value:.3e}"
)
print(customer_group_comparison.to_string(index=False))
print("\nInsight types:")
print(analysis["insights"]["insight_type"].value_counts().to_string())
