import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


REQUIRED_TRANSACTION_COLUMNS = [
    "customer_id",
    "amount",
    "category",
    "transaction_type",
    "is_weekend",
]


def build_customer_behavior_features(transactions_df):
    missing_columns = [
        column for column in REQUIRED_TRANSACTION_COLUMNS
        if column not in transactions_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if transactions_df.empty:
        raise ValueError("Transaction data is empty.")

    if transactions_df["customer_id"].isna().any():
        raise ValueError("Customer IDs cannot be missing.")

    data = transactions_df.copy()
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")

    invalid_amounts = data["amount"].isna() | ~np.isfinite(data["amount"])
    if invalid_amounts.any():
        raise ValueError("Transaction amounts must be finite numeric values.")

    data["transaction_type"] = (
        data["transaction_type"].astype("string").str.strip().str.lower()
    )
    data["customer_id"] = data["customer_id"].astype("string").str.strip()
    data["category"] = data["category"].astype("string").str.strip()
    data["is_weekend"] = data["is_weekend"].astype("boolean")

    if (
        data["customer_id"].isna().any()
        or data["customer_id"].eq("").any()
        or data["transaction_type"].isna().any()
        or data["category"].isna().any()
        or data["category"].eq("").any()
    ):
        raise ValueError("Transaction type and category cannot be missing.")

    if data["is_weekend"].isna().any():
        raise ValueError("Weekend values cannot be missing.")

    customers = pd.DataFrame({
        "customer_id": data["customer_id"].drop_duplicates().tolist()
    })
    debit_transactions = data.loc[data["transaction_type"] == "debit"].copy()
    debit_transactions["debit_amount"] = debit_transactions["amount"].abs()

    category_spending = debit_transactions.pivot_table(
        index="customer_id",
        columns="category",
        values="debit_amount",
        aggfunc="sum",
        fill_value=0,
    )
    category_spending.columns = [f"share_{category}" for category in category_spending.columns]

    total_debit_spending = category_spending.sum(axis=1)
    category_shares = category_spending.div(
        total_debit_spending.replace(0, np.nan),
        axis=0,
    ) * 100
    category_shares = category_shares.fillna(0)

    transaction_count = debit_transactions.groupby("customer_id").size().rename(
        "debit_transaction_count"
    )
    average_debit_amount = debit_transactions.groupby("customer_id")["debit_amount"].mean().rename(
        "average_debit_amount"
    )
    weekend_transaction_share = debit_transactions.groupby("customer_id")["is_weekend"].mean().rename(
        "weekend_transaction_share"
    )

    behavior_features = pd.concat(
        [transaction_count, average_debit_amount, weekend_transaction_share],
        axis=1,
    )

    customer_features = customers.merge(
        category_shares.reset_index(),
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    customer_features = customer_features.merge(
        behavior_features.reset_index(),
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    feature_columns = [
        column for column in customer_features.columns
        if column != "customer_id"
    ]
    customer_features[feature_columns] = customer_features[feature_columns].fillna(0)

    return customer_features


def get_behavior_feature_columns(customer_features):
    return [
        column for column in customer_features.columns
        if column.startswith("share_")
    ] + [
        "debit_transaction_count",
        "average_debit_amount",
        "weekend_transaction_share",
    ]


def evaluate_cluster_counts(customer_features, cluster_counts=range(2, 11), random_state=42, n_init=10):
    feature_columns = get_behavior_feature_columns(customer_features)
    missing_columns = [
        column for column in feature_columns
        if column not in customer_features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    if not feature_columns:
        raise ValueError("No behavioral features are available.")

    features = customer_features[feature_columns].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Behavioral features must be finite numeric values.")
    if features.nunique(axis=0).max() < 2:
        raise ValueError("Customer features do not contain distinct profiles to cluster.")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    evaluation_rows = []

    for cluster_count in cluster_counts:
        if cluster_count < 2 or cluster_count >= len(customer_features):
            raise ValueError("Each cluster count must be at least 2 and less than the customer count.")

        model = KMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            n_init=n_init,
        )
        labels = model.fit_predict(features_scaled)
        if len(np.unique(labels)) < 2:
            raise ValueError("The selected data did not produce multiple clusters.")

        evaluation_rows.append({
            "k": cluster_count,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(features_scaled, labels),
        })

    return pd.DataFrame(evaluation_rows)


def segment_customers(customer_features, n_clusters=2, random_state=42, n_init=10):
    if "customer_id" not in customer_features.columns:
        raise ValueError("Customer features must include customer_id.")
    if customer_features["customer_id"].isna().any() or customer_features["customer_id"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")

    feature_columns = get_behavior_feature_columns(customer_features)
    if not feature_columns:
        raise ValueError("No behavioral features are available.")

    features = customer_features[feature_columns].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Behavioral features must be finite numeric values.")
    if features.nunique(axis=0).max() < 2:
        raise ValueError("Customer features do not contain distinct profiles to cluster.")

    if n_clusters < 2 or n_clusters >= len(customer_features):
        raise ValueError("n_clusters must be at least 2 and less than the customer count.")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(features_scaled)
    if len(np.unique(labels)) < 2:
        raise ValueError("The selected data did not produce multiple clusters.")

    assignments = customer_features[["customer_id"]].copy()
    assignments["behavioral_cluster"] = labels
    profile_data = customer_features[feature_columns].copy()
    profile_data["behavioral_cluster"] = labels
    profile = profile_data.groupby("behavioral_cluster").agg(["mean", "median"])
    score = silhouette_score(features_scaled, labels)

    return {
        "assignments": assignments,
        "profile": profile,
        "silhouette_score": score,
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
