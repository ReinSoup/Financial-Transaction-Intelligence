import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


REQUIRED_TRANSACTION_COLUMNS = [
    "transaction_id",
    "customer_id",
    "date",
    "amount",
    "category",
    "transaction_type",
]

ANOMALY_FEATURE_COLUMNS = [
    "log_amount_magnitude",
    "log_customer_amount_ratio",
    "log_category_amount_ratio",
]


def _prior_median(values, minimum_history):
    return values.shift(1).expanding(min_periods=minimum_history).median()


def _ratio_text(value):
    if pd.isna(value):
        return "unavailable"
    return f"{value:.2f}"


def build_historical_anomaly_features(transactions_df, minimum_history=5):
    missing_columns = [
        column for column in REQUIRED_TRANSACTION_COLUMNS
        if column not in transactions_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if transactions_df.empty:
        raise ValueError("Transaction data is empty.")

    if minimum_history < 1:
        raise ValueError("minimum_history must be at least 1.")

    data = transactions_df[REQUIRED_TRANSACTION_COLUMNS].copy()

    if data["transaction_id"].isna().any() or data["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be present and unique.")

    if data[["customer_id", "category", "transaction_type"]].isna().any().any():
        raise ValueError("Customer, category, and transaction type values cannot be missing.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Transaction dates must be valid.")

    if data["amount"].isna().any() or not np.isfinite(data["amount"]).all():
        raise ValueError("Transaction amounts must be finite numeric values.")

    data["date"] = data["date"].dt.normalize()
    data["transaction_type"] = (
        data["transaction_type"].astype("string").str.strip().str.lower()
    )
    data["category"] = data["category"].astype("string").str.strip()
    data["customer_id"] = data["customer_id"].astype("string").str.strip()
    data["amount_magnitude"] = data["amount"].abs()

    if data["customer_id"].eq("").any() or data["category"].eq("").any():
        raise ValueError("Customer IDs and categories cannot be blank.")
    if not data["transaction_type"].isin(["debit", "credit"]).all():
        raise ValueError("Transaction types must be debit or credit.")

    customer_keys = ["customer_id", "transaction_type", "category"]
    daily_customer = data.groupby(
        customer_keys + ["date"],
        as_index=False,
        sort=True,
    )["amount_magnitude"].median()
    daily_customer = daily_customer.sort_values(customer_keys + ["date"])
    daily_customer["customer_prior_median"] = daily_customer.groupby(
        customer_keys,
        sort=False,
    )["amount_magnitude"].transform(
        lambda values: _prior_median(values, minimum_history)
    )

    category_keys = ["transaction_type", "category"]
    daily_category = data.groupby(
        category_keys + ["date"],
        as_index=False,
        sort=True,
    )["amount_magnitude"].median()
    daily_category = daily_category.sort_values(category_keys + ["date"])
    daily_category["category_prior_median"] = daily_category.groupby(
        category_keys,
        sort=False,
    )["amount_magnitude"].transform(
        lambda values: _prior_median(values, minimum_history)
    )

    features = data.merge(
        daily_customer[customer_keys + ["date", "customer_prior_median"]],
        on=customer_keys + ["date"],
        how="left",
        validate="many_to_one",
    )
    features = features.merge(
        daily_category[category_keys + ["date", "category_prior_median"]],
        on=category_keys + ["date"],
        how="left",
        validate="many_to_one",
    )

    features["reference_amount"] = features["customer_prior_median"].fillna(
        features["category_prior_median"]
    )
    customer_denominator = features["customer_prior_median"].replace(0, np.nan)
    category_denominator = features["category_prior_median"].replace(0, np.nan)
    reference_denominator = features["reference_amount"].replace(0, np.nan)

    features["customer_amount_ratio"] = features["amount_magnitude"] / customer_denominator
    features["category_amount_ratio"] = features["amount_magnitude"] / category_denominator
    features["historical_amount_ratio"] = features["amount_magnitude"] / reference_denominator
    features["log_amount_magnitude"] = np.log1p(features["amount_magnitude"])
    customer_ratio_for_model = features["customer_amount_ratio"].fillna(
        features["category_amount_ratio"]
    )
    category_ratio_for_model = features["category_amount_ratio"].fillna(
        features["customer_amount_ratio"]
    )
    features["log_customer_amount_ratio"] = np.log1p(customer_ratio_for_model)
    features["log_category_amount_ratio"] = np.log1p(category_ratio_for_model)
    features["baseline_source"] = np.where(
        features["customer_prior_median"].notna(),
        "customer_category_history",
        "category_history",
    )
    features["is_eligible"] = np.isfinite(
        features[ANOMALY_FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)

    return features


def _training_cutoff(dates, training_fraction):
    if training_fraction <= 0 or training_fraction >= 1:
        raise ValueError("training_fraction must be greater than 0 and less than 1.")

    unique_dates = pd.Index(dates.drop_duplicates().sort_values())
    if len(unique_dates) < 2:
        raise ValueError("At least two transaction dates are required for a temporal split.")

    cutoff_index = int(len(unique_dates) * training_fraction) - 1
    cutoff_index = min(max(cutoff_index, 0), len(unique_dates) - 2)
    return unique_dates[cutoff_index]


def detect_transaction_anomalies(
    transactions_df,
    contamination=0.01,
    random_state=42,
    minimum_history=5,
    training_fraction=0.8,
    n_estimators=200,
):
    if contamination <= 0 or contamination > 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")

    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    features = build_historical_anomaly_features(
        transactions_df,
        minimum_history=minimum_history,
    )
    cutoff = _training_cutoff(features["date"], training_fraction)
    eligible = features.loc[features["is_eligible"]].copy()
    predictions = features[
        [
            "transaction_id",
            "customer_id",
            "date",
            "category",
            "amount",
            "transaction_type",
            "customer_prior_median",
            "category_prior_median",
            "reference_amount",
            "customer_amount_ratio",
            "category_amount_ratio",
            "historical_amount_ratio",
            "baseline_source",
            "is_eligible",
        ]
    ].copy()
    predictions["anomaly_prediction"] = pd.Series(pd.NA, index=predictions.index, dtype="Int64")
    predictions["anomaly_score"] = np.nan
    predictions["evaluation_period"] = np.where(
        predictions["date"] > cutoff,
        "test",
        "train",
    )
    models = {}

    for transaction_type in sorted(eligible["transaction_type"].unique()):
        type_rows = eligible["transaction_type"] == transaction_type
        training_rows = type_rows & (eligible["date"] <= cutoff)
        if training_rows.sum() < 2:
            raise ValueError(
                f"Not enough historical training rows for {transaction_type} transactions."
            )

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(eligible.loc[training_rows, ANOMALY_FEATURE_COLUMNS])
        models[transaction_type] = model

        prediction_rows = eligible.index[type_rows]
        predictions.loc[prediction_rows, "anomaly_prediction"] = model.predict(
            eligible.loc[prediction_rows, ANOMALY_FEATURE_COLUMNS]
        )
        predictions.loc[prediction_rows, "anomaly_score"] = model.decision_function(
            eligible.loc[prediction_rows, ANOMALY_FEATURE_COLUMNS]
        )

    predictions["anomaly_flag"] = predictions["anomaly_prediction"].map(
        {1: "normal", -1: "anomaly"}
    )
    predictions["explanation"] = "No historical baseline was available."
    predictions.loc[
        predictions["anomaly_prediction"].notna(),
        "explanation",
    ] = "Not flagged by Isolation Forest."
    flagged = predictions["anomaly_prediction"] == -1
    predictions.loc[flagged, "explanation"] = predictions.loc[flagged].apply(
        lambda row: (
            f"Flagged by Isolation Forest; amount ₹{abs(row['amount']):.2f}; "
            f"customer-category ratio {_ratio_text(row['customer_amount_ratio'])}; "
            f"category ratio {_ratio_text(row['category_amount_ratio'])}; "
            f"reference source {row['baseline_source']}."
        ),
        axis=1,
    )
    predictions = predictions.sort_values("date").reset_index(drop=True)

    return {
        "predictions": predictions,
        "features": features,
        "models": models,
        "training_cutoff": cutoff,
        "feature_columns": ANOMALY_FEATURE_COLUMNS.copy(),
        "contamination": contamination,
    }


def evaluate_anomaly_predictions(predictions, labels, label_column="is_injected_anomaly"):
    required_prediction_columns = [
        "transaction_id",
        "anomaly_prediction",
        "anomaly_score",
        "evaluation_period",
    ]
    required_label_columns = ["transaction_id", label_column]

    missing_prediction_columns = [
        column for column in required_prediction_columns
        if column not in predictions.columns
    ]
    missing_label_columns = [
        column for column in required_label_columns
        if column not in labels.columns
    ]

    if missing_prediction_columns:
        raise ValueError(f"Missing prediction columns: {missing_prediction_columns}")
    if missing_label_columns:
        raise ValueError(f"Missing label columns: {missing_label_columns}")

    if labels["transaction_id"].duplicated().any():
        raise ValueError("Label transaction IDs must be unique.")
    if not labels[label_column].isin([0, 1]).all():
        raise ValueError("Anomaly labels must contain only 0 and 1.")

    evaluation_rows = predictions.loc[
        (predictions["evaluation_period"] == "test")
        & predictions["anomaly_prediction"].notna()
    ].merge(
        labels[["transaction_id", label_column]],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    if evaluation_rows.empty:
        raise ValueError("No eligible held-out predictions matched the labels.")

    y_true = evaluation_rows[label_column].astype(int)
    y_predicted = evaluation_rows["anomaly_prediction"].astype(int).eq(-1).astype(int)
    anomaly_strength = -evaluation_rows["anomaly_score"].astype(float)
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_strength)

    metrics = pd.DataFrame([{
        "evaluated_transactions": len(evaluation_rows),
        "actual_anomalies": int(y_true.sum()),
        "predicted_anomalies": int(y_predicted.sum()),
        "precision": precision_score(y_true, y_predicted, zero_division=0),
        "recall": recall_score(y_true, y_predicted, zero_division=0),
        "f1_score": f1_score(y_true, y_predicted, zero_division=0),
        "average_precision": average_precision_score(y_true, anomaly_strength),
    }])
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_predicted, labels=[0, 1]),
        index=["actual_normal", "actual_anomaly"],
        columns=["predicted_normal", "predicted_anomaly"],
    )
    precision_recall = pd.DataFrame({
        "precision": precision[:-1] if len(thresholds) else precision,
        "recall": recall[:-1] if len(thresholds) else recall,
        "threshold": thresholds if len(thresholds) else np.nan,
    })

    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
        "precision_recall_curve": precision_recall,
        "evaluation_rows": evaluation_rows,
    }
