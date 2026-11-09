import numpy as np
import pandas as pd
from scipy import stats


def descriptive_statistics(values):
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if numeric_values.empty:
        raise ValueError("At least one valid numeric value is required.")

    if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise ValueError("Values must be finite numbers.")

    quartiles = numeric_values.quantile([0.25, 0.75])
    return pd.DataFrame([{
        "count": int(numeric_values.count()),
        "mean": numeric_values.mean(),
        "median": numeric_values.median(),
        "standard_deviation": numeric_values.std(),
        "minimum": numeric_values.min(),
        "first_quartile": quartiles.loc[0.25],
        "third_quartile": quartiles.loc[0.75],
        "maximum": numeric_values.max(),
        "skewness": numeric_values.skew(),
        "interquartile_range": quartiles.loc[0.75] - quartiles.loc[0.25],
    }])


def iqr_outlier_bounds(values, multiplier=1.5):
    if multiplier <= 0:
        raise ValueError("multiplier must be greater than zero.")

    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric_values.empty:
        raise ValueError("At least one valid numeric value is required.")

    if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise ValueError("Values must be finite numbers.")

    first_quartile = numeric_values.quantile(0.25)
    third_quartile = numeric_values.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_bound = first_quartile - multiplier * interquartile_range
    upper_bound = third_quartile + multiplier * interquartile_range

    return {
        "first_quartile": first_quartile,
        "third_quartile": third_quartile,
        "interquartile_range": interquartile_range,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(((numeric_values < lower_bound) | (numeric_values > upper_bound)).sum()),
    }


def build_customer_debit_summary(transactions_df):
    required_columns = ["customer_id", "amount", "transaction_type"]
    missing_columns = [
        column for column in required_columns
        if column not in transactions_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = transactions_df[required_columns].copy()
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")

    if data["amount"].isna().any() or not np.isfinite(data["amount"]).all():
        raise ValueError("Transaction amounts must be finite numeric values.")
    if data["customer_id"].isna().any():
        raise ValueError("Customer IDs cannot be missing.")

    data["transaction_type"] = (
        data["transaction_type"].astype("string").str.strip().str.lower()
    )
    debit_transactions = data.loc[data["transaction_type"] == "debit"].copy()
    debit_transactions["debit_amount"] = debit_transactions["amount"].abs()

    customer_summary = debit_transactions.groupby("customer_id").agg(
        debit_transaction_count=("debit_amount", "size"),
        total_debit_amount=("debit_amount", "sum"),
        average_debit_amount=("debit_amount", "mean"),
        median_debit_amount=("debit_amount", "median"),
    ).reset_index()

    return customer_summary


def compare_customer_groups(
    customer_summary,
    customer_reference,
    value_column="average_debit_amount",
    group_column="customer_profile",
    first_group="student",
    second_group="salaried_individual",
):
    required_summary_columns = ["customer_id", value_column]
    required_reference_columns = ["customer_id", group_column]
    missing_summary_columns = [
        column for column in required_summary_columns
        if column not in customer_summary.columns
    ]
    missing_reference_columns = [
        column for column in required_reference_columns
        if column not in customer_reference.columns
    ]

    if missing_summary_columns:
        raise ValueError(f"Missing customer summary columns: {missing_summary_columns}")
    if missing_reference_columns:
        raise ValueError(f"Missing customer reference columns: {missing_reference_columns}")

    data = customer_summary[required_summary_columns].merge(
        customer_reference[required_reference_columns],
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )
    first_values = pd.to_numeric(
        data.loc[data[group_column] == first_group, value_column],
        errors="coerce",
    ).dropna()
    second_values = pd.to_numeric(
        data.loc[data[group_column] == second_group, value_column],
        errors="coerce",
    ).dropna()

    if len(first_values) < 2 or len(second_values) < 2:
        raise ValueError("Each customer group must contain at least two valid customers.")

    if not np.isfinite(first_values.to_numpy()).all() or not np.isfinite(second_values.to_numpy()).all():
        raise ValueError("Customer group values must be finite numbers.")

    test_result = stats.ttest_ind(first_values, second_values, equal_var=False)
    first_variance = first_values.var(ddof=1)
    second_variance = second_values.var(ddof=1)
    first_count = len(first_values)
    second_count = len(second_values)
    variance_term_first = first_variance / first_count
    variance_term_second = second_variance / second_count
    degrees_of_freedom = (
        (variance_term_first + variance_term_second) ** 2
        / (
            variance_term_first ** 2 / (first_count - 1)
            + variance_term_second ** 2 / (second_count - 1)
        )
    )
    mean_difference = first_values.mean() - second_values.mean()
    standard_error = np.sqrt(variance_term_first + variance_term_second)
    critical_value = stats.t.ppf(0.975, degrees_of_freedom)
    pooled_standard_deviation = np.sqrt(
        (
            (first_count - 1) * first_variance
            + (second_count - 1) * second_variance
        )
        / (first_count + second_count - 2)
    )
    hedges_correction = 1 - (3 / (4 * (first_count + second_count) - 9))
    effect_size = (
        hedges_correction * mean_difference / pooled_standard_deviation
        if pooled_standard_deviation > 0
        else np.nan
    )
    first_shapiro = stats.shapiro(first_values).pvalue
    second_shapiro = stats.shapiro(second_values).pvalue

    return pd.DataFrame([{
        "unit_of_analysis": "customer",
        "measure": value_column,
        "first_group": first_group,
        "first_group_count": first_count,
        "first_group_mean": first_values.mean(),
        "first_group_median": first_values.median(),
        "first_group_skewness": first_values.skew(),
        "first_group_shapiro_pvalue": first_shapiro,
        "second_group": second_group,
        "second_group_count": second_count,
        "second_group_mean": second_values.mean(),
        "second_group_median": second_values.median(),
        "second_group_skewness": second_values.skew(),
        "second_group_shapiro_pvalue": second_shapiro,
        "mean_difference_first_minus_second": mean_difference,
        "difference_ci_95_lower": mean_difference - critical_value * standard_error,
        "difference_ci_95_upper": mean_difference + critical_value * standard_error,
        "welch_t_statistic": test_result.statistic,
        "welch_degrees_of_freedom": degrees_of_freedom,
        "p_value": test_result.pvalue,
        "hedges_g": effect_size,
    }])
