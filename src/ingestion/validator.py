import pandas as pd

def validate_transactions(df: pd.DataFrame):

    if df.empty:
        raise ValueError("Dataset is empty")

    if not pd.api.types.is_numeric_dtype(df["amount"]):
        raise ValueError("Amount column must be numeric")

    try:
        pd.to_datetime(df["date"])
    except Exception:
        raise ValueError("Invalid date format")

    if df["merchant"].isnull().all():
        raise ValueError("Merchant column is empty")

    return True