import pandas as pd


def enrich_transactions(
    df: pd.DataFrame
):

    df = df.copy()

    df["day_of_week"] = (
        df["date"]
        .dt.day_name()
    )

    df["month"] = (
        df["date"]
        .dt.month
    )

    df["quarter"] = (
        df["date"]
        .dt.quarter
    )

    df["is_weekend"] = (
        df["date"]
        .dt.weekday >= 5
    )

    return df