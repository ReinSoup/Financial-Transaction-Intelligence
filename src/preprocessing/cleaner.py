import pandas as pd

from src.preprocessing.merchant_normalizer import (
    normalize_merchant
)

from src.preprocessing.category_mapper import (
    map_category
)


def clean_transactions(
    df: pd.DataFrame
):

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing critical rows
    df = df.dropna(
        subset=[
            "date",
            "merchant",
            "amount"
        ]
    )

    # Merchant cleanup
    df["merchant"] = (
        df["merchant"]
        .astype(str)
        .str.strip()
    )

    # Normalize merchants
    df["merchant"] = (
        df["merchant"]
        .apply(normalize_merchant)
    )

    # Convert amount
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    # Parse dates
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "date",
            "amount"
        ]
    )

    # Standardized categories
    df["category"] = (
        df["merchant"]
        .apply(map_category)
    )

    return df