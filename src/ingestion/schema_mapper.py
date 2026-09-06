import pandas as pd

CANONICAL_COLUMNS = {

    "date": [
        "date",
        "transaction_date",
        "posted_date",
        "txn_date",
        "trans_date",
        "value_date",
        "timestamp",
        "time",
        "created_at",
        "datetime"
    ],

    "merchant": [
        "merchant",
        "description",
        "narration",
        "details",
        "remarks",
        "payee",
        "beneficiary",
        "transaction_details",
        "vendor",
        "receiver",
        "sender",
        "party",
        "entity"
    ],

    "amount": [
        "amount",
        "transaction_amount",
        "value",
        "txn_amount",
        "debit",
        "credit",
        "withdrawal_amount",
        "deposit_amount",
        "money",
        "sum"
    ],

    "category": [
        "category",
        "type",
        "expense_type",
        "transaction_type",
        "segment",
        "group",
        "label",
        "classification"
    ],

    "payment_method": [
        "payment_method",
        "mode",
        "payment_mode",
        "method",
        "channel",
        "instrument",
        "transaction_mode"
    ],

    "transaction_id": [
        "transaction_id",
        "txn_id",
        "reference",
        "reference_no",
        "utr",
        "rrn",
        "ref_no",
        "id"
    ],

    "balance": [
        "balance",
        "available_balance",
        "running_balance",
        "closing_balance",
        "account_balance"
    ],

    "currency": [
        "currency",
        "currency_code",
        "curr"
    ],

    "account_type": [
        "account_type",
        "acct_type",
        "bank_account_type"
    ],

    "bank_name": [
        "bank",
        "bank_name",
        "institution",
        "provider"
    ],

    "status": [
        "status",
        "transaction_status",
        "payment_status"
    ]  
}

REQUIRED_COLUMNS = ["date", "merchant", "amount"]

def map_columns(df: pd.DataFrame):

    column_mapping = {}

    lower_columns = {
        col.lower().strip(): col
        for col in df.columns
        }
    for canonical, aliases in CANONICAL_COLUMNS.items():

        for alias in aliases:
            if alias.lower() in lower_columns:
                original_col = lower_columns[alias.lower()]
                column_mapping[original_col] = canonical
                break
    
    mapped_df = df.rename(columns=column_mapping)
    missing_required = [
        col for col in REQUIRED_COLUMNS
        if col not in mapped_df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    return mapped_df


