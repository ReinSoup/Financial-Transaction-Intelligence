import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


COLUMN_ALIASES = {
    "transaction_id": [
        "transaction_id", "transactionid", "transaction id",
        "txn_id", "txnid", "txn id", "transaction_no", "transaction_number",
        "txn_no", "txn_number", "reference_id", "reference_no",
        "reference_number", "ref_id", "payment_id", "transfer_id",
        "entry_id", "record_id", "transaction_reference", "transaction_ref",
        "bank_reference", "transaction_code",
    ],
    "customer_id": [
        "customer_id", "customerid", "customer id", "cust_id", "custid",
        "cust_no", "cust_number", "client_id", "clientid", "user_id",
        "userid", "user id", "member_id", "party_id", "account_holder_id",
        "customer", "client", "user",
    ],
    "date": [
        "date", "transaction_date", "transactiondate", "txn_date", "txndate",
        "txn_dt", "date_of_transaction", "posted_date", "posting_date",
        "value_date", "booking_date", "settlement_date", "effective_date",
        "transaction_timestamp", "timestamp", "datetime",
        "transaction_date_dd_mm_yyyy",
    ],
    "description": [
        "description", "transaction_description", "narration",
        "transaction_details", "details", "memo", "note", "notes",
        "particulars", "narrative", "remarks", "payment_details",
        "payment_description", "purpose", "comments", "reference_text",
        "bank_description",
    ],
    "amount": [
        "amount", "transaction_amount", "transactionamount", "txn_amount",
        "txnamount", "amt", "transaction_value", "value", "net_amount",
        "amount_inr", "amount_rs", "amount_rupees",
        "transaction_amount_inr", "transaction_amount_rs",
    ],
    "category": [
        "category", "transaction_category", "transactioncategory",
        "txn_category", "category_name", "expense_category",
        "spending_category", "expense_type", "spend_category",
    ],
    "transaction_type": [
        "transaction_type", "transactiontype", "txn_type", "type",
        "direction", "transaction_direction", "txn_direction", "dr_cr",
        "debit_credit", "credit_debit", "transaction_kind", "flow_type",
        "entry_type",
    ],
    "merchant": [
        "merchant", "merchant_name", "merchantname", "payee", "payee_name",
        "beneficiary", "counterparty", "counterparty_name", "recipient",
        "vendor", "store", "business_name", "shop_name",
    ],
    "currency": [
        "currency", "currency_code", "currencycode", "ccy",
        "currency_type", "currency_name",
    ],
    "customer_profile": [
        "customer_profile", "profile", "profile_type", "customer_type",
        "occupation_profile",
    ],
    "is_injected_anomaly": [
        "is_injected_anomaly", "injected_anomaly", "anomaly_label",
    ],
}


def normalize_column_name(column_name):
    """Normalize common header punctuation and spacing before alias matching."""
    normalized = str(column_name).strip().casefold()
    normalized = re.sub(r"[\s./\\-]+", "_", normalized)
    normalized = re.sub(r"[^\w]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def map_columns(df):
    """Return a copy with recognized headers mapped to canonical names.

    Returns None when multiple input columns map to one canonical field.
    """
    mapped_df = df.copy()
    normalized_columns = [normalize_column_name(column) for column in mapped_df.columns]

    if len(normalized_columns) != len(set(normalized_columns)):
        logger.error("Mapping failed: duplicate column names after normalization.")
        return None

    alias_lookup = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            existing = alias_lookup.get(normalized_alias)
            if existing is not None and existing != canonical_name:
                raise ValueError(
                    f"Alias '{alias}' is assigned to both '{existing}' and '{canonical_name}'."
                )
            alias_lookup[normalized_alias] = canonical_name

    mapped_columns = [alias_lookup.get(column, column) for column in normalized_columns]
    if len(mapped_columns) != len(set(mapped_columns)):
        logger.error("Mapping failed: multiple source columns map to the same canonical column.")
        return None

    mapped_df.columns = mapped_columns
    logger.info("Schema mapping completed successfully.")
    return mapped_df