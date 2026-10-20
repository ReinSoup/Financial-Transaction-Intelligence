import logging 
import pandas as pd

logger = logging.getLogger(__name__)


COLUMN_ALIASES = {
    "transaction_id": ["transaction_id","txn_id","reference_id"],

    "customer_id": ["customer_id","user_id"],

    "date": ["date","transaction_date","txn_date"],

    "description": ["description","narration","transaction_details","details"],

    "amount": ["amount","transaction_amount","value"],

    "category": ["category","transaction_category"],

    "transaction_type": ["transaction_type","type","direction"],

    "merchant": ["merchant","merchant_name","payee"],

    "currency": ["currency","currency_code"],
}


def normalize_column_name(column_name): # small helper function. takes one col and returns standardized output
    return column_name.strip().lower()

def map_columns(df):
    """Map recognized source column names to canonical names
    
    Returns : pd.DataFrame: a copy with mapped column names,
    None if a mapping conflict is detected"""
# create a copy so the original df stays unchanged
    mapped_df = df.copy()
# normalize the source column names
    normalized_columns = []

    for column in mapped_df.columns:
        normalize_it = normalize_column_name(column)
        normalized_columns.append(normalize_it)

    #detecting headers that become duplicates after normalization

    if len(normalized_columns) != len(set(normalized_columns)):
        # a python set removes duplicates and only keeps unique cols
        logger.error("Mapping Failed: duplicate column names after normalization")
        return None

    alias_lookup = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            alias_lookup[normalized_alias] = canonical_name

    mapped_columns = []
    #this list will hold final names to assign to dataframe

    for column in normalized_columns:
        if column in alias_lookup:
            canonical_name = alias_lookup[column]
            mapped_columns.append(canonical_name)
        else:
            mapped_columns.append(column)

    if len(mapped_columns) != len(set(mapped_columns)):
        logger.error("Mapping failed: multiple source columns map to same canonical column.")
        return None

    mapped_df.columns = mapped_columns

    logger.info("Schema mapping completed successfullly")

    return mapped_df
    




