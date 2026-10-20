import logging

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["customer_id", "date", "description", "amount", "category", "transaction_type"]

def validate_schema(df):
    missing_columns = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        logger.error("Schema Validation Failed. Missing Columns: %s", missing_columns)
        return False

    logger.info("Schema Validation Successful")
    return True