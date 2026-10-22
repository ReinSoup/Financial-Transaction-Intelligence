import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)


def preprocess_data(accepted_data):

    """Clean and Standardize accepted transaction data"""

    cleaned_data = accepted_data.copy()

    required_columns = ["customer_id", "date", "amount", "transaction_type"]

    missing_columns = []

    for column in required_columns:
        if column not in cleaned_data.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Missing required columns : {missing_columns}")

    #storing number of rows
    input_rows = len(cleaned_data)

    """Normalizing Transaction directions (credit or debit)"""

    directions = (cleaned_data["transaction_type"].astype("string").str.strip().str.lower())

    directions = directions.replace({

        "dr" : "debit",
        "cr" : "credit"
    })

    invalid_directions = ~directions.isin(["debit", "credit"])

    if invalid_directions.any():
        raise ValueError(f"Preprocessing Found unrecognized transaction directions")

    # apply changes to the variable

    cleaned_data["transaction_type"] = directions



    """Normalize Transaction amounts"""

    amounts = cleaned_data["amount"].astype("string").str.strip()

    amounts = amounts.str.replace(",", "", regex = False)

    amounts = amounts.str.replace(r"^(₹|Rs\.?|INR)\s*","", regex = True, flags = re.IGNORECASE)

    # remove cr / dr suffixes

    amounts = amounts.str.replace(r"\s*(CR|DR)$","",regex=True,flags=re.IGNORECASE,).str.strip()

    # convert accounting notation (500) to -500

    amounts = amounts.str.replace(r"^\((.*)\)$",r"-\1",regex=True)


    # back to numbers [inf = infinity]

    numeric_amounts = pd.to_numeric(amounts, errors = "coerce")

    invalid_amounts = (numeric_amounts.isna() | numeric_amounts.isin([float("inf"), float("-inf")]))

    if invalid_amounts.any():
        raise ValueError("Preprocessing Found invalid amounts in accepted data")

    # Design decision : numeric amounts CR = positive, DR = negative
    # abs (absolute value) to prevent already negative values turning positive
    numeric_amounts = numeric_amounts.abs()

    debit_rows = directions.eq("debit")
    

    numeric_amounts.loc[debit_rows] = (-numeric_amounts.loc[debit_rows])

    cleaned_data["amount"] = numeric_amounts




    """Normalize transaction dates"""
    

    normalized_dates = pd.to_datetime(cleaned_data["date"],errors = "coerce", dayfirst=True, format = "mixed")

    invalid_dates = normalized_dates.isna()

    if invalid_dates.any():
        raise ValueError("Preprocessing found invalid dates in accepted data")

    cleaned_data["date"] = normalized_dates




    """ Normalize text columns """

    text_values_normalized = 0

    text_columns = ["description", "merchant", "category"]

    for column in text_columns:
        if column in cleaned_data.columns:

            original_text = cleaned_data[column].astype("string")

            cleaned_text = (original_text.str.strip().str.replace(r"\s+", " ", regex=True).replace("", pd.NA))

            text_values_normalized += int(original_text.ne(cleaned_text).fillna(False).sum())

            cleaned_data[column] = cleaned_text



    """ Counting duplicates without removal"""

    duplicate_rows = cleaned_data.duplicated(keep = False)

    duplicate_rows_kept = int(duplicate_rows.sum())


    """Reports"""

    processing_report = {

        "input_rows" : input_rows,
        "output_rows" : len(cleaned_data),
        "amount_values_processed" : len(numeric_amounts),
        "date_values_processed" : len(normalized_dates),
        "text_values_normalized" : text_values_normalized,
        "duplicate_rows_preserved" : duplicate_rows_kept
    }

    logger.info("Preprocessing completed: %s",processing_report)

    return {
        "cleaned": cleaned_data,
        "report": processing_report
        }

                       