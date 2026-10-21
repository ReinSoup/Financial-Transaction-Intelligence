import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["customer_id", "date", "amount"]

def validate_data(df):
    data = df.copy()

    missing_columns = []

    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.reset_index(drop = True)

    # Store all validation issues and their row positions.
    row_issues = {}
    issue_counts = {}
    rejected_positions = set()
    flagged_positions = set()

    def record_issue(position, issue, reject=False):
        row_issues.setdefault(position, []).append(issue)
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if reject:
            rejected_positions.add(position)
        else:
            flagged_positions.add(position)

    missing_customer = (
        data["customer_id"].isna() | 
        data["customer_id"].astype("string").str.strip().eq("").fillna(False))

    invalid_date = pd.to_datetime(data["date"], errors = "coerce", dayfirst= True, format = "mixed").isna()


    # - - - - - - - - - - - - - - - - - putting these dashes manually btw - - - - - - - - - - - - - - - - - -
    #prepare amounts for numeric validation

    amounts = data["amount"].astype("string").str.strip()
    amounts = amounts.str.replace(",", "", regex = False)
    amounts = amounts.str.replace(r"^(₹|Rs\.?|INR)\s*", "", regex = True, flags = re.IGNORECASE)
    amounts = amounts.str.replace(r"\s*(CR|DR)$", "", regex = True, flags = re.IGNORECASE).str.strip()

    #converting accounting notation from (500) to -500

    amounts = amounts.str.replace(r"^\((.*)\)$", r"-\1", regex = True)

    #convert it back into numbers.

    numeric_amounts = pd.to_numeric(amounts, errors = "coerce")

    #storing any invalid amounts after conversion 

    invalid_amounts = numeric_amounts.isna() | numeric_amounts.isin([float("inf"), float("-inf")])

    # - - - - - - - - - - - - - - - - - - -  that was amount validation  - - - - - - - - - - - - - - - - - - 





    #check missing descriptions if column exists
    if "description" in data.columns:
        missing_descriptions = (
            data["description"].isna()|
            data["description"].astype("string").str.strip().eq("").fillna(False))
    else:
        missing_descriptions = pd.Series(False, index = data.index)







    #check missing categories if column exists
    if "category" in data.columns:
        missing_categories = (
            data["category"].isna() |
            data["category"].astype("string").str.strip().eq("").fillna(False))
    else:
        missing_categories = pd.Series(False, index = data.index)


    

    #check if directions are recognized (credit or debit)
    if "transaction_type" in data.columns:
        directions = (
            data["transaction_type"].astype("string").str.strip().str.lower())

        invalid_direction = ~directions.isin(["debit", "credit", "dr", "cr"])
    else:
        invalid_direction = pd.Series(True, data.index)





    #duplicate row check

    duplicate_rows = data.duplicated(keep = False)





    """ STORE ALL VALIDATION CHECK AND THEIR MASKS IN AN ORGANISED DICTIONARY"""

    issues = {

        "Missing customer_id" : missing_customer, 
        "Missing or Invalid date" : invalid_date,
        "Missing or invalid amount" : invalid_amounts,
        "Missing descriptions" : missing_descriptions,
        "Missing category" : missing_categories,
        "Missing or unrecognised directions" : invalid_direction,
        "Exact duplicate rows" : duplicate_rows
    }



    # Reject rows with missing customer id or invalid dates and amounts

    rejected_rows = missing_customer | invalid_date | invalid_amounts



     # Record each issue using unique row positions.
    for issue, issue_mask in issues.items():
        for position, has_issue in enumerate(issue_mask.to_numpy()):
            if has_issue:
                record_issue(
                    position,
                    issue,
                    reject=bool(rejected_rows.iloc[position]),
                )






    #flagging rows with atleast one validaton issues

    flagged_rows = pd.Series(False, index = data.index)

    for issue_mask in issues.values():
        flagged_rows = flagged_rows | issue_mask





    # Separate accepted and rejected rows using row positions.
    rejected_positions_ordered = []
    accepted_positions = []
    flagged_positions_ordered = []

    for position in range(len(data)):
        if rejected_rows.iloc[position]:
            rejected_positions_ordered.append(position)
        else:
            accepted_positions.append(position)

        if flagged_rows.iloc[position]:
            flagged_positions_ordered.append(position)




    accepted_data = data.iloc[accepted_positions].copy()
    rejected_data = data.iloc[rejected_positions_ordered].copy()
    flagged_data = data.iloc[flagged_positions_ordered].copy()





    accepted_data["validation_issues"] = [
        "; ".join(row_issues.get(position, []))
        for position in accepted_positions]

    rejected_data["validation_issues"] = [
        "; ".join(row_issues.get(position, []))
        for position in rejected_positions_ordered]

    flagged_data["validation_issues"] = [
        "; ".join(row_issues.get(position, []))
        for position in flagged_positions_ordered]



    # Summary

    summary = {
        "input_rows": len(data),
        "accepted_rows": len(accepted_data),
        "rejected_rows": len(rejected_data),
        "flagged_rows": len(flagged_data),
        "missing_optional_columns": [
            column
            for column in [
                "description",
                "category",
                "transaction_type",
            ]
            if column not in data.columns
        ],
        "issue_counts": issue_counts,
    }

    logger.info("Data validation completed: %s", summary)

    return {
        "accepted": accepted_data,
        "rejected": rejected_data,
        "flagged": flagged_data,
        "summary": summary,
}

    


    


    

