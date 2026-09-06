import pandas as pd
import re


def parse_amount(amount):

    if pd.isna(amount):
        return None

    amount = str(amount).strip()

    # remove currency symbols
    amount = re.sub(r"[₹$,]", "", amount)

    # handle parentheses negatives
    if amount.startswith("(") and amount.endswith(")"):
        amount = "-" + amount[1:-1]

    return float(amount)