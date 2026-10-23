import pandas as pd
import logging 
import datetime as dt

logger = logging.getLogger(__name__)

def enrich_data(cleaned_data):

    data = cleaned_data.copy()

    """ Adding date enrichment """


    if "date" not in data.columns:
        raise ValueError("Required column 'date' is missing.")

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.day_name()
    data["is_weekend"] = data["date"].dt.dayofweek >= 5

    report = {

        "input_rows" : len(cleaned_data),
        "output_rows" : len(data),
        "date_features_added" : ["year", "month", "day_of_week", "is_weekend"]
    }

    return {
        "enriched" : data,
        "report" : report
    }