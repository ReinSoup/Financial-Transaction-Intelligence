import os 
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_csv(file_path):

    if not os.path.isfile(file_path):
        logger.error("Error: File not found %s", file_path)
        return None 
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty %s", file_path)
        return None
    except pd.errors.ParserError as error:
        logger.error("Could not Parse CSV file: %s ", error)
        return None 
    except UnicodeDecodeError as error:
        logger.error("Could not decode CSV file: %s", error)
        return None 
    except OSError as error:
        logger.error("Could not read CSV file: %s", error)
        return None 

    if df.empty:
        logger.error("The CSV contains no data rows %s", file_path)
        return None

    logger.info("CSV Loaded successfully. Rows: %s Columns: %s", len(df), len(df.columns))

    return df
