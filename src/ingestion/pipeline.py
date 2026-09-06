from src.ingestion.file_detector import (
    detect_file_type
)

from src.ingestion.loader import (
    load_file
)

from src.preprocessing.amount_parser import (
    parse_amount
)


from src.ingestion.schema_mapper import (
    map_columns
)

from src.ingestion.validator import (
    validate_transactions
)

from src.preprocessing.cleaner import (
    clean_transactions
)

from src.enrichment.transaction_enricher import (
    enrich_transactions
)

from src.utils.logger import logger

from src.utils.config import (
    PROCESSED_DATA_DIR
)


from src.ingestion.file_detector import (
    detect_file_type
)

from src.ingestion.loader import (
    load_file
)

from src.ingestion.schema_mapper import (
    map_columns
)

from src.ingestion.validator import (
    validate_transactions
)

from src.preprocessing.cleaner import (
    clean_transactions
)

from src.preprocessing.amount_parser import (
    parse_amount
)

from src.enrichment.transaction_enricher import (
    enrich_transactions
)

from src.utils.logger import logger

from src.utils.config import (
    PROCESSED_DATA_DIR
)


def run_ingestion_pipeline(
    file_path: str
):

    try:

        logger.info(
            "Detecting file type..."
        )

        file_type = detect_file_type(
            file_path
        )

        logger.info(
            "Loading file..."
        )

        df = load_file(
            file_path,
            file_type
        )

        logger.info(
            "Mapping schema..."
        )

        df = map_columns(df)

        logger.info(
            "Parsing amounts..."
        )

        df["amount"] = df["amount"].apply(
            parse_amount
        )

        logger.info(
            "Validating transactions..."
        )

        validate_transactions(df)

        logger.info(
            "Cleaning transactions..."
        )

        df = clean_transactions(df)

        logger.info(
            "Enriching transactions..."
        )

        df = enrich_transactions(df)

        output_path = (
            f"{PROCESSED_DATA_DIR}/"
            "clean_transactions.parquet"
        )

        df.to_parquet(
            output_path,
            index=False
        )

        logger.info(
            f"Processed data saved to "
            f"{output_path}"
        )

        logger.info(
            "Ingestion completed successfully."
        )

        return df

    except Exception as e:

        logger.error(
            f"Ingestion pipeline failed: {e}"
        )

        raise