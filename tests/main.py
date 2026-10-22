
from src.ingestion import load_csv
from src.schema_mapper import map_columns
from src.schema_validation import validate_schema
from src.data_validation import validate_data


# Stage 1: Load the raw CSV.
file_path = "data/raw/synthetic_transactions_messy.csv"

df = load_csv(file_path)

if df is None:
    print("Pipeline stopped: CSV ingestion failed.")

else:
    print("Stage 1 complete: CSV loaded.")
    print("Original shape:", df.shape)

    # Stage 2: Map source columns to canonical names.
    mapped_df = map_columns(df)

    if mapped_df is None:
        print("Pipeline stopped: Schema mapping failed.")

    else:
        print("Stage 2 complete: Schema mapped.")
        print("Mapped columns:", mapped_df.columns.tolist())

        # Stage 3: Check required canonical columns.
        schema_valid = validate_schema(mapped_df)

        if not schema_valid:
            print("Pipeline stopped: Schema validation failed.")

        else:
            print("Stage 3 complete: Schema is valid.")

            # Stage 4: Validate transaction rows.
            report = validate_data(mapped_df)

            print("\nData validation summary:")
            print(report["summary"])

            print("\nRejected transactions:")
            print(
                report["rejected"][
                    ["customer_id", "date", "amount", "validation_issues"]
                ].to_string(index=False)
            )

            print("\nFlagged transactions (first 10):")
            print(
                report["flagged"][
                    ["customer_id", "date", "amount", "validation_issues"]
                ].head(10).to_string(index=False)
            )

            print("\nPipeline completed.")
