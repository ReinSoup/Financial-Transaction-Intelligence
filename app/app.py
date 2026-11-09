"""Streamlit dashboard for Financial Transaction Intelligence.
    (message from Ryan: this section of the project has been written using AI guidance.)
Run from the repository root with ``streamlit run app/app.py``.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics_pipeline import run_analytical_pipeline
from src.data_enrichment import enrich_data
from src.data_preprocessing import preprocess_data
from src.data_validation import validate_data
from src.schema_mapper import map_columns


st.set_page_config(
    page_title="Financial Transaction Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def load_and_prepare(upload: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map, validate, clean, and enrich one uploaded CSV in memory."""
    raw = pd.read_csv(io.BytesIO(upload.getvalue()))
    mapped = map_columns(raw)
    if mapped is None:
        raise ValueError(
            "Column mapping is ambiguous. Make sure the file does not contain "
            "two headers that map to the same field."
        )

    # The current analytics pipeline needs these fields for customer features
    # and historical anomaly baselines; the transaction ID must be stable.
    required = {
        "transaction_id",
        "customer_id",
        "date",
        "amount",
        "category",
        "transaction_type",
    }
    missing = sorted(required.difference(mapped.columns))
    if missing:
        raise ValueError(
            "Required columns could not be identified: " + ", ".join(missing)
        )

    validation = validate_data(mapped)
    if validation["accepted"].empty:
        raise ValueError("No rows passed validation. Review the input columns and values.")

    cleaned_result = preprocess_data(validation["accepted"])
    enriched_result = enrich_data(cleaned_result["cleaned"])
    enriched = enriched_result["enriched"]
    summary = {
        "raw_rows": len(raw),
        "mapped_columns": list(mapped.columns),
        "validation": validation["summary"],
        "preprocessing": cleaned_result["report"],
        "enrichment": enriched_result["report"],
    }
    return enriched, summary


def run_analysis(transactions: pd.DataFrame, contamination: float) -> dict[str, Any]:
    return run_analytical_pipeline(transactions, contamination=contamination)


def show_quality(summary: dict[str, Any], transactions: pd.DataFrame) -> None:
    validation = summary["validation"]
    cols = st.columns(4)
    cols[0].metric("Rows received", f"{summary['raw_rows']:,}")
    cols[1].metric("Rows accepted", f"{validation['accepted_rows']:,}")
    cols[2].metric("Rows rejected", f"{validation['rejected_rows']:,}")
    cols[3].metric("Rows flagged for review", f"{validation['flagged_rows']:,}")

    with st.expander("Data quality details", expanded=False):
        st.write("**Recognized columns**", ", ".join(summary["mapped_columns"]))
        st.write("**Optional fields not provided**", ", ".join(validation["missing_optional_columns"]) or "None")
        issue_counts = validation["issue_counts"]
        if issue_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"Issue": issue, "Rows affected": count} for issue, count in issue_counts.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.caption("Rows with non-critical issues are retained and counted for review. Exact duplicates are reported, not silently removed.")
        st.dataframe(transactions.head(100), hide_index=True, use_container_width=True)


def show_overview(transactions: pd.DataFrame) -> None:
    debit = transactions.loc[transactions["transaction_type"].eq("debit"), "amount"].abs()
    credit = transactions.loc[transactions["transaction_type"].eq("credit"), "amount"]
    cols = st.columns(4)
    cols[0].metric("Transactions", f"{len(transactions):,}")
    cols[1].metric("Customers", f"{transactions['customer_id'].nunique():,}")
    cols[2].metric("Recorded outflow", money(float(debit.sum())))
    cols[3].metric("Recorded inflow", money(float(credit.sum())))

    left, right = st.columns(2)
    with left:
        st.subheader("Debit spending by category")
        if "category" in transactions:
            category = transactions.loc[transactions["transaction_type"].eq("debit")].copy()
            category["spending"] = category["amount"].abs()
            totals = category.groupby("category", dropna=False, as_index=False)["spending"].sum().sort_values("spending", ascending=False)
            if not totals.empty:
                st.plotly_chart(px.bar(totals, x="spending", y="category", orientation="h", labels={"spending": "Debit amount (₹)", "category": "Category"}), use_container_width=True)
            else:
                st.info("No debit categories are available in this upload.")
        else:
            st.info("Add a category column to explore category spending.")
    with right:
        st.subheader("Monthly recorded cash flow")
        monthly = transactions.assign(
            inflow=transactions["amount"].where(transactions["transaction_type"].eq("credit"), 0),
            outflow=-transactions["amount"].where(transactions["transaction_type"].eq("debit"), 0),
        ).groupby(transactions["date"].dt.to_period("M").astype(str), as_index=False)[["inflow", "outflow"]].sum()
        monthly = monthly.rename(columns={"date": "month"})
        if not monthly.empty:
            chart = monthly.melt(id_vars=monthly.columns[0], var_name="Flow", value_name="Amount (₹)")
            chart = chart.rename(columns={chart.columns[0]: "Month"})
            st.plotly_chart(px.line(chart, x="Month", y="Amount (₹)", color="Flow", markers=True), use_container_width=True)


def show_segments(analysis: dict[str, Any]) -> None:
    st.subheader("Behavioral customer groups")
    st.caption(f"Exploratory K-Means grouping selected {analysis['selected_cluster_count']} clusters using the highest silhouette score in the evaluated range.")
    profiles = analysis["cluster_profiles"]
    st.dataframe(profiles, use_container_width=True)
    features = analysis["customer_segments"]
    if {"behavioral_cluster", "debit_transaction_count", "average_debit_amount"}.issubset(features.columns):
        st.plotly_chart(px.scatter(features, x="debit_transaction_count", y="average_debit_amount", color="behavioral_cluster", hover_name="customer_id", labels={"debit_transaction_count": "Debit transactions", "average_debit_amount": "Average debit (₹)", "behavioral_cluster": "Cluster"}), use_container_width=True)
    with st.expander("Cluster-count evaluation"):
        st.dataframe(analysis["cluster_evaluation"], hide_index=True, use_container_width=True)
    st.caption("Clusters are exploratory groupings from this dataset. They are not fixed customer types or financial recommendations.")


def show_anomalies(analysis: dict[str, Any]) -> None:
    st.subheader("Unusual transactions")
    predictions = analysis["anomaly_predictions"].copy()
    if "anomaly_flag" in predictions.columns:
        flagged = predictions.loc[predictions["anomaly_flag"].eq("anomaly")]
    elif "anomaly_prediction" in predictions.columns:
        flagged = predictions.loc[predictions["anomaly_prediction"].eq(-1)]
    else:
        flagged = predictions
    st.metric("Transactions flagged", f"{len(flagged):,}")
    st.caption("A flag means the record is unusual relative to modeled historical patterns. It does not establish fraud, error, or harm.")
    st.dataframe(flagged, hide_index=True, use_container_width=True)
    evaluation = analysis.get("anomaly_evaluation")
    if evaluation:
        with st.expander("Evaluation on supplied reference labels"):
            st.dataframe(evaluation["metrics"], hide_index=True, use_container_width=True)
            st.dataframe(evaluation["confusion_matrix"], use_container_width=True)


def main() -> None:
    st.title("Financial Transaction Intelligence")
    st.markdown("Explore transaction quality, customer behavior, unusual activity, and data-supported findings.")
    st.caption("Prototype analytics on uploaded CSV data. Unusual activity is not a fraud determination.")

    with st.sidebar:
        st.header("Transaction data")
        upload = st.file_uploader("Upload a CSV", type=["csv"], help="The file is processed in memory and is not written to disk.")
        st.markdown("**Required fields**: transaction ID, customer, date, amount, category, and transaction direction.")
        contamination = st.slider("Anomaly flagging assumption", min_value=0.005, max_value=0.05, value=0.01, step=0.005, format="%.3f", help="The assumed share of unusual observations used by Isolation Forest.")

    if upload is None:
        st.info("Upload a CSV to begin. Common headers such as `transaction_date`, `txn_id`, `merchant_name`, and `direction` are recognized.")
        st.markdown("### What you can explore")
        cols = st.columns(4)
        cols[0].write("**Data quality**\n\nAccepted, rejected, and flagged rows.")
        cols[1].write("**Cash flow**\n\nCategory outflow and monthly inflow/outflow.")
        cols[2].write("**Customer groups**\n\nBehavioral cluster profiles and evaluation.")
        cols[3].write("**Unusual activity**\n\nModel flags with transaction context.")
        return

    try:
        transactions, quality_summary = load_and_prepare(upload)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not prepare this file: {error}")
        return
    except Exception as error:  # Surface actionable processing errors without a blank app crash.
        st.error(f"The uploaded file could not be processed: {error}")
        return

    st.success("File checked and prepared. No uploaded file is saved by this dashboard.")
    show_quality(quality_summary, transactions)
    try:
        with st.spinner("Analyzing customer behavior and transaction patterns…"):
            analysis = run_analysis(transactions, contamination)
    except (ValueError, KeyError) as error:
        st.warning(f"The file passed basic preparation, but analytics could not run: {error}")
        st.dataframe(transactions.head(100), hide_index=True, use_container_width=True)
        return
    except Exception as error:
        st.error(f"Analytics could not complete: {error}")
        return

    overview_tab, segments_tab, anomalies_tab, insights_tab = st.tabs(["Overview", "Customer groups", "Unusual activity", "Insights & limits"])
    with overview_tab:
        show_overview(transactions)
    with segments_tab:
        show_segments(analysis)
    with anomalies_tab:
        show_anomalies(analysis)
    with insights_tab:
        st.subheader("Calculated insights")
        st.dataframe(analysis["insights"], hide_index=True, use_container_width=True)
        st.markdown("**Interpretation limits**")
        st.write("- Results describe only the uploaded data; they do not establish causes or population-wide patterns.")
        st.write("- Customer groups are exploratory and may vary with the dataset and model settings.")
        st.write("- An anomaly is a statistical flag, not a fraud classification.")
        st.write("- Insights summarize measured values and are not financial advice.")


if __name__ == "__main__":
    main()
