"""Streamlit dashboard for Financial Transaction Intelligence.

Run from the repository root with ``streamlit run app/app.py``.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics_pipeline import run_analytical_pipeline
from src.anomaly_detection import evaluate_anomaly_predictions
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

MUTED = "#C4CDD8"
MONEY_IN = "#007F78"
MONEY_OUT = "#C4455C"
PALETTE = ["#2463A6", "#6741C6", "#007F78", "#A24F22", "#9B3C8C", "#567A32", "#386F82"]


def money(value: float, decimals: int = 0) -> str:
    return f"₹{value:,.{decimals}f}"


def money_summary(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10_000_000:
        return f"₹{value / 10_000_000:,.2f} crore"
    if magnitude >= 100_000:
        return f"₹{value / 100_000:,.2f} lakh"
    return money(value)


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    """Set sizing and spacing while leaving colors and text to Streamlit's theme."""
    fig.update_layout(
        height=height,
        legend={"orientation": "h", "y": -0.22, "x": 0},
        margin={"l": 30, "r": 25, "t": 24, "b": 55},
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def render_chart(fig: go.Figure) -> None:
    """Render charts with Streamlit's active light or dark theme."""
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def _is_blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("").fillna(False)


def load_and_prepare(upload: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, clean, and enrich an uploaded transaction file in memory."""
    try:
        # Keeping values as text preserves identifiers such as 000123. Amounts
        # and dates are parsed by the existing validation/preprocessing steps.
        raw = pd.read_csv(io.BytesIO(upload.getvalue()), dtype="string")
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file is empty.") from error

    if raw.empty:
        raise ValueError("The CSV has headers but no transaction rows.")

    mapped = map_columns(raw)
    if mapped is None:
        raise ValueError("Some column names map to the same field. Keep only one column for each transaction detail.")

    required = {"transaction_id", "customer_id", "date", "amount", "category", "transaction_type"}
    missing = sorted(required.difference(mapped.columns))
    if missing:
        raise ValueError("This analysis needs these columns: " + ", ".join(missing))

    validation = validate_data(mapped)
    candidates = validation["accepted"].copy()
    if candidates.empty:
        raise ValueError("No rows passed the customer, date, and amount checks.")

    # The analytics pipeline requires unique transaction IDs, a category, and
    # a recognized direction. The shared validator reports some of these as
    # review flags rather than rejecting them, so exclude them explicitly here.
    ids = candidates["transaction_id"].astype("string").str.strip()
    candidates["transaction_id"] = ids
    missing_id = _is_blank(candidates["transaction_id"])
    duplicate_id = ids.duplicated(keep=False) & ~missing_id
    missing_category = _is_blank(candidates["category"])
    directions = candidates["transaction_type"].astype("string").str.strip().str.lower()
    invalid_direction = ~directions.isin(["debit", "credit", "dr", "cr"])

    custom_issues = {
        "Missing transaction ID": missing_id,
        "Duplicate transaction ID": duplicate_id,
        "Missing category": missing_category,
        "Unrecognized transaction direction": invalid_direction,
    }
    issue_labels = {
        "Missing transaction ID": "Missing transaction ID",
        "Duplicate transaction ID": "Duplicate transaction ID",
        "Missing category": "Missing category",
        "Unrecognized transaction direction": "Missing or unrecognised directions",
    }
    validation_issues = candidates["validation_issues"].fillna("").astype("string")
    exclude = pd.Series(False, index=candidates.index)
    issue_counts = dict(validation["summary"]["issue_counts"])
    for issue, mask in custom_issues.items():
        label = issue_labels[issue]
        for row_index in candidates.index[mask]:
            current = validation_issues.at[row_index]
            if label not in current.split("; "):
                validation_issues.at[row_index] = "; ".join(part for part in [current, label] if part)
        # Category and direction checks are already counted by validate_data.
        if issue not in {"Missing category", "Unrecognized transaction direction"}:
            count = int(mask.sum())
            if count:
                issue_counts[issue] = issue_counts.get(issue, 0) + count
        exclude |= mask
    candidates["validation_issues"] = validation_issues

    rejected_additional = set(candidates.index[exclude])
    accepted_positions = set(candidates.index[~exclude])
    flagged_positions = set(validation["flagged"].index) | rejected_additional
    rejected_positions = set(validation["rejected"].index) | rejected_additional
    excluded_candidates = candidates.loc[exclude].copy()
    rejected_preview = pd.concat(
        [validation["rejected"], excluded_candidates],
        axis=0,
    ).sort_index().head(100)
    candidates = candidates.loc[~exclude].copy()
    if candidates.empty:
        raise ValueError("No rows have all the fields required for analysis. Review the data-quality issues.")

    cleaned = preprocess_data(candidates)
    enriched = enrich_data(cleaned["cleaned"])
    summary = {
        "raw_rows": len(raw),
        "mapped_columns": list(mapped.columns),
        "validation": {
            **validation["summary"],
            "accepted_rows": len(accepted_positions),
            "rejected_rows": len(rejected_positions),
            "flagged_rows": len(flagged_positions),
            "issue_counts": issue_counts,
            "excluded_for_model_requirements": len(rejected_additional),
        },
        "rejected_preview": rejected_preview,
        "flagged_preview": validation["flagged"].head(100),
        "preprocessing": cleaned["report"],
    }
    return enriched["enriched"], summary


def load_reference_labels(upload: Any) -> pd.DataFrame:
    """Read optional labels used only to evaluate predictions, never as features."""
    try:
        labels = pd.read_csv(io.BytesIO(upload.getvalue()), dtype="string")
    except pd.errors.EmptyDataError as error:
        raise ValueError("The labels CSV is empty.") from error
    required = {"transaction_id", "is_injected_anomaly"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError("The labels file needs these columns: " + ", ".join(missing))
    labels = labels[["transaction_id", "is_injected_anomaly"]].copy()
    labels["transaction_id"] = labels["transaction_id"].astype("string").str.strip()
    labels["is_injected_anomaly"] = pd.to_numeric(labels["is_injected_anomaly"], errors="coerce")
    if _is_blank(labels["transaction_id"]).any() or labels["transaction_id"].duplicated().any():
        raise ValueError("Each reference label must have a non-empty, unique transaction ID.")
    if labels["is_injected_anomaly"].isna().any() or not labels["is_injected_anomaly"].isin([0, 1]).all():
        raise ValueError("is_injected_anomaly must contain only 0 or 1.")
    return labels


def run_analysis(
    transactions: pd.DataFrame,
    contamination: float,
    reference_labels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    # Keep labels out of model fitting. They are joined only after prediction.
    analysis = run_analytical_pipeline(transactions, contamination=contamination)
    analysis["anomaly_evaluation"] = None
    analysis["anomaly_evaluation_message"] = None
    if reference_labels is not None:
        try:
            analysis["anomaly_evaluation"] = evaluate_anomaly_predictions(
                analysis["anomaly_predictions"], reference_labels
            )
        except ValueError as error:
            analysis["anomaly_evaluation_message"] = str(error)
    return analysis


def cached_analysis(
    transactions: pd.DataFrame,
    contamination: float,
    upload_digest: str,
    reference_labels: pd.DataFrame | None,
    label_digest: str,
) -> dict[str, Any]:
    """Avoid rerunning the models on ordinary Streamlit reruns."""
    cache = st.session_state.setdefault("_financial_analysis_cache", {})
    key = (upload_digest, contamination, label_digest)
    if key not in cache:
        cache.clear()
        cache[key] = run_analysis(transactions, contamination, reference_labels)
    return cache[key]


def monthly_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    rows = transactions.copy()
    rows["month"] = rows["date"].dt.to_period("M").astype(str)
    rows["Money in"] = rows["amount"].where(rows["transaction_type"].eq("credit"), 0)
    rows["Money out"] = rows["amount"].abs().where(rows["transaction_type"].eq("debit"), 0)
    return rows.groupby("month", as_index=False)[["Money in", "Money out"]].sum()


def show_overview(transactions: pd.DataFrame) -> None:
    inflow = transactions.loc[transactions["transaction_type"].eq("credit"), "amount"].sum()
    outflow = transactions.loc[transactions["transaction_type"].eq("debit"), "amount"].abs().sum()
    metrics = st.columns(4)
    metrics[0].metric("Transactions", f"{len(transactions):,}")
    metrics[1].metric("Customers", f"{transactions['customer_id'].nunique():,}")
    metrics[2].metric("Money in", money_summary(float(inflow)))
    metrics[3].metric("Money out", money_summary(float(outflow)))

    st.subheader("Monthly money in and out")
    st.caption("Each bar is a monthly total across all customers in the uploaded file.")
    monthly = monthly_summary(transactions)
    flow_data = monthly.melt(id_vars="month", var_name="Flow", value_name="Amount (₹)")
    fig = px.bar(
        flow_data,
        x="month",
        y="Amount (₹)",
        color="Flow",
        barmode="group",
        color_discrete_map={"Money in": MONEY_IN, "Money out": MONEY_OUT},
        category_orders={"Flow": ["Money in", "Money out"]},
        labels={"month": "Month"},
    )
    fig.update_layout(legend_title_text="")
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: ₹%{y:,.0f}<extra></extra>")
    render_chart(chart_layout(fig, 380))

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Where money went")
        st.caption("Categories ranked by total debit amount. Investment and transfers may not be everyday spending.")
        debit = transactions.loc[transactions["transaction_type"].eq("debit")].copy()
        debit["Debit amount (₹)"] = debit["amount"].abs()
        totals = debit.groupby("category", as_index=False)["Debit amount (₹)"].sum().sort_values("Debit amount (₹)")
        if not totals.empty:
            fig = px.bar(totals, x="Debit amount (₹)", y="category", orientation="h", text="Debit amount (₹)", color="category", color_discrete_sequence=PALETTE)
            fig.update_traces(texttemplate="₹%{x:,.0f}", textposition="outside", cliponaxis=False, showlegend=False)
            fig.update_layout(yaxis_title="")
            fig = chart_layout(fig, max(300, len(totals) * 30 + 100))
            fig.update_layout(margin={"l": 20, "r": 85, "t": 24, "b": 55})
            render_chart(fig)
        else:
            st.info("There are no debit transactions in this file.")
    with right:
        st.subheader("Transaction count by direction")
        st.caption("This is a count of records, not a comparison of money amounts.")
        mix = transactions["transaction_type"].value_counts().rename(index={"credit": "Money in", "debit": "Money out"}).rename_axis("Direction").reset_index(name="Transactions")
        fig = px.pie(mix, names="Direction", values="Transactions", hole=.55, color="Direction", color_discrete_map={"Money in": MONEY_IN, "Money out": MONEY_OUT})
        fig.update_traces(textposition="outside", textinfo="label+percent", hovertemplate="%{label}: %{value:,} transactions (%{percent})<extra></extra>")
        fig.update_layout(showlegend=False, margin={"l": 30, "r": 30, "t": 20, "b": 30})
        render_chart(chart_layout(fig, 350))


def show_spending(transactions: pd.DataFrame) -> None:
    st.subheader("Spending patterns")
    st.caption("Compare categories over time, merchant totals, and the range of debit amounts.")
    debit = transactions.loc[transactions["transaction_type"].eq("debit")].copy()
    debit["Debit amount (₹)"] = debit["amount"].abs()
    debit["Month"] = debit["date"].dt.to_period("M").astype(str)
    pivot = debit.pivot_table(index="category", columns="Month", values="Debit amount (₹)", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    pivot.columns = [pd.Period(month).strftime("%b %Y") for month in pivot.columns]
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### How category spending changes by month")
        st.caption("Darker cells show larger totals. Values are debit amounts in rupees.")
        if not pivot.empty:
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale=[[0, "#F5E9B8"], [.5, "#E99B54"], [1, "#A53B50"]], labels={"x": "Month", "y": "Category", "color": "Debit (₹)"}, text_auto=".2s")
            fig.update_layout(coloraxis_colorbar={"title": "Debit (₹)", "tickformat": ",.0f"})
            render_chart(chart_layout(fig, max(340, 30 * len(pivot) + 100)))
        else:
            st.info("No debit activity is available for the heatmap.")
    with right:
        st.markdown("#### Merchants by total debit")
        st.caption("Labels show transaction count; bar length shows amount.")
        if "merchant" in debit.columns and debit["merchant"].notna().any():
            merchant = debit.groupby("merchant", as_index=False).agg(
                **{"Debit amount (₹)": ("Debit amount (₹)", "sum"), "Transactions": ("transaction_id", "count")}
            ).nlargest(10, "Debit amount (₹)").sort_values("Debit amount (₹)")
            fig = px.bar(merchant, x="Debit amount (₹)", y="merchant", orientation="h", color="Transactions", color_continuous_scale=[[0, "#9BC9D2"], [1, "#2463A6"]], text="Transactions", labels={"merchant": "Merchant"})
            fig.update_traces(texttemplate="%{text:,} txns", textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Total debit: ₹%{x:,.0f}<br>Transactions: %{text:,}<extra></extra>")
            fig.update_layout(coloraxis_colorbar={"title": "Transactions"})
            fig = chart_layout(fig, 390)
            fig.update_layout(margin={"l": 20, "r": 85, "t": 24, "b": 55})
            render_chart(fig)
        else:
            st.info("Merchant names were not provided in this file.")

    st.markdown("#### How debit amounts vary by category")
    st.caption("Boxes show the middle half of values. The vertical axis is logarithmic because transaction amounts vary widely.")
    if not debit.empty:
        fig = px.box(debit, x="category", y="Debit amount (₹)", color="category", color_discrete_sequence=PALETTE, points="outliers")
        fig.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Debit amount (₹)")
        fig.update_yaxes(type="log", tickprefix="₹")
        render_chart(chart_layout(fig, 370))


def show_customers(analysis: dict[str, Any]) -> None:
    st.subheader("Customer groups")
    st.caption("These are exploratory groups based on similar transaction patterns in this file, not fixed customer types.")
    features = analysis["customer_segments"].copy()
    profiles = analysis["cluster_profiles"]
    means = profiles.xs("mean", axis=1, level=1)
    share_columns = [column for column in means if column.startswith("share_")]
    if share_columns:
        profile = means[share_columns].rename(columns=lambda column: column.removeprefix("share_")).T
        profile.index.name = "Spending category"
        profile.columns = [f"Group {cluster}" for cluster in profile.columns]
        st.markdown("#### What each group spends on")
        st.caption("Cells show each group’s average share of debit spending by category.")
        fig = px.imshow(profile, aspect="auto", text_auto=".1f", color_continuous_scale=[[0, "#E5EDF5"], [.5, "#70B6A9"], [1, "#6741C6"]], labels={"x": "Customer group", "y": "Category", "color": "Average share (%)"})
        render_chart(chart_layout(fig, max(320, len(profile) * 30 + 100)))

    behavior_columns = ["debit_transaction_count", "average_debit_amount", "weekend_transaction_share"]
    available = [column for column in behavior_columns if column in features]
    labels = {"debit_transaction_count": "Debit transactions per customer", "average_debit_amount": "Average debit amount (₹)", "weekend_transaction_share": "Weekend transaction share (%)"}
    left, right = st.columns(2)
    with left:
        st.markdown("#### Compare a behavior")
        measure = st.selectbox("Choose a measure", available, format_func=lambda value: labels.get(value, value))
        display = features.copy()
        if measure == "weekend_transaction_share":
            display[measure] = display[measure] * 100
        display["Customer group"] = display["behavioral_cluster"].map(lambda cluster: f"Group {cluster}")
        fig = px.box(display, x="Customer group", y=measure, color="Customer group", points="all", color_discrete_sequence=PALETTE, labels={measure: labels.get(measure, measure)})
        fig.update_layout(showlegend=False)
        render_chart(chart_layout(fig, 340))
    with right:
        st.markdown("#### How distinct are the groups?")
        st.caption("Higher silhouette scores mean groups are more separate in this model’s feature space.")
        scores = analysis["cluster_evaluation"]
        fig = px.line(scores, x="k", y="silhouette_score", markers=True, labels={"k": "Number of groups", "silhouette_score": "Separation score"})
        selected = int(analysis["selected_cluster_count"])
        chosen = scores.loc[scores["k"].eq(selected)]
        if not chosen.empty:
            fig.add_scatter(x=chosen["k"], y=chosen["silhouette_score"], mode="markers+text", text=["Selected"], textposition="top center", marker={"size": 13, "color": "#9A5A00", "symbol": "star"}, name="Selected group count")
        render_chart(chart_layout(fig, 340))

    st.markdown("#### Customer pattern map")
    st.caption("Each point is one customer. Nearby points have more similar debit frequency and average debit size.")
    scatter = features.copy()
    scatter["Customer group"] = scatter["behavioral_cluster"].map(lambda cluster: f"Group {cluster}")
    fig = px.scatter(scatter, x="debit_transaction_count", y="average_debit_amount", color="Customer group", hover_name="customer_id", color_discrete_sequence=PALETTE, labels={"debit_transaction_count": "Debit transactions", "average_debit_amount": "Average debit amount (₹)"})
    render_chart(chart_layout(fig, 380))

    with st.expander("View detailed cluster profiles and customer assignments"):
        st.dataframe(profiles, use_container_width=True)
        st.dataframe(features, hide_index=True, use_container_width=True)


def show_anomalies(analysis: dict[str, Any]) -> None:
    predictions = analysis["anomaly_predictions"].copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["Review status"] = predictions["anomaly_flag"].map({"anomaly": "Worth reviewing", "normal": "Within modeled pattern"}).fillna("No historical baseline")
    flagged = predictions.loc[predictions["anomaly_flag"].eq("anomaly")].copy()
    st.subheader("Transactions worth a closer look")
    st.caption("A flag means a transaction differs from modeled past patterns. It does not prove fraud, error, or harm.")
    st.metric("Flagged transactions", f"{len(flagged):,}")

    routine = predictions.loc[~predictions["Review status"].eq("Worth reviewing")]
    if len(routine) > 6000:
        routine = routine.sample(6000, random_state=42)
        st.caption("To keep the chart responsive, it samples up to 6,000 unflagged transactions. All flagged transactions remain visible.")
    timeline = pd.concat([routine, flagged], ignore_index=True)
    timeline["Amount magnitude (₹)"] = timeline["amount"].abs()
    timeline = timeline.loc[timeline["Amount magnitude (₹)"].gt(0)]
    fig = px.scatter(timeline, x="date", y="Amount magnitude (₹)", color="Review status", symbol="Review status", color_discrete_map={"Worth reviewing": "#9A5A00", "Within modeled pattern": "#2463A6", "No historical baseline": "#66758B"}, hover_data=["transaction_id", "customer_id", "category", "historical_amount_ratio", "explanation"], labels={"date": "Date"})
    fig.update_yaxes(type="log", tickprefix="₹")
    fig.update_layout(legend_title_text="")
    render_chart(chart_layout(fig, 380))
    if (predictions["amount"].eq(0)).any():
        st.caption("Zero-value rows are left off this logarithmic chart; flagged zero-value rows remain in the details table.")

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### Transaction amount and historical reference")
        st.caption("Points above the dashed equal-value line are larger than their historical reference amount.")
        comparable = predictions.loc[
            predictions["reference_amount"].gt(0) & predictions["amount"].abs().gt(0)
        ].copy()
        if not comparable.empty:
            comparable["Amount magnitude (₹)"] = comparable["amount"].abs()
            fig = px.scatter(comparable, x="reference_amount", y="Amount magnitude (₹)", color="Review status", symbol="Review status", color_discrete_map={"Worth reviewing": "#9A5A00", "Within modeled pattern": "#2463A6", "No historical baseline": "#66758B"}, hover_data=["transaction_id", "customer_id", "category", "historical_amount_ratio"], labels={"reference_amount": "Historical reference (₹)"})
            positive_values = pd.concat([comparable["reference_amount"], comparable["Amount magnitude (₹)"]])
            positive_values = positive_values.loc[positive_values.gt(0)]
            if not positive_values.empty:
                low, high = float(positive_values.min()), float(positive_values.max())
                fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"color": MUTED, "dash": "dash", "width": 2})
            fig.update_xaxes(type="log", tickprefix="₹")
            fig.update_yaxes(type="log", tickprefix="₹")
            render_chart(chart_layout(fig, 340))
        else:
            st.info("No transactions had enough history to make this comparison.")
    with right:
        st.markdown("#### Flagged transaction details")
        if flagged.empty:
            st.info("No transactions were flagged for this file.")
        else:
            categories = sorted(flagged["category"].dropna().astype(str).unique())
            chosen_categories = st.multiselect("Filter flagged transactions by category", categories, default=categories)
            shown = flagged.loc[flagged["category"].astype(str).isin(chosen_categories)]
            columns = [column for column in ["date", "transaction_id", "customer_id", "category", "amount", "historical_amount_ratio", "baseline_source", "explanation"] if column in shown]
            st.dataframe(shown[columns], hide_index=True, use_container_width=True, height=340)

    evaluation = analysis.get("anomaly_evaluation")
    evaluation_message = analysis.get("anomaly_evaluation_message")
    if evaluation:
        st.markdown("#### Evaluation against supplied reference labels")
        st.caption("Labels are used only after model predictions are made. Synthetic labels are not verified real-world fraud outcomes.")
        metric_col, matrix_col = st.columns(2)
        with metric_col:
            st.dataframe(evaluation["metrics"], hide_index=True, use_container_width=True)
        with matrix_col:
            matrix = evaluation["confusion_matrix"]
            fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, "#EAF0F6"], [1, "#2463A6"]], labels={"x": "Model result", "y": "Reference label", "color": "Transactions"})
            render_chart(chart_layout(fig, 300))
        pr = evaluation.get("precision_recall_curve")
        if pr is not None and not pr.empty:
            st.markdown("#### Flagging trade-off")
            fig = px.line(pr, x="recall", y="precision", labels={"recall": "Share of reference anomalies found", "precision": "Share of flags matching a reference anomaly"})
            render_chart(chart_layout(fig, 320))
    elif evaluation_message:
        st.warning(f"The predictions were created, but the supplied labels could not be evaluated: {evaluation_message}")


def show_quality(summary: dict[str, Any], transactions: pd.DataFrame) -> None:
    quality = summary["validation"]
    st.subheader("Data quality")
    st.caption("Rows excluded from modeling are shown separately from rows kept for analysis.")
    metrics = st.columns(4)
    metrics[0].metric("Rows received", f"{summary['raw_rows']:,}")
    metrics[1].metric("Rows analyzed", f"{quality['accepted_rows']:,}")
    metrics[2].metric("Rows excluded", f"{quality['rejected_rows']:,}")
    metrics[3].metric("Rows with issues", f"{quality['flagged_rows']:,}")

    issue_counts = quality["issue_counts"]
    if issue_counts:
        issues = pd.DataFrame({"Issue": list(issue_counts), "Rows affected": list(issue_counts.values())}).sort_values("Rows affected")
        fig = px.bar(issues, x="Rows affected", y="Issue", orientation="h", text="Rows affected", color="Issue", color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
        fig = chart_layout(fig, max(280, len(issues) * 34 + 80))
        fig.update_layout(margin={"l": 30, "r": 60, "t": 24, "b": 55})
        render_chart(fig)
    st.markdown("Rows with missing customer IDs, invalid dates or amounts, duplicate/missing transaction IDs, missing categories, or invalid directions are excluded. Other issues, such as missing descriptions, are reported and retained.")
    with st.expander("Recognized columns and prepared data"):
        st.write(", ".join(summary["mapped_columns"]))
        st.dataframe(transactions.head(100), hide_index=True, use_container_width=True)
    rejected_preview = summary.get("rejected_preview")
    if rejected_preview is not None and not rejected_preview.empty:
        with st.expander("Review sample rows excluded from analysis"):
            st.dataframe(rejected_preview, hide_index=True, use_container_width=True)
    flagged_preview = summary.get("flagged_preview")
    if flagged_preview is not None and not flagged_preview.empty:
        with st.expander("Review sample rows kept with data-quality issues"):
            st.dataframe(flagged_preview, hide_index=True, use_container_width=True)


def show_insights(analysis: dict[str, Any]) -> None:
    st.subheader("Calculated insights")
    st.caption("These summaries come from measured customer, category, cluster, and transaction outputs.")
    insights = analysis["insights"]
    text_column = insights.loc[insights["insight_text"].fillna("").ne("")].copy()
    if text_column.empty:
        st.info("There were no rule-based insights for this file.")
    else:
        st.dataframe(text_column[["insight_type", "customer_id", "transaction_id", "insight_text"]].head(100), hide_index=True, use_container_width=True)
        if len(text_column) > 100:
            st.caption("Showing the first 100 insights. The full result is limited to keep the page easy to scan.")
    st.markdown("#### How to interpret the results")
    st.write("- Results describe the uploaded records, not financial activity outside the file.")
    st.write("- Customer groups are exploratory and may change with the sample and model settings.")
    st.write("- An unusual-transaction flag is a prompt to review a record, not a fraud classification.")
    st.write("- Category outflow can include transfers or investments, not only consumption.")
    st.write("- Findings from the project’s generated sample are synthetic and should not be generalized to real customers.")


def main() -> None:
    with st.sidebar:
        st.header("Transaction data")
        with st.form("analysis_inputs", clear_on_submit=False):
            upload = st.file_uploader("Upload a transaction CSV", type=["csv"], help="The file is processed in memory and is not saved by the dashboard.")
            labels_upload = st.file_uploader("Optional: evaluation labels CSV", type=["csv"], help="Needs transaction_id and is_injected_anomaly (0 or 1). Labels are used only to evaluate predictions, never to train the model.")
            st.markdown("**Required transaction fields**\n\nTransaction ID · Customer ID · Date · Amount · Category · Direction")
            contamination = st.slider("Assumed share of unusual transactions", 0.005, 0.05, 0.01, 0.005, format="%.3f", help="Isolation Forest uses this assumption to set how many records it flags. It is not a fraud threshold.")
            start_analysis = st.form_submit_button("Analyze transactions", width="stretch")
        if start_analysis:
            st.session_state["analysis_confirmed"] = True

    st.title("Financial Transaction Intelligence")
    st.markdown("Explore cash flow, spending patterns, customer groups, and transactions that may deserve a closer look.")
    st.caption("Amounts are shown in Indian rupees. A model flag is not a fraud finding.")

    if upload is None:
        st.info("Upload a transaction CSV to begin. Common headers such as transaction_date, txn_id, merchant_name, and direction are recognized.")
        st.markdown("### Dashboard sections")
        st.write("**Overview** — monthly money in and out, category totals, and transaction counts.")
        st.write("**Spending patterns** — category-by-month heatmap, merchant totals, and transaction-size distributions.")
        st.write("**Customer groups** — category-share profiles and cluster comparisons.")
        st.write("**Unusual activity** — model flags, historical comparisons, and optional reference-label evaluation.")
        st.write("**Data quality and insights** — row checks, exclusions, measured findings, and interpretation notes.")
        return

    if not st.session_state.get("analysis_confirmed", False):
        st.info("Your file is selected. Click **Analyze transactions** in the sidebar when you’re ready to start.")
        return

    try:
        transactions, quality = load_and_prepare(upload)
        reference_labels = load_reference_labels(labels_upload) if labels_upload is not None else None
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"This file could not be prepared: {error}")
        return
    except Exception as error:
        st.error(f"The uploaded file could not be processed: {error}")
        return

    upload_digest = hashlib.sha256(upload.getvalue()).hexdigest()
    label_digest = hashlib.sha256(labels_upload.getvalue()).hexdigest() if labels_upload is not None else ""
    analysis = None
    analysis_error = None
    try:
        with st.spinner("Analyzing transaction patterns…"):
            analysis = cached_analysis(transactions, contamination, upload_digest, reference_labels, label_digest)
    except Exception as error:
        # Keep descriptive charts available even when a model's assumptions
        # cannot be met by a small or unusual upload.
        analysis_error = str(error)

    tabs = st.tabs(["Overview", "Spending patterns", "Customer groups", "Unusual activity", "Data quality & insights"])
    with tabs[0]:
        show_overview(transactions)
    with tabs[1]:
        show_spending(transactions)
    with tabs[2]:
        if analysis is not None:
            show_customers(analysis)
        else:
            st.warning(f"Customer-group analysis is unavailable for this file: {analysis_error}")
    with tabs[3]:
        if analysis is not None:
            show_anomalies(analysis)
        else:
            st.warning(f"Unusual-transaction analysis is unavailable for this file: {analysis_error}")
    with tabs[4]:
        show_quality(quality, transactions)
        if analysis is not None:
            show_insights(analysis)
        else:
            st.info("The descriptive charts and data checks are available, but model-based insights could not be calculated for this file.")


if __name__ == "__main__":
    main()
