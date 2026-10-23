
import pandas as pd

from src.feature_engineering import customer_features


def create_test_data():
    """Create sample enriched transaction data for testing."""

    data = pd.DataFrame({
        "customer_id": ["C001", "C001", "C001", "C002", "C002"],
        "transaction_type": ["credit", "debit", "credit", "debit", "credit"],
        "amount": [5000, -1000, 2000, -3000, 4000],
        "year": [2026, 2026, 2026, 2026, 2026],
        "month": [1, 1, 2, 1, 2],
    })

    return data


def test_total_inflow():
    data = create_test_data()

    result = customer_features(data)

    c001_inflow = result.loc[
        result["customer_id"] == "C001", "total_inflow"
    ].iloc[0]

    c002_inflow = result.loc[
        result["customer_id"] == "C002", "total_inflow"
    ].iloc[0]

    assert c001_inflow == 7000
    assert c002_inflow == 4000


def test_total_outflow():
    data = create_test_data()

    result = customer_features(data)

    c001_outflow = result.loc[
        result["customer_id"] == "C001", "total_outflow"
    ].iloc[0]

    c002_outflow = result.loc[
        result["customer_id"] == "C002", "total_outflow"
    ].iloc[0]

    assert c001_outflow == 1000
    assert c002_outflow == 3000


def test_net_cash_flow():
    data = create_test_data()

    result = customer_features(data)

    c001_net_flow = result.loc[
        result["customer_id"] == "C001", "net_cash_flow"
    ].iloc[0]

    c002_net_flow = result.loc[
        result["customer_id"] == "C002", "net_cash_flow"
    ].iloc[0]

    assert c001_net_flow == 6000
    assert c002_net_flow == 1000


def test_transaction_count():
    data = create_test_data()

    result = customer_features(data)

    c001_count = result.loc[
        result["customer_id"] == "C001", "transaction_count"
    ].iloc[0]

    c002_count = result.loc[
        result["customer_id"] == "C002", "transaction_count"
    ].iloc[0]

    assert c001_count == 3
    assert c002_count == 2


def test_average_transaction_amount():
    data = create_test_data()

    result = customer_features(data)

    c001_average = result.loc[
        result["customer_id"] == "C001",
        "average_transaction_amount",
    ].iloc[0]

    c002_average = result.loc[
        result["customer_id"] == "C002",
        "average_transaction_amount",
    ].iloc[0]

    assert abs(c001_average - (8000 / 3)) < 0.001
    assert c002_average == 3500


def test_one_row_per_customer():
    data = create_test_data()

    result = customer_features(data)

    assert len(result) == 2
    assert result["customer_id"].nunique() == 2


def test_input_data_is_not_modified():
    data = create_test_data()
    original_data = data.copy(deep=True)

    customer_features(data)

    pd.testing.assert_frame_equal(data, original_data)
