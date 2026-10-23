import pandas as pd

def customer_features(enriched_df):

    """
    Decided features:

    total inflow, total outflow,
    net cash flow,
    transaction count, avg transaction amount
    
    """


    data = enriched_df.copy()

    #total inflow

    credit_transactions = data.loc[data["transaction_type"] == "credit"]

    total_inflow = credit_transactions.groupby("customer_id")["amount"].sum().rename("total_inflow")

    #total outflow

    debit_transactions = data.loc[data["transaction_type"] == "debit"]

    total_outflow = (
        debit_transactions["amount"].abs()
        .groupby(debit_transactions["customer_id"])
        .sum().rename("total_outflow")
        )


    # rest of the features

    net_cash_flow = data.groupby("customer_id")["amount"].sum().rename("net_cash_flow")

    transaction_count = data.groupby("customer_id").size().rename("transaction_count")

    average_transaction_amount = (
        data["amount"].abs()
        .groupby(data["customer_id"])
        .mean()
        .rename("average_transaction_amount")
        )


    customer_features = pd.concat([
        total_inflow,
        total_outflow,
        net_cash_flow,
        transaction_count,
        average_transaction_amount], axis = 1)


    customer_features[["total_inflow", "total_outflow"]] = customer_features[["total_inflow", "total_outflow"]].fillna(0)

    customer_features = customer_features.reset_index()

    return customer_features    

    





