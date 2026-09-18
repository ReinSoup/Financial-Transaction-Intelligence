# Financial Transaction Intelligence

An ML-powered financial analytics project that transforms raw transaction data into meaningful spending patterns and anomaly signals.

## Overview

Financial transaction data is often messy and inconsistent. Merchant names, dates, amounts, categories, and column names can vary across different data sources.

Financial Transaction Intelligence processes this data through a complete data science pipeline:

```text
Raw Transactions
       ↓
Ingestion & Validation
       ↓
Cleaning & Normalization
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Machine Learning
   ┌───┴──────────┐
   ↓              ↓
Clustering   Anomaly Detection
   ↓              ↓
Behavioral      Unusual
Patterns      Transactions
   └──────┬───────┘
          ↓
   Explainable Insights
          ↓
 Streamlit Dashboard
```

The project focuses on understanding spending behavior through data rather than simply reporting transaction totals.

## Key Features

* Ingests transaction data from CSV files
* Standardizes inconsistent schemas
* Cleans and normalizes dates, amounts, merchants, and categories
* Performs exploratory data analysis on spending behavior
* Engineers behavioral and spending features
* Uses K-Means to identify behavioral patterns
* Uses Isolation Forest to detect unusual transactions
* Evaluates and interprets machine learning results
* Generates simple, explainable insights
* Presents results through an interactive Streamlit dashboard

## Machine Learning

### Behavioral Clustering

K-Means clustering is used to group similar spending behavior based on engineered features such as transaction frequency, average transaction value, category spending ratios, and temporal spending patterns.

The resulting clusters are analyzed based on their characteristics rather than being treated as predefined labels.

### Anomaly Detection

Isolation Forest is used to identify transactions that differ significantly from typical transaction behavior.

The system focuses on detecting unusual activity rather than attempting to determine whether a transaction is fraudulent.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly
* Streamlit
* Jupyter
* Git & GitHub

## Project Structure

```text
financial-transaction-intelligence/
│
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── analytics/
│   ├── ml/
│   └── insights/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

## Project Status

In development.

Current focus:

```text
Ingestion
    ↓
Preprocessing
    ↓
EDA
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Dashboard
```

## Privacy

Only public, synthetic, or anonymized transaction data will be used for development and demonstration.

Real financial information will not be committed to the repository.

## Authors

Abdul Rab & Bonthu Sai Vennela
