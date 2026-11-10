# Financial Transaction Intelligence

An analytical project that turns transaction data into customer spending patterns, unusual-transaction signals, descriptive statistics, and traceable insights. The current dataset is synthetic and denominated in INR.

**Live dashboard:** [Financial Transaction Intelligence](https://reinsoup-money-map.streamlit.app/)
**Sample files:** [Download syntheticmessy.csv](data/raw/syntheticmessy.csv) · [Download labels.csv](data/reference/labels.csv)

## Project status

The project is complete. The analytical backend is implemented and tested against the local synthetic v2.0 dataset, and the Streamlit dashboard is deployed at [reinsoup-money-map.streamlit.app](https://reinsoup-money-map.streamlit.app/).

| Area | Status | Details |
|---|---|---|
| Ingestion and validation | Implemented and tested | CSV loading, schema mapping, validation, preprocessing, date enrichment, and customer-level baseline features |
| Exploratory analysis | Completed | v2.0 transaction, customer, category, merchant, and time-period analysis |
| Financial-scale segmentation | Exploratory | Existing three-feature K-Means baseline is preserved in `Customer_Segmentation.ipynb` |
| Behavioral segmentation | Implemented and tested | Category spending shares, transaction count, average debit amount, weekend share, K-Means evaluation, profiles, and visualizations |
| Transaction anomaly analysis | Experimental and tested | Past-only customer/category reference features, chronological holdout evaluation, Isolation Forest, and false-positive/false-negative review |
| Statistical analysis | Implemented and tested | Descriptive statistics, IQR bounds, customer-level debit summaries, and one exploratory Welch comparison |
| Explainable insights | Implemented and tested | Rule-based outputs tied to customer, transaction, category, cluster, and measured values |
| Analytical integration | Implemented and tested | `run_analytical_pipeline` returns separate customer-level and transaction-level outputs |
| Streamlit dashboard | Implemented | Five plain-language sections, cash-flow and category charts, monthly spending heatmap, customer profiles, anomaly review, and data-quality feedback |

## Dataset

The frozen synthetic v2.0 dataset contains 56,421 transactions from 180 customers, covering January through December 2025. It uses INR and includes 354 injected anomaly labels. The labels are evaluation-only data and are never used as anomaly-model input features.

The repository includes the source transactions, processed transactions, reference customer profiles, anomaly labels, and a generation manifest. The analytical pipeline reads these files but does not overwrite them.

## Analytical workflow

The reusable entry point is `src/analytics_pipeline.py`. It accepts processed transactions and optional reference labels and customer profiles, then returns:

- Customer behavior features and cluster assignments.
- Candidate cluster evaluation scores and original-unit cluster profiles.
- Transaction predictions linked to `transaction_id` and supporting historical ratios.
- Held-out anomaly metrics and a confusion matrix when labels are provided.
- Debit amount summaries, IQR bounds, and an optional customer-level group comparison.
- Structured rule-based insights.

Customer and transaction outputs remain at their own data grains. The pipeline returns results in memory and does not write analytical outputs to the raw or processed data folders.

To run the full analysis from the repository root:

```powershell
.\venv\Scripts\python.exe run_analytical_workflow.py
```

The same workflow is available in `notebooks/analytical_workflow.ipynb`. Run the customer segmentation analysis in `notebooks/Customer_Segmentation.ipynb` and the anomaly workflow in `notebooks/transaction_anomaly.ipynb` for detailed tables, charts, and error review.

## Current analytical results

### Customer segmentation

The category-only K-Means analysis selected two clusters with a silhouette score of 0.595 on category percentage points. The mixed behavioral model also selected two clusters and scored 0.421 on its standardized feature set. The category-only and mixed assignments agree for 178 of 180 customers after matching by customer ID.

The category profiles show a group averaging 34.3% of debit spending on Housing, and another averaging 21.0% on Education, 13.5% on Food & Dining, 12.9% on Transport, and 11.7% on Shopping. Median debit counts and amounts also differ between the groups; weekend transaction shares are similar. The original financial-scale baseline has three clusters and a silhouette score of 0.522 in a different feature space. These scores are not directly comparable and do not establish which grouping is more useful.

Category shares add to 100%, so they are dependent features. Euclidean K-Means on percentage points is retained as a simple, interpretable educational baseline. The notebook documents this limitation and does not treat category shares as independent measurements.

### Transaction anomaly analysis

The model builds historical medians from prior daily observations, separated by customer, transaction direction, and category. It falls back to category-and-direction history when a customer history is sparse. Isolation Forest models are trained separately for debit and credit transactions using an explicit 1% contamination assumption. The chronological training cutoff is October 19, 2025; evaluation covers 11,114 later transactions, including 63 injected anomalies.

On that held-out period, precision was 0.316, recall 0.587, F1 was 0.411, and average precision was 0.444. The confusion matrix contained 10,971 true negatives, 80 false positives, 26 false negatives, and 37 true positives. The model found more labeled anomalies at the cost of more false positives. These experimental results are limited to the injected labels. An anomaly is unusual relative to the modeled reference data; it is not a fraud determination.

### Statistical analysis

The exploratory customer-level comparison found a lower average debit transaction amount for synthetic student-profile customers than for salaried-individual customers. The mean difference was -₹1,011.59 (student minus salaried individual), with a 95% Welch confidence interval from -₹1,131.42 to -₹891.76, p = 7.77 × 10⁻²⁷, and Hedges' g of -3.43. The test uses one aggregated value per customer rather than treating repeated transactions as independent. It describes these synthetic customer profiles only and should not be generalized to real populations.

### Explainable insights

The rule-based component currently reports measured top-category shares above a 25% threshold, cluster average debit transaction counts, and flagged transaction explanations containing the observed amount and historical ratios. Insight text is generated from those calculations and model outputs; no LLM or unsupported financial advice is used.

## Run the Streamlit dashboard

From the repository root, start the app with:

```powershell
.\venv\Scripts\python.exe -m streamlit run app/app.py
```

The dashboard accepts a CSV with transaction ID, customer ID, date, amount, category, and transaction direction. Select the file and click **Analyze transactions** to begin. Common source header aliases are mapped by the existing schema mapper. It presents results in five plain-language sections, with cash-flow and category charts, monthly spending and customer-profile heatmaps, cluster comparisons, an anomaly timeline, and data-quality feedback. Processing occurs in memory, charts follow the active Streamlit theme, and interpretation limits remain visible throughout. An optional evaluation-label CSV can be supplied with `transaction_id` and `is_injected_anomaly` columns; labels are used only after predictions are made.

## Reproduce and test

Run the complete pytest suite from the repository root:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

The suite covers the existing ingestion and preprocessing functionality and the new segmentation, anomaly, statistics, insight, and integration modules. The ingestion pipeline test writes into a temporary directory so it does not replace the checked-in processed datasets.

## Known limitations

- All analytical findings come from one synthetic year and 180 synthetic customers.
- The anomaly model's contamination setting affects the number of flagged records. The reported metrics are modest and should be treated as experimental.
- Customer/category historical baselines can be weak for sparse groups. Rows without sufficient past observations remain unscored.
- The segmentation metrics are exploratory. Category proportions are compositional, and the mixed feature model gives more total dimensions to category shares than to the three transaction behavior features.
- The Welch comparison is an exploratory test of synthetic customer profiles and does not support conclusions about real customer populations.
- Rule-based insights describe measured values; they are not financial recommendations.
- The deployed Streamlit dashboard analyzes uploaded CSV files; it does not connect to a database or live financial accounts.
- Anomaly flags are experimental statistical signals, not fraud classifications.

## Project completion

The project is complete, with an implemented analytical workflow, documented findings and limitations, tests, and a deployed interactive dashboard. The dashboard is ready to explore using a compatible transaction CSV. This is an educational analysis built on synthetic data; its patterns and experimental anomaly signals do not represent real customers or establish fraud.

## Project structure

```text
Financial_Transaction_Intelligence/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   ├── raw/
│   └── reference/
├── notebooks/
├── src/
│   ├── analytics_pipeline.py
│   ├── anomaly_detection.py
│   ├── customer_segmentation.py
│   ├── data_enrichment.py
│   ├── data_preprocessing.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── ingestion.py
│   ├── insight_generation.py
│   ├── schema_mapper.py
│   ├── schema_validation.py
│   └── statistical_analysis.py
├── tests/
├── run_analytical_workflow.py
├── requirements.txt
└── README.md
```
