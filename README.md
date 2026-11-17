# SwiftShip Logistics – Customer Intelligence & Prioritization

## Project Overview

This project develops a machine learning-based customer intelligence solution for SwiftShip Logistics.

The solution uses customer demographic and financial attributes to predict income categories and rank customers according to their predicted probability of belonging to the `>50K` income group.

The objective is to demonstrate how machine learning predictions can support data-driven customer segmentation and prioritization.

**Selected Model:** Gradient Boosting Classifier

**Technology Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, FastAPI, Joblib

---

## Business Problem

Businesses need data-driven approaches to identify and prioritize customer segments for targeted engagement.

This project addresses the problem by:

- Analyzing customer demographic and financial information.
- Building and comparing machine learning classification models.
- Predicting the probability of a customer belonging to the `>50K` income category.
- Ranking customers based on predicted probabilities.
- Selecting the top 20% of customers for prioritization.
- Evaluating model performance and potential fairness concerns.
- Exposing predictions through a REST API.

**Important:** The dataset contains income classifications, not actual SwiftShip logistics transactions. Therefore, predicted income is used as a proxy for this assessment and should not be interpreted as actual shipping frequency, customer loyalty, revenue, or profitability.

---

## Dataset Description

The project uses two CSV files:

- `Census_income_train.csv`
- `Census_income_test.csv`

The dataset contains demographic and financial attributes, including:

- Age
- Workclass
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

**Target Variable:** `Income`

- `<=50K` → 0
- `>50K` → 1

### Data Limitations

- The dataset does not contain a time or date column. Therefore, a time-based split and time-series forecasting are not applicable.
- The test dataset does not contain usable income labels. Consequently, test-set accuracy and other supervised evaluation metrics cannot be calculated.
- Customer IDs generated for ranking are row-based identifiers, not original business customer identifiers.

A stratified train-validation split was used for model evaluation.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Inspected dataset dimensions, data types, and distributions.
2. Identified and removed duplicate records.
3. Replaced missing-value markers (`?`) with null values.
4. Separated numerical and categorical features.
5. Excluded `fnlwgt`, which represents a census sampling weight.
6. Applied median imputation and standardization to numerical features.
7. Applied most-frequent imputation and one-hot encoding to categorical features.
8. Used a stratified 80:20 train-validation split.

The preprocessing steps were implemented using Scikit-learn pipelines to maintain consistency between training and prediction.

---

## Model Development

Three classification algorithms were evaluated:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Model Evaluation

The selected Gradient Boosting model achieved the following validation results:

| Metric | Score |
|---|---:|
| Accuracy | 87.48% |
| Precision | 80.29% |
| Recall | 63.65% |
| F1 Score | 71.01% |
| ROC-AUC | 93.13% |

Gradient Boosting was selected based on its overall validation performance.

---

## Customer Prioritization

The trained model generates a probability score for each customer.

Customers are sorted in descending order of predicted probability, and the top 20% are selected for prioritization.

### Validation Results

| Metric | Result |
|---|---:|
| Total validation customers | 6,508 |
| Top 20% selected | 1,302 |
| Actual positive customers in selected group | 1,023 |
| Precision at 20% | 78.57% |
| Recall at 20% | 65.24% |
| Lift at 20% | 3.26 |

The selected group had approximately 3.26 times the positive-class concentration of the overall validation population.

### Test Dataset Ranking

The trained model was applied to the test dataset to generate customer probability scores.

- Total customers ranked: 16,276
- Customers selected in the top 20%: 3,256

The following output files are generated:

- `outputs/all_customer_rankings.csv`
- `outputs/top_20_percent_customers.csv`

---

## Fairness Analysis

A fairness comparison was performed using three Gradient Boosting models:

1. Baseline model using all 13 selected features.
2. Model excluding Race and Sex.
3. Model excluding Race, Sex, and Relationship.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 87.48% | 80.29% | 63.65% | 71.01% | 93.13% |
| Without Race and Sex | 87.29% | 79.66% | 63.46% | 70.64% | 92.66% |
| Without Race, Sex, and Relationship | 87.14% | 79.64% | 62.63% | 70.12% | 92.53% |

### Key Observations

- Removing Race and Sex resulted in a small decrease in overall predictive performance.
- Differences in subgroup selection rates remained after removing protected attributes.
- Removing Relationship as an additional feature did not eliminate the observed disparities.
- Smaller demographic groups had limited sample sizes, so their metrics should be interpreted cautiously.

Removing protected attributes alone does not guarantee fairness because other features may still be associated with protected characteristics.

This analysis is diagnostic and does not establish that the model is fair or suitable for real-world customer targeting. Further fairness evaluation, mitigation, and legal and ethical review would be required before deployment.

---

## REST API – FastAPI

A REST API was developed using FastAPI to make the trained model accessible for customer prediction and prioritization.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome endpoint |
| GET | `/health` | Checks API and model availability |
| POST | `/predict` | Predicts an individual customer's income category |
| POST | `/prioritize` | Ranks multiple customers and selects the requested percentage |

### API Features

- Individual customer prediction.
- Batch customer prioritization.
- Configurable selection percentage.
- Probability-based customer ranking.
- Duplicate customer ID validation.
- Empty customer list validation.
- Selection percentage validation.
- Handling of missing values.
- Handling of previously unseen categorical values.

### Run the API Locally

Clone the repository:

```bash
git clone https://github.com/dayaram77/DF-Analytics-2026-Assessment-SableDayaram.git
```

Navigate to the project directory:

```bash
cd DF-Analytics-2026-Assessment-SableDayaram
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

### API Documentation

Once the server is running, access the interactive Swagger documentation:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

---

## Project Structure

```text
DF-Analytics-2026-Assessment-SableDayaram/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── gradient_boosting_pipeline.pkl
│
├── notebooks/
│   └── SwiftShip_Customer_Intelligence.ipynb
│
├── outputs/
│   ├── all_customer_rankings.csv
│   └── top_20_percent_customers.csv
│
├── screenshots/
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **API Development:** FastAPI
- **API Server:** Uvicorn
- **Development Environment:** Jupyter Notebook, VS Code

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, model development, evaluation, customer ranking, fairness analysis, and REST API development.

The Gradient Boosting model achieved 87.48% validation accuracy and 93.13% ROC-AUC. The customer prioritization analysis achieved 78.57% precision among the top 20% of validation customers.

The trained model was also integrated into a FastAPI application, enabling individual predictions and batch customer prioritization.

The solution is a demonstration using census income data. Further validation with actual logistics customer data would be necessary before using it for operational business decisions.

---

## Author

**Sable Dayaram**

GitHub: [dayaram77](https://github.com/dayaram77)
