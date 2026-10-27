# Bank Customer Churn Prediction with Explainable AI

## Project overview

This project builds an end-to-end machine-learning workflow for predicting whether a bank customer will leave the bank. The target variable is `Exited`:

```text
0 = retained customer
1 = churned customer
```

The business objective is proactive retention: identify customers with elevated churn risk, prioritize outreach, estimate the balance exposed to churn, and provide explanations that can support retention decisions.

The dataset contains 10,000 customer records and includes demographic, financial, product, activity, and tenure variables. The churn class is the minority class, so accuracy alone is not sufficient for model selection.

## Business use

The model output is designed to support three decisions:

- Which customers should be prioritized for retention outreach?
- Which high-risk customers represent the greatest balance or revenue exposure?
- What customer characteristics contribute to a prediction, and what hypothetical changes could move a customer to the retained class?

The exported predictions can be loaded into Power BI to create a dynamic risk dashboard, an operational customer hit list, and revenue-at-risk measures.

## Leakage-safe modelling workflow

The authoritative workflow is [`notebook/Leakage_Safe_Modeling.ipynb`](notebook/Leakage_Safe_Modeling.ipynb). The non-interactive entry point is [`run_modeling.py`](run_modeling.py).

The workflow:

1. Splits the data into an 80% training partition and a sealed 20% holdout partition using stratification.
2. Applies feature engineering, preprocessing, and classification inside one scikit-learn pipeline.
3. Compares Logistic Regression, SVM, Random Forest, and XGBoost.
4. Tunes model hyperparameters using five-fold `StratifiedKFold` cross-validation on the training partition only.
5. Evaluates candidate probability thresholds of `0.35`, `0.40`, `0.45`, and `0.50` using out-of-fold validation predictions.
6. Selects one global threshold for each model based on its highest mean cross-validated F2 score.
7. Selects the overall model using the tuned `cv_f2_mean`.
8. Refits the selected pipeline on all training rows.
9. Evaluates the selected model and threshold once on the untouched holdout set.

The holdout is never used to fit feature engineering, scaling, encoding, hyperparameters, or the threshold. It is used only for the final performance report.

### Why F2 and a tuned threshold?

Churn is the positive class, and missing a real churner can be more expensive than contacting a customer who would have stayed. F2 gives recall more weight than precision:

```text
F2 emphasizes finding as many real churners as possible.
```

The probability threshold controls how aggressively customers are classified as high risk:

```text
lower threshold → more predicted churners, higher recall, more false positives
higher threshold → fewer predicted churners, lower recall, fewer false positives
```

For each model, the code calculates F2 at every candidate threshold on every validation fold, averages each threshold across folds, and chooses one deployable threshold with the highest mean F2. It does not choose a different threshold for every fold.

`cv_f2_mean` is the mean F2 at the selected threshold and is the primary comparison metric. `cv_f2_std` measures variation across folds. The report also stores the F2 score for each candidate threshold so the precision-recall trade-off can be inspected.

The current implementation first selects the best hyperparameter configuration for each model using the search scorer, then performs probability-threshold tuning for that selected configuration. The threshold comparison uses `predict_proba()` consistently. This distinction is important for SVM, because `SVC.predict()` uses its decision function while `SVC.predict_proba()` uses calibrated probabilities.

## Feature engineering and preprocessing

The custom [`ChurnFeatureEngineer`](src/churn_workflow.py) transformer:

- Removes technical identifiers: `RowNumber`, `CustomerId`, and `Surname`.
- Excludes `HasCrCard` from the model features.
- Creates:
  - `Balance_per_Age`
  - `Balance_per_Salary`
  - `Tenure_per_Age`
  - `CreditScore_per_Age`

The preprocessing step uses a `ColumnTransformer`:

- Numerical variables are standardized with `StandardScaler`.
- Categorical variables (`Geography`, `Gender`, and `IsActiveMember`) are one-hot encoded.
- Unknown categories are ignored safely during prediction.
- Any columns not explicitly selected are dropped.

Because all of these steps are inside the pipeline, calling:

```python
pipeline.predict(X_holdout)
```

automatically applies the same feature engineering and fitted preprocessing to the holdout data before generating predictions. The holdout is transformed, but it is not fitted.

## Explainable AI

### SHAP: why did the model make this prediction?

SHAP is run after model selection. The fitted pipeline transforms the holdout data, and `transformed_frame()` preserves the encoded feature names for explanation.

SHAP is used to:

- Identify the most influential features globally.
- Show whether a feature increases or decreases predicted churn risk.
- Investigate nonlinear relationships in the selected tree model.
- Support communication of model behavior to business stakeholders.

SHAP explanations describe the model’s learned associations. They do not prove that changing a feature will cause a customer to stay.

### DiCE: what hypothetical changes could alter the prediction?

DiCE generates Diverse Counterfactual Explanations for an individual high-risk customer. The workflow:

1. Filters customers whose predicted churn probability is at or above the selected threshold.
2. Sends a high-risk customer to DiCE.
3. Allows only selected actionable features to vary, such as `NumOfProducts`, `IsActiveMember`, and `CreditScore`.
4. Requests counterfactuals with the opposite predicted class.
5. Passes the generated counterfactual rows through the original fitted pipeline again.
6. Displays the verified probability and prediction in a comparison table.

The cross-check table includes:

```python
[
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "IsActiveMember",
    "EstimatedSalary", "New_Prediction", "New_Churn_Risk_%"
]
```

Successful retention scenarios should have:

```text
New_Prediction = 0
```

The counterfactual rows are hypothetical scenarios, not new training observations. They must not be added back into the training dataset. They are model-consistency checks and possible intervention hypotheses, not causal guarantees.

## Power BI output

The final holdout predictions are exported to:

```text
reports/bank_churn_predictions.csv
```

The file contains the original customer information together with:

- `Predicted_Churn`
- `Churn_Risk_Percentage`

The selected cross-validated threshold is used to create `Predicted_Churn`.

The DAX measures in [`dax-measures/measures.md`](dax-measures/measures.md) support:

- Total customer count.
- Churn rate.
- Critical flight-risk count.
- Revenue at risk based on a dynamic risk threshold.
- High-balance, high-risk customer highlighting.

## Reports and reproducibility

Run the complete non-interactive workflow with:

```bash
python run_modeling.py
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

Generated files include:

- `reports/cv_model_comparison.csv`
- `reports/holdout_results.json`
- `reports/bank_churn_predictions.csv`

The original [`notebook/Churn_Prediction.ipynb`](notebook/Churn_Prediction.ipynb) is retained as exploratory project history. The leakage-safe notebook and [`src/churn_workflow.py`](src/churn_workflow.py) are the current reference implementation.

## Technologies

- Python
- Pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- DiCE
- Matplotlib and Seaborn
- Power BI

![Power BI Dashboard Screenshot](figures/powerbi_dashboard_screenshot.png)
