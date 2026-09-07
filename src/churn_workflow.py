"""Leakage-safe model selection and holdout evaluation for bank churn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
EXCLUDED_FEATURES = ["HasCrCard"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "IsActiveMember"]
NUMERIC_FEATURES = [
    "Balance", "Tenure", "Age", "CreditScore", "NumOfProducts",
    "EstimatedSalary", "Balance_per_Age", "Balance_per_Salary",
    "Tenure_per_Age", "CreditScore_per_Age",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stateless domain features and remove non-model columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ChurnFeatureEngineer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame = frame.drop(columns=ID_COLUMNS + EXCLUDED_FEATURES, errors="ignore")
        frame["Balance_per_Age"] = frame["Balance"] / (frame["Age"] + 1)
        frame["Balance_per_Salary"] = frame["Balance"] / (frame["EstimatedSalary"] + 1)
        frame["Tenure_per_Age"] = frame["Tenure"] / (frame["Age"] + 1)
        frame["CreditScore_per_Age"] = frame["CreditScore"] / (frame["Age"] + 1)
        return frame


def make_preprocessor() -> ColumnTransformer:
    """Return a fresh, unfitted preprocessing graph for one pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_searches(cv: StratifiedKFold, random_state: int = 42) -> dict[str, Any]:
    """Construct searches whose estimators include every learned preprocessing step."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    common = {"cv": cv, "scoring": scoring, "refit": "f1", "n_jobs": -1}

    searches: dict[str, Any] = {
        "Logistic Regression": GridSearchCV(
            make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
            {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver": ["liblinear"],
            },
            **common,
        ),
        "SVM": GridSearchCV(
            make_pipeline(SVC(class_weight="balanced", probability=True, random_state=random_state)),
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": [0.1, 0.01, 0.001],
            },
            **common,
        ),
        "Random Forest": RandomizedSearchCV(
            make_pipeline(RandomForestClassifier(random_state=random_state)),
            {
                "classifier__n_estimators": [100, 200, 300, 500],
                "classifier__max_depth": [5, 8, 12, 15, None],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 5, 10],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__class_weight": ["balanced", "balanced_subsample"],
            },
            n_iter=30,
            random_state=random_state,
            **common,
        ),
        "XGBoost": RandomizedSearchCV(
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                )
            ),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__scale_pos_weight": [1.0, 2.0, 3.0, 4.0],
            },
            n_iter=20,
            random_state=random_state,
            **common,
        ),
    }
    return searches


def summarize_search(name: str, search: Any) -> dict[str, Any]:
    """Summarize the winning candidate using its held-out CV fold scores."""
    index = search.best_index_
    result: dict[str, Any] = {"model": name, "best_params": search.best_params_}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        result[f"cv_{metric}_mean"] = float(search.cv_results_[f"mean_test_{metric}"][index])
        result[f"cv_{metric}_std"] = float(search.cv_results_[f"std_test_{metric}"][index])
    return result


def select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    """Tune and select exclusively with stratified CV on training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    searches = build_searches(cv, random_state)
    summaries = []
    for name, search in searches.items():
        search.fit(X_train, y_train)
        summaries.append(summarize_search(name, search))

    comparison = pd.DataFrame(summaries).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)
    selected_name = str(comparison.loc[0, "model"])
    # Explicitly refit a clean clone on all training rows after selection.
    final_model = clone(searches[selected_name].best_estimator_).fit(X_train, y_train)
    metadata = {
        "selection_metric": "cv_f1_mean",
        "selected_model": selected_name,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "best_params": searches[selected_name].best_params_,
    }
    return final_model, comparison, metadata


def evaluate_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate once after model selection; this function is never used by the searches."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def transformed_frame(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw rows with the fitted pipeline and retain encoded feature names for SHAP."""
    engineered = model.named_steps["feature_engineering"].transform(X)
    preprocessor = model.named_steps["preprocessor"]
    values = preprocessor.transform(engineered)
    return pd.DataFrame(values, columns=preprocessor.get_feature_names_out(), index=X.index)
