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
    fbeta_score, 
    make_scorer
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

f2_scorer = make_scorer(fbeta_score, beta=2)
THRESHOLDS = (0.35, 0.40, 0.45, 0.50)

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


def build_searches(cv: StratifiedKFold, random_state: int = 123) -> dict[str, Any]:
    """Construct searches whose estimators include every learned preprocessing step."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": f2_scorer,
        "roc_auc": "roc_auc",
    }
    common = {"cv": cv, "scoring": scoring, "refit": "f2", "n_jobs": -1}

    searches: dict[str, Any] = {
        "Logistic Regression": GridSearchCV(
            make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
            {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__l1_ratio": [0, 1],   # 0 = L2, 1 = L1,
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
    result["cv_f2_at_0_50_mean"] = float(search.cv_results_["mean_test_f2"][index])
    result["cv_f2_at_0_50_std"] = float(search.cv_results_["std_test_f2"][index])
    return result


def tune_threshold_cv(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> tuple[float, dict[float, float], dict[float, float]]:
    """Select one probability threshold using out-of-fold training predictions.

    Each threshold is evaluated on every validation fold. The selected threshold
    is the single candidate with the highest mean F2 across all folds; we do not
    select a different threshold for each fold.
    """
    fold_scores = {threshold: [] for threshold in thresholds}

    # Fit a fresh copy on each fold. This keeps feature engineering and
    # preprocessing inside the fold and prevents validation-fold leakage.
    for train_indices, validation_indices in cv.split(X, y):
        fold_model = clone(estimator).fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
        )

        # Use probabilities because the threshold is an explicit business decision.
        # SVM probabilities can differ from SVM.predict(), so this
        # is intentionally separate from GridSearchCV's default predict rule.
        probabilities = fold_model.predict_proba(X.iloc[validation_indices])[:, 1]
        y_validation = y.iloc[validation_indices]

        # Score every candidate threshold on the same validation fold.
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            fold_scores[threshold].append(
                float(fbeta_score(y_validation, predictions, beta=2))
            )

    # Average the fold scores separately for each threshold. These are the
    # cross-validated scores used to choose the one deployable threshold.
    mean_scores = {
        threshold: float(np.mean(scores))
        for threshold, scores in fold_scores.items()
    }
    std_scores = {
        threshold: float(np.std(scores, ddof=1))
        for threshold, scores in fold_scores.items()
    }

    # Choose the threshold with the best mean F2 across folds.
    best_threshold = max(mean_scores, key=mean_scores.get)
    return best_threshold, mean_scores, std_scores


def select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 123,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    """Tune and select the model and threshold using training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    searches = build_searches(cv, random_state)
    summaries = []

    for name, search in searches.items():
        # First tune the classifier hyperparameters using only the training partition.
        # GridSearchCV/RandomizedSearchCV refit the best pipeline.
        search.fit(X_train, y_train)
        summary = summarize_search(name, search)

        # Next, use fresh CV fits of that model's best hyperparameters to test
        # the candidate probability thresholds. Every model gets its own
        # threshold, so the comparison is fair for the chosen operating rule.
        threshold, threshold_scores, threshold_stds = tune_threshold_cv(
            search.best_estimator_, X_train, y_train, cv
        )
        summary["selected_threshold"] = threshold

        # This is the model's mean F2 at its best single threshold. It is the
        # metric used below to compare the candidate models.
        summary["cv_f2_mean"] = threshold_scores[threshold]
        summary["cv_f2_std"] = threshold_stds[threshold]

        # Keep the result for every candidate threshold so the trade-off can
        # be inspected later instead of reporting only the winning threshold.
        for candidate_threshold, score in threshold_scores.items():
            summary[f"cv_f2_threshold_{candidate_threshold:.2f}"] = score
            summary[f"cv_f2_threshold_{candidate_threshold:.2f}_std"] = threshold_stds[candidate_threshold]
        summaries.append(summary)

    # Select the model with the highest cross-validated F2 after threshold
    # tuning. The holdout set has not been used at this point.
    comparison = pd.DataFrame(summaries).sort_values("cv_f2_mean", ascending=False).reset_index(drop=True)
    selected_name = str(comparison.loc[0, "model"])

    # Refit a clean copy of the selected pipeline on all training rows. The
    # selected threshold is stored separately and applied to probabilities
    # during holdout evaluation; it is not learned from the holdout set.
    final_model = clone(searches[selected_name].best_estimator_).fit(X_train, y_train)
    metadata = {
        "selection_metric": "cv_f2_mean",
        "selected_model": selected_name,
        "selected_threshold": float(comparison.loc[0, "selected_threshold"]),
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "best_params": searches[selected_name].best_params_,
    }
    return final_model, comparison, metadata


def evaluate_holdout(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate once after model selection; this function is never used by the searches."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "f2": float(fbeta_score(y_test, predictions, beta=2)),
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
