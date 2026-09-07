"""Run leakage-safe model selection and one-time holdout evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.churn_workflow import TARGET, evaluate_holdout, select_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Churn_Modelling.csv"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    final_model, comparison, metadata = select_model(X_train, y_train)
    # First and only use of X_holdout/y_holdout in the modelling workflow.
    holdout = evaluate_holdout(final_model, X_holdout, y_holdout)

    REPORT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(REPORT_DIR / "cv_model_comparison.csv", index=False)
    payload = {
        **metadata,
        "training_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "holdout_evaluation": holdout,
    }
    (REPORT_DIR / "holdout_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    predictions = data.loc[X_holdout.index].copy()
    predictions["Predicted_Churn"] = final_model.predict(X_holdout)
    predictions["Churn_Risk_Percentage"] = final_model.predict_proba(X_holdout)[:, 1]
    predictions.to_csv(REPORT_DIR / "bank_churn_predictions.csv", index=False)

    print(comparison.drop(columns="best_params").to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
