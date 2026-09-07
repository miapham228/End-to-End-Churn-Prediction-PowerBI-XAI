import unittest

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.utils.validation import check_is_fitted

from src.churn_workflow import ChurnFeatureEngineer, make_pipeline, transformed_frame


class ChurnWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv("data/Churn_Modelling.csv")
        cls.X = data.drop(columns="Exited")
        cls.y = data["Exited"]

    def test_holdout_indices_are_disjoint(self):
        X_train, X_holdout, _, _ = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        self.assertTrue(set(X_train.index).isdisjoint(X_holdout.index))
        self.assertEqual(len(X_train), 8000)
        self.assertEqual(len(X_holdout), 2000)

    def test_pipeline_starts_unfitted_and_contains_all_preprocessing(self):
        pipeline = make_pipeline(RandomForestClassifier(n_estimators=5, random_state=42))
        self.assertEqual(list(pipeline.named_steps), ["feature_engineering", "preprocessor", "classifier"])
        with self.assertRaises(Exception):
            check_is_fitted(pipeline.named_steps["preprocessor"])

    def test_cross_validation_fits_independent_pipeline_clones(self):
        pipeline = make_pipeline(RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42))
        sample_X, sample_y = self.X.iloc[:500], self.y.iloc[:500]
        results = cross_validate(
            pipeline,
            sample_X,
            sample_y,
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            return_estimator=True,
        )
        preprocessors = [est.named_steps["preprocessor"] for est in results["estimator"]]
        self.assertEqual(len({id(item) for item in preprocessors}), 3)
        with self.assertRaises(Exception):
            check_is_fitted(pipeline.named_steps["preprocessor"])

    def test_feature_engineering_does_not_mutate_raw_data(self):
        raw = self.X.iloc[:2].copy()
        original_columns = raw.columns.tolist()
        transformed = ChurnFeatureEngineer().fit_transform(raw)
        self.assertEqual(raw.columns.tolist(), original_columns)
        self.assertIn("Balance_per_Age", transformed.columns)
        self.assertNotIn("CustomerId", transformed.columns)

    def test_transformed_frame_maps_encoded_feature_names(self):
        pipeline = make_pipeline(RandomForestClassifier(n_estimators=5, random_state=42))
        pipeline.fit(self.X.iloc[:500], self.y.iloc[:500])
        frame = transformed_frame(pipeline, self.X.iloc[500:510])
        self.assertEqual(frame.shape[0], 10)
        self.assertIn("num__Age", frame.columns)
        self.assertTrue(any(name.startswith("cat__Geography_") for name in frame.columns))


if __name__ == "__main__":
    unittest.main()
