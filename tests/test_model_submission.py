from unittest import TestCase

import os


__author__ = "ivallesp"

class TestModelSubmission(TestCase):
    def test_get_numerai_api_access(self):
        from numerai_utilities import build_submission, upload_submission
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from file_loaders import load_numerai_data
        version, alias = "trial", "trial_glm"

        df_train, df_test = load_numerai_data()
        x_train = df_train[df_train.columns[df_train.columns != "target"]]
        y_train = df_train["target"]

        x_test = df_test[x_train.columns]
        test_id = df_test["t_id"]

        p = Pipeline([("scaler", StandardScaler()), ("glm", LogisticRegression())])

        p.fit(x_train, y_train)
        test_preds = p.predict_proba(x_test)[:,1]

        build_submission(version=version, indices=test_id, probs=test_preds, alias=alias, replace=True)

        status, score = upload_submission(version=version, alias=alias)

        assert status == 200
        assert 0 < score < 1
