from unittest import TestCase
from model_helpers import *
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
__author__ = "ivallesp"


data, target = make_classification(n_samples=1000, n_features=20, random_state=655321)

class TestModelHelpers(TestCase):
    def test_train_model_with_gridsearch(self):
        glm_pipeline = Pipeline([("stdsc", StandardScaler()), ("glm", LogisticRegression())])
        glm_params = {"glm__penalty": ["l1", "l2"],
                      "glm__C": [1.0, 0.95, 0.90, 0.80, 0.60, 0.30, 0.05],
                      "glm__class_weight": ["balanced", None],
                      "glm__random_state": [655321],
                      "glm__n_jobs": [1]}

        model, results = train_model_with_gridsearch(data, target, glm_pipeline, glm_params,
                                                     scoring="neg_log_loss", cv=10, n_jobs=1, verbose=0)

        print model
        print results
