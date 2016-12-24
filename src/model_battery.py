import json
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

__author__ = "ivallesp"


def get_general_glm(n_jobs_model=-1, random_seed=655321):
    """
    Logistic Regression gridsearch (28 parameter sets)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested GLM general grid search for classification. 28 parameters sets retrieved")
    from sklearn.linear_model import LogisticRegression
    model = Pipeline([("stdsc", StandardScaler()), ("glm", LogisticRegression())])
    params = {"glm__penalty": ["l1", "l2"],
              "glm__C": [1.0, 0.95, 0.90, 0.80, 0.60, 0.30, 0.05],
              "glm__class_weight": ["balanced", None],
              "glm__random_state": [random_seed],
              "glm__n_jobs": [n_jobs_model]}
    return model, params


def get_general_glmnet(n_jobs_model=-1, random_seed=655321):
    """
    ElasticNet gridsearch for classification (64 parameter sets)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested GLMNET general grid search for classification. 64 parameters sets retrieved")
    from sklearn.linear_model import SGDClassifier
    model = Pipeline([("stdsc", StandardScaler()), ("glmnet", SGDClassifier())])
    params = {"glmnet__loss": ["log"],
              "glmnet__alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2],
              "glmnet__l1_ratio": [1.0, 0.99, 0.95, 0.90, 0.80, 0.50, 0.2, 0.1],
              "glmnet__random_state": [random_seed],
              "glmnet__n_jobs": [n_jobs_model],
              "glmnet__n_iter": [100]}
    return model, params


def get_general_tree(n_jobs_model=-1, random_seed=655321):
    """
    DecisionTreeClassifier gridsearch for classification (72 parameter sets)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested TREE general grid search for classification. 72 parameters sets retrieved")
    from sklearn.tree import DecisionTreeClassifier
    model = Pipeline([("tree", DecisionTreeClassifier())])
    params = {"tree__criterion": ["gini", "entropy"],
              "tree__max_features": ["sqrt", "log2", None],
              "tree__max_depth": [None, 5, 10, 20],
              "tree__min_samples_split": [2, 10, 100],
              "tree__class_weight": ["balanced", None],
              "tree__random_state": [random_seed]}
    return model, params


def get_general_knn(n_jobs_model=-1, random_seed=655321):
    """
    KNN gridsearch for classification (40 parameter sets)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested KNN general grid search for classification. 40 parameters sets retrieved")
    from sklearn.neighbors import KNeighborsClassifier
    model = Pipeline([("stdsc", StandardScaler()), ("knn", KNeighborsClassifier())])
    params = {"knn__n_neighbors": [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
              "knn__weights": ["uniform", "distance"],
              "knn__p": [1, 2],
              "knn__n_jobs": [n_jobs_model]}
    return model, params


def get_general_svc(n_jobs_model=-1, random_seed=655321):
    """
    SVM gridsearch for classification (60 parameter trials)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested SVC general grid search for classification. 60 parameters sets retrieved")
    from sklearn.svm import SVC
    model = Pipeline([("stdsc", StandardScaler()), ("svc", SVC())])
    params = [{"svc__kernel": ["rbf"],
               "svc__C": [.01, 0.1, 1.0, 10.0, 100.0],
               "svc__gamma": ["auto", 0.01, 0.001, 0.0001],
               "svc__class_weight": ["balanced"],
               "svc__random_state": [random_seed],
               "svc__probability": [True]},

              {"svc__kernel": ["linear"],
               "svc__C": [.01, 0.1, 1.0, 10.0],
               "svc__random_state": [random_seed],
               "svc__probability": [True]},

              {"svc__kernel": ["poly"],
               "svc__C": [.01, 0.1, 1.0, 10.0],
               "svc__degree": [2, 3],
               "svc__gamma": ["auto", 0.01, 0.001, 0.0001],
               "svc__random_state": [random_seed],
               "svc__probability": [True]}]
    return model, params


def get_general_rf(n_jobs_model=-1, random_seed=655321):
    """
    RandomForest gridsearch for classification (24 parameter trials)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested RF general grid search for classification. 24 parameters sets retrieved")
    from sklearn.ensemble import RandomForestClassifier
    model = Pipeline([("rf", RandomForestClassifier())])
    params = [{"rf__n_estimators": [2000],
               "rf__criterion": ["gini", "entropy"],
               "rf__max_features": ["auto", "sqrt", "log2", None],
               "rf__max_depth": [None, 5, 10],
               "rf__random_state": [random_seed],
               "rf__n_jobs": [n_jobs_model],
               "rf__class_weight": ["balanced"]}]
    return model, params


def get_general_et(n_jobs_model=-1, random_seed=655321):
    """
    ExtraTrees gridsearch for classification (48 parameter trials)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested ET general grid search for classification. 48 parameters sets retrieved")
    from sklearn.ensemble import ExtraTreesClassifier
    model = Pipeline([("et", ExtraTreesClassifier())])
    params = [{"et__n_estimators": [2000],
               "et__criterion": ["gini", "entropy"],
               "et__max_features": ["auto", "sqrt", "log2", None],
               "et__max_depth": [None, 5, 10],
               "et__random_state": [random_seed],
               "et__n_jobs": [n_jobs_model],
               "et__class_weight": ["balanced", "balanced_subsample"]}]
    return model, params


def get_general_nb(n_jobs_model=-1, random_seed=655321):
    """
    Naive Bayes gridsearch for classification (1 parameter trial)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested NB general grid search for classification. 1 parameters set retrieved")
    from sklearn.naive_bayes import GaussianNB
    model = Pipeline([("stdsc", StandardScaler()), ("nb", GaussianNB())])
    params = [{}]
    return model, params


def get_general_mlp(n_jobs_model=-1, random_seed=655321):
    """
    Multi Layer Perceptron gridsearch for classification (64 parameter sets)
    :param n_jobs_model: number of jobs (model passed as a model parameter (int)
    :param random_seed: random seed of the model to be tested (int)
    :return: model, parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested MLP general grid search for classification. 64 parameters sets retrieved")
    from sklearn.neural_network import MLPClassifier
    model = Pipeline([("stdsc", StandardScaler()), ("mlp", MLPClassifier())])
    params = {"mlp__hidden_layer_sizes": [(500, 250), (200, 100), (100, 20), (100, 50), (100, 100), (50, 50), (50, 20)],
              "mlp__activation": ["logistic", "tanh", "relu"],
              "mlp__solver": ["adam"],
              "mlp__random_state": [random_seed]}
    return model, params
