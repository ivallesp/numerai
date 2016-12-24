__author__ = "ivallesp"
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV

def train_model_with_gridsearch(X, y, estimator, param_grid, scoring="neg_log_loss", cv=10, n_jobs=1, verbose=0):
    """
    Trains a model using gridsearch and returns the best model trained with all the data and the results dictionary.
    :param estimator: sklearn-like model (sklearn object)
    :param param_grid: dictionary of parameters from which the grid is going to be built (dict)
    :param scoring: either a scoring function or a string (func|str)
    :param cv: either a cv object or an integer indicating the number of folds (sklearn obj|int)
    :param n_jobs: number of jobs (int)
    :param verbose: verbosity level (the greater the number, the more verbose is the model) (int)
    :return: model, results dict (sklearn model|dictionary)
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested gridsearch process")
    logger.info("Training the requested estimator using gridsearch")

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs,
                               refit=True, verbose=verbose, return_train_score=True)

    trained_model = grid_search.fit(X, y)
    results = trained_model.cv_results_
    logger.info("Gridsearch trained successfully. Generating results JSON")
    # Fix dictionary for allowing a further JSON conversion
    for key in results:
        if type(results[key]) in [np.ndarray, np.array, np.ma.core.MaskedArray]:
            results[key] = results[key].tolist()
    return trained_model, results

