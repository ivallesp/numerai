import os
import pandas as pd
import logging

__author__ = "ivallesp"


def load_train_data(version=None):
    """
    Loads the training dataset
    :param version: version which is intended to be loaded. If None, last version is loaded. (str|None)
    :return: the dataset (pd.Dataframe)
    """
    logger = logging.getLogger(__name__)
    from common_paths import get_raw_data_version_path
    logger.info("Requested training data load")
    filepath = os.path.join(get_raw_data_version_path(version), "numerai_training_data.csv")
    logger.info("Loading train dataset from {0}".format(filepath))
    df = pd.read_csv(filepath, sep=",", encoding="utf-8", index_col=False)
    logger.info("Train dataset loaded successfully")
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert "target" in df.columns
    return df

def load_tournament_data(version=None):
    """
    Loads the tournament dataset
    :param version: version which is intended to be loaded. If None, last version is loaded. (str|None)
    :return: the dataset (pd.Dataframe)
    """
    logger = logging.getLogger(__name__)
    from common_paths import get_raw_data_version_path
    logger.info("Requested tournament data load")
    filepath = os.path.join(get_raw_data_version_path(version), "numerai_tournament_data.csv")
    logger.info("Loading test dataset from {0}".format(filepath))
    df = pd.read_csv(filepath, sep=",", encoding="utf-8", index_col=False, dtype={"t_id": str})
    logger.info("Test dataset loaded successfully")
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert "target" not in df.columns
    return df

def load_numerai_data(version=None):
    """
    Function responsible for calling the train and test loader functions.
    :param version: version to load (str|unicode)
    :return: training_set, test_set (tuple of pd.DataFrames)
    """
    df_train = load_train_data(version)
    df_test = load_tournament_data(version)
    assert ["t_id"] + df_train.columns.tolist() == df_test.columns.tolist() + ["target"]
    return df_train, df_test
