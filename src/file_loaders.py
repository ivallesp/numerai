import os
import pandas as pd

__author__ = "ivallesp"


def load_train_data(version=None):
    """
    Loads the training dataset
    :param version: version which is intended to be loaded. If None, last version is loaded. (str|None)
    :return: the dataset (pd.Dataframe)
    """
    from common_paths import get_raw_data_version_path
    path = get_raw_data_version_path(version)
    df = pd.read_csv(os.path.join(path, "numerai_training_data.csv"), sep=",", encoding="utf-8", index_col=False)
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
    from common_paths import get_raw_data_version_path
    path = get_raw_data_version_path(version)
    df = pd.read_csv(os.path.join(path, "numerai_tournament_data.csv"), sep=",", encoding="utf-8", index_col=False,
                     dtype={"t_id": str})
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert "target" not in df.columns
    return df

def load_numerai_data(version=None):
    df_train = load_train_data(version)
    df_test = load_tournament_data(version)
    assert ["t_id"] + df_train.columns.tolist() == df_test.columns.tolist() + ["target"]
    return df_train, df_test