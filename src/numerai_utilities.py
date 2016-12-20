import json
import os
import datetime
import pandas as pd
__author__ = "ivallesp"


def get_numerai_api_access():
    """
    Returns an authenticated object for accessing comfortably to the numerai API.
    :return: None (void)
    """
    from NumerAPI.numerapi import NumerAPI
    from src.common_paths import get_numerai_secrets_path
    secrets = json.load(open(get_numerai_secrets_path()))
    n_api = NumerAPI(email=secrets["email"], password=secrets["password"])
    return n_api


def download_last_numerai_data(version_name= None):
    """
    Downloads the last version of the dataset from the numerai API, stores it in the data/raw folder and
    updates the settings.json file with the name of the version downloaded.
    :version_name: If it is specified, it will be used as the new version name, instead of building it using the date.
    (str|none)
    :return: None (void)
    """
    from src.common_paths import get_raw_data_path
    n_api = get_numerai_api_access()
    if not version_name:
        version_name = datetime.datetime.now().strftime("%Y%m%d")
    path = os.path.normpath(os.path.join(get_raw_data_path(), version_name))
    if not os.path.exists(path):
        os.makedirs(path)
    n_api.download_current_dataset(dest_path=path)
    with open("settings.json", 'rb') as f:
        settings = json.load(f)
    settings["last_data_version"] = version_name
    with open("settings.json", 'wb') as f:
        f.write(json.dumps(settings, sort_keys=True, indent=4, separators=(',', ': ')))

def build_submission(version, indices, probs, alias, replace=False):
    """
    Given a set of indices and probs, builds a submission ad stores it in the submissions folder.
    :param indices: Indices to be put in the submission
    :param probs: probabilities predicted
    :param alias: unique alias of the submission. Used to build the csv name in order to identify the submission among
    the other ones
    :return: None (void)
    """
    from src.common_paths import get_submissions_version_path
    if not type(indices) == list:
        indices = list(indices)
    if not type(probs) == list:
        probs = list(probs)
    assert len(indices) == len(probs)
    path = os.path.join(get_submissions_version_path(version), "submission_{0}.csv".format(alias))
    if not replace:
        assert not os.path.exists(path)
    df = pd.DataFrame({"t_id": indices, "probability": probs})[["t_id", "probability"]]
    df.to_csv(path, sep=",", index=False, decimal=".", encoding="utf-8")

def upload_submission(version, alias):
    """
    Given a submission, uploads it and returns the score obtained.
    :param version: Indices to be put in the submission
    :param alias: Unique alias of the submission. Used to build the csv name in order to identify the submission among
    the other ones
    :return:
    """
    from src.common_paths import get_submissions_version_path, get_project_path
    with open(os.path.join(get_project_path(), "NumerAPI", "secrets.json")) as f:
        secrets = json.load(f)
    username = secrets["username"]
    path = os.path.join(get_submissions_version_path(version), "submission_{0}.csv".format(alias))
    assert os.path.exists(path)
    n_api = get_numerai_api_access()
    status = n_api.upload_prediction(path)
    score = n_api.get_scores(username)[0][0]
    return (status, score)
