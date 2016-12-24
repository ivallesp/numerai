import json
import os
import datetime
import logging
import pandas as pd

__author__ = "ivallesp"


def get_numerai_api_access():
    """
    Returns an authenticated object for accessing comfortably to the numerai API.
    :return: None (void)
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested Numer.ai api access")
    from NumerAPI.numerapi import NumerAPI
    from src.common_paths import get_numerai_secrets_path
    secrets = json.load(open(get_numerai_secrets_path()))
    n_api = NumerAPI(email=secrets["email"], password=secrets["password"])
    logger.info("Authenticating with Numer.ai API...")
    _, _, _, status = n_api.login()
    logger.info("Numer.ai API returned status {0}".format(status))
    return n_api


def download_last_numerai_data(version_name=None):
    """
    Downloads the last version of the dataset from the numerai API, stores it in the data/raw folder and
    updates the settings.json file with the name of the version downloaded.
    :version_name: If it is specified, it will be used as the new version name, instead of building it using the date.
    (str|none)
    :return: None (void)
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested numer.ai data download")
    from src.common_paths import get_raw_data_path
    n_api = get_numerai_api_access()
    if not version_name:
        version_name = datetime.datetime.now().strftime("%Y%m%d")
    path = os.path.normpath(os.path.join(get_raw_data_path(), version_name))
    if not os.path.exists(path):
        os.makedirs(path)
    logger.info('Downloading data with version name: {0}'.format(version_name))
    status = n_api.download_current_dataset(dest_path=path)
    logger.info("Numer.ai API returned status {0}".format(status))
    logger.info("New data sets stored in {0}".format(path))
    with open("settings.json", 'rb') as f:
        settings = json.load(f)
    settings["last_data_version"] = version_name
    with open("settings.json", 'wb') as f:
        f.write(json.dumps(settings, sort_keys=True, indent=4, separators=(',', ': ')))
    return status

def build_submission(version, indices, probs, alias, replace=False):
    """
    Given a set of indices and probs, builds a submission ad stores it in the submissions folder.
    :param version: Version of the data used to generate the submission (str|unicode)
    :param indices: Indices to be put in the submission (list|np.array|pd.Series)
    :param probs: probabilities predicted (list|np.array|pd.Series)
    :param alias: unique alias of the submission. Used to build the csv name in order to identify the submission among
    the other ones (str|unicode)
    :return: None (void)
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested numer.ai submission build: {0}, {1}".format(version, alias))
    from src.common_paths import get_submissions_version_path
    if not type(indices) == list:
        indices = list(indices)
    if not type(probs) == list:
        probs = list(probs)
    assert len(indices) == len(probs)
    path = os.path.join(get_submissions_version_path(version), "submission_{0}.csv".format(alias))
    if not replace:
        assert not os.path.exists(path)
    logger.info("Storing submission")
    df = pd.DataFrame({"t_id": indices, "probability": probs})[["t_id", "probability"]]
    df.to_csv(path, sep=",", index=False, decimal=".", encoding="utf-8")
    logger.info("Submission stored successfully in: {0}".format(path))


def upload_submission(version, alias):
    """
    Given a submission, uploads it and returns the score obtained.
    :param version: Indices to be put in the submission (str|unicode)
    :param alias: Unique alias of the submission. Used to build the csv name in order to identify the submission among
    the other ones (str|unicode)
    :return: status, score
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested numer.ai data submission upload: {0}, {1}".format(version, alias))
    from src.common_paths import get_submissions_version_path, get_project_path
    with open(os.path.join(get_project_path(), "NumerAPI", "secrets.json")) as f:
        secrets = json.load(f)
    username = secrets["username"]
    path = os.path.join(get_submissions_version_path(version), "submission_{0}.csv".format(alias))
    logger.info("Using submission path: {0}".format(path))
    assert os.path.exists(path)
    n_api = get_numerai_api_access()
    logger.info("Uploading submission...")
    status = n_api.upload_prediction(path)
    logger.info("Submission uploader returned status {0}".format(status))
    logger.info("Requesting score to the Numer.ai API".format(status))
    score = n_api.get_scores(username)[0][0]
    logger.info("Score obtained: {0}".format(score))
    return status, score
