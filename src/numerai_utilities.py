import json
import os
import datetime

__author__ = "ivallesp"


def get_numerai_api_access():
    """
    Returns an authenticated object for accessing comfortably to the numerai API.
    :return: None (void)
    """
    from NumerAPI.numerapi import NumerAPI
    from src.common_paths import get_numerai_secrets_path
    secrets = json.load(open(get_numerai_secrets_path()))
    n_api = NumerAPI(**secrets)
    return n_api


def download_last_numerai_data():
    """
    Downloads the last version of the dataset from the numerai API, stores it in the data/raw folder and
    updates the settings.json file with the name of the version downloaded.
    :return: None (void)
    """
    from src.common_paths import get_raw_data_path
    n_api = get_numerai_api_access()
    data_version = datetime.datetime.now().strftime("%Y%m%d")
    path = os.path.normpath(os.path.join(get_raw_data_path(), data_version))
    if not os.path.exists(path):
        os.makedirs(path)
    n_api.download_current_dataset(dest_path=path)
    with open("settings.json", 'rb') as f:
        settings = json.load(f)
    settings["_last_data_version"] = data_version
    with open("settings.json", 'wb') as f:
        f.write(json.dumps(settings, sort_keys=True, indent=4, separators=(',', ': ')))
