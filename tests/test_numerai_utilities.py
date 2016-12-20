from unittest import TestCase
from src.numerai_utilities import *
from src.common_paths import *
import os
import shutil
import json
import pandas as pd
import numpy as np
import time
__author__ = "ivallesp"


class TestNumeraiUtilities(TestCase):
    def test_get_numerai_api_access(self):
        n_api = get_numerai_api_access()
        assert n_api.login()[3] == 201

    def test_download_last_numerai_data(self):
        with open("settings.json", 'rb') as f:
            settings_backup = f.read()
        version_name = "test"
        download_last_numerai_data(version_name)
        with open("settings.json", 'rb') as f:
            last_version_name = json.load(f)["last_data_version"]
        with open("settings.json", 'wb') as f:
            f.write(settings_backup)
        files = os.listdir(get_raw_data_version_path(version_name))
        assert len(files) > 0
        time.sleep(2)
        shutil.rmtree(get_raw_data_version_path(version_name))
        assert last_version_name == version_name

    def test_build_submission_from_lists(self):
        indices = ["1","2","3","4","5","6","7","8","9","10"]
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        build_submission(version="demo", indices=indices, probs=probs, alias="test_demo")
        df = pd.read_csv(os.path.join(get_submissions_version_path("demo"), "submission_test_demo.csv"),
                         sep=",", decimal=".", encoding="utf-8", index_col=False, dtype={"t_id":str})
        shutil.rmtree(os.path.join(get_submissions_version_path("demo")))
        assert df.shape == (10, 2)
        assert df.columns.tolist() == ["t_id", "probability"]
        assert df.equals(pd.DataFrame({"t_id": indices, "probability": probs})[["t_id", "probability"]])

    def test_build_submission_from_arrays(self):
        indices = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        build_submission(version="demo", indices=np.array(indices), probs=np.array(probs), alias="test_demo")
        df = pd.read_csv(os.path.join(get_submissions_version_path("demo"), "submission_test_demo.csv"),
                         sep=",", decimal=".", encoding="utf-8", index_col=False, dtype={"t_id": str})
        shutil.rmtree(os.path.join(get_submissions_version_path("demo")))
        assert df.shape == (10, 2)
        assert df.columns.tolist() == ["t_id", "probability"]
        assert df.equals(pd.DataFrame({"t_id": indices, "probability": probs})[["t_id", "probability"]])

    def test_build_submission_from_series(self):
        indices = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        build_submission(version="demo", indices=pd.Series(indices), probs=pd.Series(probs), alias="test_demo")
        df = pd.read_csv(os.path.join(get_submissions_version_path("demo"), "submission_test_demo.csv"),
                         sep=",", decimal=".", encoding="utf-8", index_col=False, dtype={"t_id": str})
        shutil.rmtree(os.path.join(get_submissions_version_path("demo")))
        assert df.shape == (10, 2)
        assert df.columns.tolist() == ["t_id", "probability"]
        assert df.equals(pd.DataFrame({"t_id": indices, "probability": probs})[["t_id", "probability"]])
