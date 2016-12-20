from unittest import TestCase
from src.common_paths import *
import os
import shutil
__author__ = "ivallesp"


class TestCommonPaths(TestCase):
    def test_get_project_path(self):
        path = get_project_path()
        assert os.path.exists(path)
        assert path == os.path.abspath(".")

    def test_get_data_path(self):
        path = get_data_path()
        assert os.path.exists(path)

    def test_get_raw_data_path(self):
        path = get_raw_data_path()
        assert os.path.exists(path)

    def test_get_numerai_secrets_path(self):
        path = get_numerai_secrets_path()
        assert os.path.exists(path)

    def test_get_submissions_path(self):
        path = get_submissions_path()
        assert os.path.exists(path)

    def test_get_reports_path(self):
        path = get_reports_path()
        assert os.path.exists(path)

    def test_get_raw_data_version_path(self):
        path = get_raw_data_version_path("demo")
        assert os.path.exists(path)
        assert "example_predictions.csv" in os.listdir(path)
        assert "numerai_dataset_demo.zip" in os.listdir(path)
        assert "numerai_tournament_data.csv" in os.listdir(path)
        assert "numerai_training_data.csv" in os.listdir(path)

    def test_get_reports_version_path(self):
        path = get_reports_version_path("demo")
        assert os.path.exists(path)
        shutil.rmtree(path)
        assert not os.path.exists(path)

    def test_get_submissions_version_path(self):
        path = get_submissions_version_path("demo")
        assert os.path.exists(path)
        shutil.rmtree(path)
        assert not os.path.exists(path)
