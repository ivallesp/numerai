from unittest import TestCase
from src.common_paths import *
from src.utilities import *
import json
import shutil

__author__ = "ivallesp"


class TestUtilities(TestCase):
    def test_utilities(self):
        version_from_function = get_last_data_version()
        with open("settings.json", "rb") as f:
            version_manual = json.load(f)["last_data_version"]
        assert version_manual == version_from_function
