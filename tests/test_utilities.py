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

    def test_prettify_json(self):
        dictionary = {"a":[1,2,3], "b":"something more"}
        expected_return = '{\n    "a": [\n        1,\n        2,\n        3\n    ],\n    "b": "something more"\n}'
        json_prettified = prettify_json(dictionary)
        print json_prettified
        print expected_return
        assert json_prettified == expected_return

