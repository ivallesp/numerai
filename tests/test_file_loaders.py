from unittest import TestCase
from src.file_loaders import *

__author__ = "ivallesp"


class TestFileLoaders(TestCase):
    def test_load_train_data(self):
        df = load_train_data(version="demo")
        assert df.shape == (99, 22)

    def test_load_tournament_data(self):
        df = load_tournament_data(version="demo")
        assert df.shape == (99, 22)

    def test_load_numerai_data(self):
        df_train, df_tournament = load_numerai_data("demo")
        df_train_well = load_train_data("demo")
        df_tournament_well = load_tournament_data("demo")
        assert df_train.equals(df_train_well)
        assert df_tournament.equals(df_tournament_well)
