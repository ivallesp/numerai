import pandas as pd
import numpy as np
import logging

from src.logging_tools import setup_logging_environment
from src.file_loaders import load_numerai_data
from src.numerai_utilities import download_last_numerai_data
from src.reporting_tools import generate_correlation_matrices, generate_profiling_reports

__author__ = "ivallesp"


setup_logging_environment()
status = download_last_numerai_data()
generate_profiling_reports()
generate_correlation_matrices()

df_train, df_test = load_numerai_data()
df_dev = df_train.sample(frac=0.1, random_state=655321)
df_train = df_train.drop(df_dev.index)

1/0