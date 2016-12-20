from unittest import TestCase
from src.common_paths import *
from src.reporting_tools import *
import os
import shutil

__author__ = "ivallesp"


class TestReportingTools(TestCase):
    def test_generate_profiling_reports(self):
        generate_profiling_reports("demo")
        assert "profiling_report_train.html" in os.listdir(get_reports_version_path("demo"))
        assert "profiling_report_test.html" in os.listdir(get_reports_version_path("demo"))
        shutil.rmtree(get_reports_version_path("demo"))

    def test_generate_correlation_matrices(self):
        generate_correlation_matrices("demo")
        assert "correlation_matrix_train.png" in os.listdir(get_reports_version_path("demo"))
        assert "correlation_matrix_test.png" in os.listdir(get_reports_version_path("demo"))
        shutil.rmtree(get_reports_version_path("demo"))
