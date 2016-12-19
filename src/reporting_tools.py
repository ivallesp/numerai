__author__ = "ivallesp"

import os
import pandas_profiling

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_profiling_reports(version=None):
    """
    Generates a pandas-profiling based report for the current training and test set and store them in HTML in the
    corresponding reports path.
    :param version: Version of the data to generate the reports. If not specified, last version is chosen.
    """
    from src.common_paths import get_reports_version_path
    from src.file_loaders import load_numerai_data

    report_path = get_reports_version_path(version)
    df_train, df_test = load_numerai_data(version)
    report = pandas_profiling.ProfileReport(df_train)
    report.to_file(os.path.join(report_path, "profiling_report_train.html"))
    report = pandas_profiling.ProfileReport(df_test)
    report.to_file(os.path.join(report_path, "profiling_report_test.html"))


def generate_correlation_matrices(version=None):
    """
    Generates correlation matrices for the training and test set and stores them in the reports path corresponding to
    the version specified
    :param version: Version of the data to generate the reports. If not specified, last version is chosen.
    """
    from src.common_paths import get_reports_version_path
    from src.file_loaders import load_numerai_data

    sns.set(style="white")
    report_path = get_reports_version_path(version)
    df_train, df_test = load_numerai_data(version)

    corr_train = df_train.select_dtypes(["int64", "int32", "int16", "int8", "float"]).corr(method="spearman")
    # Generate a mask for the upper triangle
    mask_train = np.zeros_like(corr_train, dtype=np.bool)
    mask_train[np.triu_indices_from(mask_train, k=1)] = True

    corr_test = df_test.select_dtypes(["int64", "int32", "int16", "int8", "float"]).corr(method="spearman")
    # Generate a mask for the upper triangle
    mask_test = np.zeros_like(corr_test, dtype=np.bool)
    mask_test[np.triu_indices_from(mask_test, k=1)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(11, 9))
    ptr = sns.heatmap(corr_train, mask=mask_train, cmap=cmap,
                      square=True, linewidths=.5, ax=ax)
    plt.setp(ptr.get_xticklabels(), rotation=45)
    ax.collections[0].colorbar.set_ticks(np.arange(-1, 1.0001, 0.1))
    f.savefig(os.path.join(report_path, "correlation_matrix_train.png"))

    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(11, 9))
    pte = sns.heatmap(corr_test, mask=mask_test, cmap=cmap,
                      square=True, linewidths=.5, ax=ax)
    ax.collections[0].colorbar.set_ticks(np.arange(-1, 1.0001, 0.1))
    plt.setp(pte.get_xticklabels(), rotation=45)
    f.savefig(os.path.join(report_path, "correlation_matrix_test.png"))