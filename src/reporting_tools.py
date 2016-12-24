import os
import pandas_profiling
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "ivallesp"


def generate_profiling_reports(version=None):
    """
    Generates a pandas-profiling based report for the current training and test set and store them in HTML in the
    corresponding reports path.
    :param version: Version of the data to generate the reports. If not specified, last version is chosen.
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested profiling report generation")
    from src.common_paths import get_reports_version_path
    from src.file_loaders import load_numerai_data

    if not version:
        logger.info("Using last version of the data")
    else:
        logger.info("Using data version: {0}".format(version))
    report_path = get_reports_version_path(version)
    logger.info("Loading data...")
    df_train, df_test = load_numerai_data(version)
    logger.info("Data loaded successfully!")
    logger.info("Generating pandas profiling report for training set...")
    report_tr = pandas_profiling.ProfileReport(df_train)
    logger.info("Generating pandas profiling report for test set...")
    report_te = pandas_profiling.ProfileReport(df_test)
    logger.info("Reports generated successfully. Storing them.")
    report_tr.to_file(os.path.join(report_path, "profiling_report_train.html"))
    report_te.to_file(os.path.join(report_path, "profiling_report_test.html"))
    logger.info("Reports stored in {0}".format(report_path))


def generate_correlation_matrices(version=None):
    """
    Generates correlation matrices for the training and test set and stores them in the reports path corresponding to
    the version specified
    :param version: Version of the data to generate the reports. If not specified, last version is chosen.
    """
    logger = logging.getLogger(__name__)
    logger.info("Requested correlation matrices generation")
    from src.common_paths import get_reports_version_path
    from src.file_loaders import load_numerai_data

    sns.set(style="white")
    report_path = get_reports_version_path(version)
    logger.info("Loading data...")
    df_train, df_test = load_numerai_data(version)
    logger.info("Data loaded successfully!")

    logger.info("Calculating training set correlation matrix...")
    corr_train = df_train.select_dtypes(["int64", "int32", "int16", "int8", "float"]).corr(method="spearman")
    # Generate a mask for the upper triangle
    mask_train = np.zeros_like(corr_train, dtype=np.bool)
    mask_train[np.triu_indices_from(mask_train, k=1)] = True

    logger.info("Calculating test set correlation matrix...")
    corr_test = df_test.select_dtypes(["int64", "int32", "int16", "int8", "float"]).corr(method="spearman")
    # Generate a mask for the upper triangle
    mask_test = np.zeros_like(corr_test, dtype=np.bool)
    mask_test[np.triu_indices_from(mask_test, k=1)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    logger.info("Generating the training plot...")
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(11, 9))
    ptr = sns.heatmap(corr_train, mask=mask_train, cmap=cmap,
                      square=True, linewidths=.5, ax=ax)
    plt.setp(ptr.get_xticklabels(), rotation=45)
    ax.collections[0].colorbar.set_ticks(np.arange(-1, 1.0001, 0.1))
    f.savefig(os.path.join(report_path, "correlation_matrix_train.png"))
    logger.info("Plot generated and stored successfully in: {0}".format(report_path))

    logger.info("Generating the test plot...")
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(11, 9))
    pte = sns.heatmap(corr_test, mask=mask_test, cmap=cmap,
                      square=True, linewidths=.5, ax=ax)
    ax.collections[0].colorbar.set_ticks(np.arange(-1, 1.0001, 0.1))
    plt.setp(pte.get_xticklabels(), rotation=45)
    f.savefig(os.path.join(report_path, "correlation_matrix_test.png"))
    logger.info("Plot generated and stored successfully in: {0}".format(report_path))
