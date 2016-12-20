__author__ = "ivallesp"

import os
import json

def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def normalize_path(*args):
        return os.path.normpath(path(*args))
    return normalize_path


def _assure_path_exists(path):
    """
    Decorator function intended for checking the existence of a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def assure_exists(*args):
        assert os.path.exists(path(*args))
        return path(*args)
    return assure_exists


def _is_output_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an output path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence_or_create_it(*args):
        if not os.path.exists(path(*args)):
            "Path didn't exist... creating it: {}".format(path(*args))
            os.makedirs(path(*args))
        return path(*args)
    return check_existence_or_create_it


def _is_input_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an input path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence(*args):
        return path(*args)
    return check_existence


@_is_input_path
def get_project_path():
    """
    Function used for retrieving the path where the project is located
    :return: the checked path (str|unicode)
    """
    with open("./settings.json") as f:
        settings = json.load(f)
    return settings["project_path"]


@_is_input_path
def get_data_path():
    with open("./settings.json") as f:
        settings = json.load(f)
    return settings["data_path"]


@_is_input_path
def get_raw_data_path():
    return os.path.join(get_data_path(), "raw")


@_is_input_path
def get_numerai_secrets_path():
    return os.path.join(get_project_path(), "NumerAPI", "secrets.json")


@_is_output_path
def get_submissions_path():
    return os.path.join(get_data_path(), "submissions")


@_is_output_path
def get_reports_path():
    return os.path.join(get_data_path(), "reports")


@_is_input_path
def get_raw_data_version_path(version):
    """
    Retrieves the path where the raw data is saved.
    :param version: version of the data which is intended to be loaded. If not specified, the last version name is
    retrieved from the settings.json file and used to build the path (str|unicode|None).
    :return: the path of the data requested (str|unicode).
    """
    from utilities import get_last_data_version
    if not version:
        version = get_last_data_version()
    return os.path.join(get_raw_data_path(), version)


@_is_output_path
def get_reports_version_path(version):
    """
    Retrieves the path where the reports are going to be saved
    :param version: version of the data which is intended to be saved. If not specified, the last version name is
    retrieved from the settings.json file and used to build the path (str|unicode|None).
    :return: the path of the data requested (str|unicode).
    """
    from utilities import get_last_data_version
    if not version:
        version = get_last_data_version()
    return os.path.join(get_reports_path(), version)

@_is_output_path
def get_submissions_version_path(version):
    """
    Retrieves the path where the submissions are going to be saved
    :param version: version of the data which is intended to be saved. If not specified, the last version name is
    retrieved from the settings.json file and used to build the path (str|unicode|None).
    :return: the path of the data requested (str|unicode).
    """
    from utilities import get_last_data_version
    if not version:
        version = get_last_data_version()
    return os.path.join(get_submissions_path(), version)
