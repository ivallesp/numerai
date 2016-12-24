__author__ = "ivallesp"
import json
import logging


def get_last_data_version():
    logger = logging.getLogger(__name__)
    with open("./settings.json") as f:
        settings = json.load(f)
        version = settings["last_data_version"]
    logger.info("Retrieved last data version name: {}".format(version))
    return version


def prettify_json(dictionary, sort_keys=True, indent=4):
    """
    Takes a dictionary as an input and returns a string containing the JSON with correct format.
    :param dictionary: dictionary to be prettified (dict)
    :param sort_keys: indicates if the keys should be sorted (bool)
    :param indent: number of spaces to use as indentation (int)
    :return: prettified json (str)
    """
    report = json.dumps(dictionary, sort_keys=sort_keys, indent=indent, separators=(',', ': '))
    return report