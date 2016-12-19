__author__ = "ivallesp"
import json

def get_last_data_version():
    with open("./settings.json") as f:
        settings = json.load(f)
        version = settings["last_data_version"]
    return version