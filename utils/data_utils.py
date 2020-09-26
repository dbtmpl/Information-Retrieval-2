import json


def load_json(path):
    with open(path) as f:
        qulac = json.load(f)
    return qulac
