import imp
import json
import sys


class ProgressText:
    def __init__(self):
        self.last_text = ''

    def text(self, text):
        sys.stdout.write('\b' * len(self.last_text))
        sys.stdout.write('\r')
        sys.stdout.write(text)
        sys.stdout.flush()

        self.last_text = text


def load_json(filename):
    with open(filename, 'r') as jsonfile:
        data = json.load(jsonfile, 'UTF-8')

    return data


def save_json(filename, data):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, sort_keys=True)


def load_python(filename):
    return imp.load_source('', filename).create()
