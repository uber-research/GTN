"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from os.path import join
import json
from datetime import datetime
import torch


class TLogger(object):
    def __init__(self, name):
        self.name = name
        self.values = {}

    def info(self, *strings):
        datefmt = '%m/%d/%Y %I:%M:%S.%f %p'
        strings = [str(s) for s in strings]
        print('{asctime} {message}'.format(asctime=datetime.now().strftime(datefmt), message=' '.join(strings)))

    def get_dir(self):
        result = join("/tmp/gtn", self.name)
        os.makedirs(result, exist_ok=True)
        return result

    def record_tabular(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values[key] = value

    def dump_tabular(self):
        with open(join(self.get_dir(), "results.txt"), "a+") as file:
            file.write(json.dumps(self.values) + "\n")
            self.info("progress:", json.dumps(self.values))
            self.values.clear()

_tlogger = None
def CURRENT():
    global _tlogger
    if not _tlogger:
        _tlogger = TLogger()
    return _tlogger

def set_tlogger(*args, **kwargs):
    global _tlogger
    _tlogger = TLogger(*args, **kwargs)


def info(*args, **kwargs):
    return CURRENT().info(*args, **kwargs)


def get_dir(*args, **kwargs):
    return CURRENT().get_dir(*args, **kwargs)


def record_tabular(*args, **kwargs):
    return CURRENT().record_tabular(*args, **kwargs)


def dump_tabular(*args, **kwargs):
    return CURRENT().dump_tabular(*args, **kwargs)
