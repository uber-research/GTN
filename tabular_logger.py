"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
from os.path import join
import json
from datetime import datetime


class TLogger(object):
    def __init__(self, name):
        self.name = name
        self.values = {}

    def info(self, *strings):
        datefmt = '%m/%d/%Y %I:%M:%S.%f %p'
        strings = [str(s) for s in strings]
        print('{asctime} {message}'.format(asctime=datetime.now().strftime(datefmt), message=' '.join(strings)))

    def get_dir():
        return join("/tmp/gtn", self.name)

    def record_tabular(self, key, value):
        self.values[key] = value

    def dump_tabular():
        with open(join(get_dir(), "results.txt")) as file:
            file.write(json.dumps(self.values) + "\n")
            self.info("progress:", json.dumps(self.values))
            self.values.clear()
