"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json


def main(input_data, t=100000):
    directory = os.path.dirname(input_data)
    entries = []
    with open(input_data, "r") as input_file:
        for i, line in enumerate(input_file):
            entries.append(json.loads(line))
    entries = sorted(entries, key=lambda row: row["validation_accuracy"], reverse=True)
    output_data = os.path.join(directory, f"top_{t}_" + os.path.basename(input_data))
    entries = entries[:t]
    with open(output_data, "w") as output_file:
        for i, row in enumerate(entries):
            print(json.dumps(row), file=output_file)
            with open(os.path.join(directory, "dag.{}.json".format(i + 1)), "w+") as dag_file:
                dag_file.write(row["dag"] + "\n")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
