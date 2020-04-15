"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""

def map_structure(fn, structure):
    if isinstance(structure, (list, tuple)):
        return type(structure)([map_structure(fn, a) for a in structure])
    return fn(structure)


def pack_sequence_as(structure, flatten):
    if not hasattr(flatten, "__next__"):
        flatten = iter(flatten)

    result = _pack_sequence_as_impl(structure, flatten)
    try:
        extra = next(flatten)
        raise ValueError("Flatten input does not match structure (too many values)")
    except StopIteration:
        return result


def _pack_sequence_as_impl(structure, flatten):
    try:
        if isinstance(structure, (list, tuple)):
            return type(structure)([_pack_sequence_as_impl(a, flatten) for a in structure])
        return next(flatten)
    except StopIteration:
        raise ValueError("Flatten input does not match structure (not enough values)")


def flatten(structure):
    return list(_flatten(structure))


def _flatten(structure):
    if isinstance(structure, (list, tuple)):
        for a in structure:
            for b in _flatten(a):
                yield b
    else:
        yield structure
