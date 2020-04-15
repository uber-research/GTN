"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_backward_cpp',
      ext_modules=[cpp_extension.CppExtension('custom_backward_cpp', ['custom_backward.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
