/*
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
*/

#include <torch/extension.h>

#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

at::Tensor backward_weight(
    c10::ArrayRef<long int> weight_size,
    const at::Tensor &grad_output,
    const at::Tensor &input,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic)
{

  return at::cudnn_convolution_backward_weight(
      weight_size,
      grad_output,
      input,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("backward", &backward_weight, "Conv2d backward cudnn");
}
