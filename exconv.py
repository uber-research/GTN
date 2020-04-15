"""
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""
import time
import torch
from torch import nn
from torch.autograd import grad as grad_f
import torch.nn.functional as F
import custom_backward_cpp


class ExConv2d2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_shape, grad_output, input, padding, stride, dilation, groups):
        ctx.save_for_backward(grad_output, input)
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        return custom_backward_cpp.backward(weight_shape, grad_output, input, padding, stride, dilation, groups, False, True)

    @staticmethod
    def backward(ctx, second_grad_output):
        grad_output, input = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            grad_output_grad = ExConv2d.apply(input, second_grad_output, ctx.padding, ctx.stride, ctx.dilation, ctx.groups)
        else:
            grad_output_grad = None
        if ctx.needs_input_grad[2]:
            input_grad = F.grad.conv2d_input(input.shape, second_grad_output, grad_output, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation, groups=ctx.groups)
        else:
            input_grad = None
        return None, grad_output_grad, input_grad, None, None, None, None


class ExConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, padding, stride, dilation, groups):
        ctx.save_for_backward(input, weight)
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        return F.conv2d(input, weight, padding=padding, stride=stride, dilation=dilation, groups=groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            input_grad = F.grad.conv2d_input(input.shape, weight, grad_output, padding=ctx.padding, stride=ctx.stride, dilation=ctx.dilation, groups=ctx.groups)
        else:
            input_grad = None
        if ctx.needs_input_grad[1]:
            weight_grad = ExConv2d2.apply(weight.shape, grad_output, input, ctx.padding, ctx.stride, ctx.dilation, ctx.groups)
        else:
            weight_grad = None
        return input_grad, weight_grad, None, None, None, None
